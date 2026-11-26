from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..config import StructureKind
from ..tokenizer import NumericTokenizer
from .config import TrainingSpec
from .data import DatasetBundle, TokenDataset, build_dataset, collate_examples
from .transformer import TopologyTransformer


class TrainingError(RuntimeError):
    pass


@dataclass
class TrainingResult:
    history: list[dict[str, Any]]
    best_val_loss: float
    best_checkpoint: str
    status: str = "completed"
    samples: dict[str, str] = field(default_factory=dict)


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise TrainingError("CUDA requested but not available")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise TrainingError("MPS requested but not available")
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _masked_ce(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = input_ids[:, 1:]
    mask = loss_mask[:, :-1]
    losses = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    masked = losses[mask]
    if masked.numel() == 0:
        return logits.sum() * 0.0, torch.tensor(0.0, device=logits.device)
    accuracy = (logits[:, :-1].argmax(dim=-1)[mask] == targets[mask]).float().mean()
    return masked.mean(), accuracy


def _logprob_from_logits(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    gathered = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    mask = loss_mask[:, :-1]
    return (gathered * mask).sum(dim=1)


def _pad_rows(rows: list[list[int]], pad_id: int) -> torch.Tensor:
    width = max(len(row) for row in rows)
    tensor = torch.full((len(rows), width), pad_id, dtype=torch.long)
    for index, row in enumerate(rows):
        tensor[index, : len(row)] = torch.tensor(row, dtype=torch.long)
    return tensor


def _answer_mask(prefix_length: int, total_length: int) -> list[bool]:
    mask = [False] * total_length
    for index in range(prefix_length - 1, total_length - 1):
        mask[index] = True
    return mask


class Trainer:
    def __init__(
        self,
        spec: TrainingSpec,
        *,
        tokenizer: NumericTokenizer,
        checkpoint_dir: Path,
        bundle: DatasetBundle | None = None,
    ) -> None:
        self.spec = spec
        self.tokenizer = tokenizer
        self.checkpoint_dir = checkpoint_dir
        self.bundle = bundle or build_dataset(tokenizer, spec.data)
        self.device = resolve_device(spec.training.device)
        self.pad_id = tokenizer.pad_id
        self.eos_id = tokenizer.eos_id
        self.model = TopologyTransformer(
            spec.model,
            vocab_size=tokenizer.vocab_size,
            n_structures=len(StructureKind),
            max_dim=self.bundle.max_dim,
        ).to(self.device)
        self.history: list[dict[str, Any]] = []
        self.best_val_loss = float("inf")
        self.best_checkpoint = checkpoint_dir / "best.pt"
        self.samples: dict[str, str] = {}
        self._fixed_infer = next(
            (example for example in self.bundle.val if example.task == "infer"),
            self.bundle.val[0],
        )
        self._fixed_generate = next(
            (example for example in self.bundle.val if example.task == "generate"),
            self.bundle.val[0],
        )
        self._rng = random.Random(spec.training.seed)

    def _optimizer(self) -> torch.optim.Optimizer:
        config = self.spec.training
        parameters = self.model.parameters()
        if config.optimizer == "adamw":
            return torch.optim.AdamW(parameters, lr=config.lr, weight_decay=config.weight_decay)
        if config.optimizer == "adam":
            return torch.optim.Adam(parameters, lr=config.lr, weight_decay=config.weight_decay)
        return torch.optim.SGD(
            parameters,
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay,
        )

    def _scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int):
        config = self.spec.training
        warmup = max(1, config.warmup_steps)

        def schedule(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)

    def _aux_loss(
        self,
        output_hidden: torch.Tensor,
        batch: dict[str, Any],
        tasks: list[str],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mask = torch.tensor([task == "infer" for task in tasks], device=self.device)
        if not bool(mask.any()):
            zero = output_hidden.sum() * 0.0
            return zero, {}
        boundary = batch["boundary_index"].to(self.device)
        selected_hidden = output_hidden[torch.arange(output_hidden.shape[0]), boundary][mask]
        aux = self.model.aux_forward(selected_hidden)
        structure = batch["aux_structure"].to(self.device)[mask]
        genus = batch["aux_genus"].to(self.device)[mask]
        components = batch["aux_components"].to(self.device)[mask]
        betti = batch["aux_betti"].to(self.device)[mask]
        ph = batch["aux_ph"].to(self.device)[mask]
        structure_loss = F.cross_entropy(aux["structure_logits"], structure)
        genus_loss = F.mse_loss(aux["genus"], genus)
        components_loss = F.mse_loss(aux["n_components"], components)
        betti_loss = F.mse_loss(aux["betti"], betti)
        ph_loss = F.mse_loss(aux["ph"], ph)
        total = structure_loss + genus_loss + components_loss + betti_loss + ph_loss
        metrics = {
            "aux_structure_ce": float(structure_loss.detach().cpu()),
            "aux_genus_mse": float(genus_loss.detach().cpu()),
            "aux_components_mse": float(components_loss.detach().cpu()),
            "aux_betti_mse": float(betti_loss.detach().cpu()),
            "aux_ph_mse": float(ph_loss.detach().cpu()),
        }
        return total, metrics

    def _ranking_loss(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        tasks = batch["tasks"]
        indices = [index for index, task in enumerate(tasks) if task == "infer"]
        if len(indices) < 2:
            return logits.sum() * 0.0
        correct_logp = _logprob_from_logits(logits, input_ids, loss_mask)
        targets = batch["target_ids"]
        prefixes = batch["prefix_ids"]
        negative_targets = [targets[(index + 1) % len(targets)] for index in indices]
        rows = [
            [*prefixes[index], *negative_targets[offset], self.eos_id]
            for offset, index in enumerate(indices)
        ]
        neg_input_ids = _pad_rows(rows, self.pad_id).to(self.device)
        neg_mask = torch.zeros_like(neg_input_ids, dtype=torch.bool)
        for row, index in enumerate(indices):
            length = len(rows[row])
            mask = _answer_mask(len(prefixes[index]), length)
            neg_mask[row, :length] = torch.tensor(mask, dtype=torch.bool)
        with torch.no_grad():
            neg_logits = self.model(neg_input_ids).logits
        neg_logp = _logprob_from_logits(neg_logits, neg_input_ids, neg_mask)
        correct_selected = correct_logp[torch.tensor(indices, device=self.device)]
        margin = self.spec.training.ranking_margin
        return F.relu(margin - (correct_selected - neg_logp)).mean()

    @torch.no_grad()
    def _evaluate(self) -> dict[str, float]:
        self.model.eval()
        loader = DataLoader(
            TokenDataset(self.bundle.val),
            batch_size=self.spec.training.batch_size,
            shuffle=False,
            collate_fn=partial(collate_examples, pad_id=self.pad_id),
        )
        total_loss = 0.0
        total_ce = 0.0
        total_acc = 0.0
        total_aux = 0.0
        batches = 0
        correct_mc = 0
        total_mc = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            loss_mask = batch["loss_mask"].to(self.device)
            output = self.model(input_ids)
            ce, accuracy = _masked_ce(output.logits, input_ids, loss_mask)
            aux, _ = self._aux_loss(output.final_hidden, batch, batch["tasks"])
            total_loss += float((ce + self.spec.training.aux_weight * aux).detach().cpu())
            total_ce += float(ce.detach().cpu())
            total_acc += float(accuracy.detach().cpu())
            total_aux += float(aux.detach().cpu())
            batches += 1
            mc = self._multiple_choice_batch(output.logits, input_ids, loss_mask, batch)
            correct_mc += mc[0]
            total_mc += mc[1]
        self.model.train()
        metrics = {
            "val_loss": total_loss / max(1, batches),
            "val_ce": total_ce / max(1, batches),
            "val_token_accuracy": total_acc / max(1, batches),
            "val_aux_loss": total_aux / max(1, batches),
        }
        if total_mc:
            metrics["val_multiple_choice"] = correct_mc / total_mc
        return metrics

    @torch.no_grad()
    def _multiple_choice_batch(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        batch: dict[str, Any],
    ) -> tuple[int, int]:
        indices = [index for index, task in enumerate(batch["tasks"]) if task == "infer"]
        if len(indices) < 2:
            return 0, 0
        correct_logp = _logprob_from_logits(logits, input_ids, loss_mask)
        targets = batch["target_ids"]
        prefixes = batch["prefix_ids"]
        correct = 0
        total = 0
        for index in indices:
            candidates = [correct_logp[index]]
            for offset in range(1, self.spec.training.mc_negatives + 1):
                target = targets[(index + offset) % len(targets)]
                row = [*prefixes[index], *target, self.eos_id]
                row_tensor = _pad_rows([row], self.pad_id).to(self.device)
                mask = torch.zeros_like(row_tensor, dtype=torch.bool)
                mask[0, : len(row)] = torch.tensor(
                    _answer_mask(len(prefixes[index]), len(row)), dtype=torch.bool
                )
                neg_logits = self.model(row_tensor).logits
                candidates.append(_logprob_from_logits(neg_logits, row_tensor, mask)[0])
            total += 1
            if int(torch.stack(candidates).argmax()) == 0:
                correct += 1
        return correct, total

    @torch.no_grad()
    def _sample_generation(self) -> dict[str, str]:
        self.model.eval()
        output: dict[str, str] = {}
        for task, example in (
            ("infer", self._fixed_infer),
            ("generate", self._fixed_generate),
        ):
            prompt = torch.tensor([example.prefix_ids], dtype=torch.long, device=self.device)
            generated = self.model.generate(
                prompt,
                max_new_tokens=self.spec.training.max_new_tokens,
                eos_id=self.eos_id,
            )
            continuation = generated[0, len(example.prefix_ids) :].tolist()
            output[task] = self.tokenizer.decode(continuation)[:1000]
        self.model.train()
        return output

    def _save_checkpoint(self, path: Path, *, epoch: int, metrics: dict[str, float]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_spec": self.spec.model.model_dump(mode="json"),
                "data_spec": self.spec.data.model_dump(mode="json"),
                "train_spec": self.spec.training.model_dump(mode="json"),
                "vocab": self.tokenizer.tokens,
                "max_dim": self.bundle.max_dim,
                "n_structures": len(StructureKind),
                "epoch": epoch,
                "metrics": metrics,
            },
            path,
        )

    def run(
        self,
        *,
        callback: Callable[[dict[str, Any]], None] | None = None,
        stop_event=None,
    ) -> TrainingResult:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(
            TokenDataset(self.bundle.train),
            batch_size=self.spec.training.batch_size,
            shuffle=True,
            collate_fn=partial(collate_examples, pad_id=self.pad_id),
        )
        optimizer = self._optimizer()
        total_steps = max(1, len(loader) * self.spec.training.epochs)
        scheduler = self._scheduler(optimizer, total_steps)
        self.model.train()
        global_step = 0
        stale_epochs = 0
        status = "completed"
        for epoch in range(1, self.spec.training.epochs + 1):
            epoch_loss = 0.0
            epoch_ce = 0.0
            epoch_aux = 0.0
            epoch_rank = 0.0
            batches = 0
            for batch in loader:
                if stop_event is not None and stop_event.is_set():
                    status = "stopped"
                    break
                input_ids = batch["input_ids"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                output = self.model(input_ids)
                ce, _ = _masked_ce(output.logits, input_ids, loss_mask)
                aux, aux_metrics = self._aux_loss(output.final_hidden, batch, batch["tasks"])
                rank = self._ranking_loss(output.logits, input_ids, loss_mask, batch)
                loss = (
                    ce
                    + self.spec.training.aux_weight * aux
                    + self.spec.training.ranking_weight * rank
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.spec.training.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.spec.training.grad_clip)
                optimizer.step()
                scheduler.step()
                global_step += 1
                batches += 1
                epoch_loss += float(loss.detach().cpu())
                epoch_ce += float(ce.detach().cpu())
                epoch_aux += float(aux.detach().cpu())
                epoch_rank += float(rank.detach().cpu())
                if callback is not None:
                    callback(
                        {
                            "type": "batch",
                            "epoch": epoch,
                            "step": global_step,
                            "total_steps": total_steps,
                            "loss": epoch_loss / batches,
                            "ce": epoch_ce / batches,
                            "aux": epoch_aux / batches,
                            "ranking": epoch_rank / batches,
                            "lr": scheduler.get_last_lr()[0],
                            **aux_metrics,
                        }
                    )
            if status == "stopped":
                break
            train_metrics = {
                "epoch": epoch,
                "step": global_step,
                "train_loss": epoch_loss / max(1, batches),
                "train_ce": epoch_ce / max(1, batches),
                "train_aux": epoch_aux / max(1, batches),
                "train_ranking": epoch_rank / max(1, batches),
                "lr": scheduler.get_last_lr()[0],
            }
            val_metrics = self._evaluate()
            samples = self._sample_generation()
            self.samples = samples
            metrics = {**train_metrics, **val_metrics, "samples": samples}
            self.history.append(metrics)
            improved = val_metrics["val_loss"] < self.best_val_loss - 1e-6
            if improved:
                self.best_val_loss = val_metrics["val_loss"]
                self._save_checkpoint(self.best_checkpoint, epoch=epoch, metrics=metrics)
                stale_epochs = 0
            else:
                stale_epochs += 1
            self._save_checkpoint(self.checkpoint_dir / "last.pt", epoch=epoch, metrics=metrics)
            if callback is not None:
                callback({"type": "epoch", **metrics})
            if stale_epochs >= self.spec.training.patience:
                status = "early_stopped"
                break
        self._save_history()
        if not self.best_checkpoint.exists():
            self._save_checkpoint(
                self.best_checkpoint,
                epoch=len(self.history),
                metrics=self.history[-1] if self.history else {},
            )
        return TrainingResult(
            history=self.history,
            best_val_loss=self.best_val_loss,
            best_checkpoint=str(self.best_checkpoint),
            status=status,
            samples=self.samples,
        )

    def _save_history(self) -> None:
        path = self.checkpoint_dir / "history.json"
        path.write_text(json.dumps(self.history, indent=2), encoding="utf-8")


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> tuple[TopologyTransformer, NumericTokenizer, dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    tokenizer = NumericTokenizer()
    if list(tokenizer.tokens) != list(checkpoint["vocab"]):
        raise TrainingError("checkpoint tokenizer vocabulary does not match this codebase")
    from .config import ModelSpec

    model_spec = ModelSpec.model_validate(checkpoint["model_spec"])
    model = TopologyTransformer(
        model_spec,
        vocab_size=tokenizer.vocab_size,
        n_structures=checkpoint["n_structures"],
        max_dim=checkpoint["max_dim"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer, checkpoint
