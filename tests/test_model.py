from __future__ import annotations

import torch

from llm_topology.model.config import InitScheme, ModelSpec, PositionEncoding
from llm_topology.model.transformer import TopologyTransformer
from llm_topology.tokenizer import NumericTokenizer

VOCAB = NumericTokenizer().vocab_size


def _model(**overrides) -> TopologyTransformer:
    defaults = dict(
        max_seq_len=64,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        dropout=0.0,
        position_encoding=PositionEncoding.LEARNED,
    )
    defaults.update(overrides)
    tokenizer = NumericTokenizer()
    return TopologyTransformer(
        ModelSpec(**defaults),
        vocab_size=tokenizer.vocab_size,
        n_structures=12,
        max_dim=2,
    )


def test_forward_shapes_and_internals() -> None:
    model = _model()
    input_ids = torch.randint(0, VOCAB, (3, 20))
    output = model(input_ids, return_attentions=True, return_ffn=True)
    assert output.logits.shape == (3, 20, VOCAB)
    assert len(output.hidden_states) == 3
    assert len(output.attentions) == 2
    assert output.attentions[0].shape == (3, 2, 20, 20)
    assert output.ffn_activations[0].shape == (3, 20, 64)
    for attention in output.attentions:
        assert torch.allclose(attention.sum(dim=-1), torch.ones_like(attention.sum(dim=-1)))


def test_causal_mask_blocks_future_tokens() -> None:
    model = _model()
    model.eval()
    input_ids = torch.randint(0, VOCAB, (1, 16))
    changed = input_ids.clone()
    changed[0, 10:] = torch.randint(0, VOCAB, (6,))
    with torch.no_grad():
        first = model(input_ids).logits
        second = model(changed).logits
    assert torch.allclose(first[:, :10], second[:, :10], atol=1e-5)


def test_init_is_reproducible() -> None:
    first = _model(init={"scheme": InitScheme.NORMAL, "std": 0.02, "seed": 123})
    second = _model(init={"scheme": InitScheme.NORMAL, "std": 0.02, "seed": 123})
    for left, right in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.equal(left, right)


def test_auxiliary_heads() -> None:
    model = _model()
    hidden = torch.randn(4, 32)
    aux = model.aux_forward(hidden)
    assert aux["structure_logits"].shape == (4, 12)
    assert aux["genus"].shape == (4,)
    assert aux["n_components"].shape == (4,)
    assert aux["betti"].shape == (4, 3)
    assert aux["ph"].shape == (4, 3)


def test_parameter_count_is_positive() -> None:
    assert _model().parameter_count() > 0


def test_generate_stops_at_eos() -> None:
    model = _model()
    prompt = torch.zeros((1, 4), dtype=torch.long)
    output = model.generate(prompt, max_new_tokens=5, eos_id=2)
    assert output.shape[1] <= 9
    assert torch.equal(output[:, :4], prompt)
