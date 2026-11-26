from __future__ import annotations

import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from .. import __version__
from ..config import (
    BatchDatasetResponse,
    CoefficientField,
    DatasetResponse,
    DatasetSpec,
    FiltrationMethod,
    StructureKind,
    TargetSchema,
    TransformSpec,
    all_target_blocks,
    optional_target_blocks,
)
from ..datasets import STORE as DATASET_STORE
from ..datasets import DatasetStoreError
from ..experiment import ExperimentConfig, config_from_training_spec
from ..limits import estimate_dataset, estimate_generation, estimate_training
from ..model.config import AnalysisSpec, DataBuildSpec, ModelSpec, TrainConfig, TrainingSpec
from ..pipeline import PipelineError, generate_dataset
from ..preflight import system_snapshot
from ..tokenizer import (
    DEFAULT_BOUNDARY,
    NumericTokenizer,
    TargetParseError,
    TokenizerError,
    parse_target_groups,
    serialize_target_groups,
)
from ..topology.generators import STRUCTURE_IDS, STRUCTURE_LABELS
from .jobs import MANAGER, JobError
from .state import STORE

PACKAGE_ROOT = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_ROOT / "static"
TEMPLATES_DIR = PACKAGE_ROOT / "templates"

app = FastAPI(
    title="LLM Topology Lab",
    version=__version__,
    description="A local observatory for topological representations in a custom transformer.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)
TOKENIZER = NumericTokenizer()


class TokenizeRequest(BaseModel):
    text: str = Field(min_length=0, max_length=2_000_000)
    strict: bool = True


class TokenizeResponse(BaseModel):
    text: str
    tokens: list[str]
    ids: list[int]
    vocab_size: int


class ParseTargetRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    text: str = Field(min_length=0, max_length=100_000)
    target_schema: TargetSchema = Field(default_factory=TargetSchema, alias="schema")


class ParseTargetResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    groups: list[list[float]]
    target_text: str
    target_schema: TargetSchema = Field(alias="schema")


class CatalogResponse(BaseModel):
    phase: str
    version: str
    structures: list[dict[str, object]]
    default_spec: dict[str, object]
    target_blocks: list[dict[str, object]]
    tokens: list[str]
    boundary_token: str
    methods: list[str]
    fields: list[str]
    model_defaults: dict[str, object]
    data_defaults: dict[str, object]
    training_defaults: dict[str, object]
    analysis_defaults: dict[str, object]


class DatasetCreateRequest(BaseModel):
    name: str = Field(default="dataset", min_length=1, max_length=80)
    data_spec: DataBuildSpec


class ConfigExportRequest(BaseModel):
    name: str = "experiment"
    spec: TrainingSpec
    notes: str = ""
    run_analysis: bool = True


class ConfigImportRequest(BaseModel):
    yaml: str


class BatchGenerateRequest(BaseModel):
    spec: DatasetSpec = Field(default_factory=DatasetSpec)
    count: int = Field(default=6, ge=1, le=24)
    mode: str = Field(default="single", pattern="^(single|mix)$")
    structures: list[StructureKind] | None = None
    vary_transform: bool = False
    variation_strength: float = Field(default=0.35, ge=0, le=1)


def _random_batch_transform(seed: int, strength: float) -> TransformSpec:
    rng = np.random.default_rng(seed)
    return TransformSpec(
        rotation_deg=tuple(rng.uniform(-180, 180, size=3).tolist()),
        stretch=tuple(rng.uniform(1 - 0.2 * strength, 1 + 0.2 * strength, size=3).tolist()),
        twist=float(rng.uniform(-0.65, 0.65) * strength),
        shear=tuple((rng.uniform(-0.4, 0.4, size=3) * strength).tolist()),
        bend=float(rng.uniform(-0.45, 0.45) * strength),
        warp=float(rng.uniform(-0.4, 0.4) * strength),
        noise=0.0,
        permute=True,
        seed=seed + 7,
    )


def _catalog() -> CatalogResponse:
    structures = [
        {
            "value": kind.value,
            "label": STRUCTURE_LABELS[kind],
            "id": STRUCTURE_IDS[kind],
            "needs_ambient": 4 if kind is StructureKind.KLEIN_BOTTLE else 3,
        }
        for kind in StructureKind
    ]
    blocks = [
        {"name": block.name, "fields": block.fields, "optional": False}
        for block in all_target_blocks(max_dim=2)
    ]
    blocks.extend(
        {"name": block.name, "fields": block.fields, "optional": True}
        for block in optional_target_blocks()
    )
    return CatalogResponse(
        phase="Phase 1/2 - Model & Interpretability",
        version=__version__,
        structures=structures,
        default_spec=DatasetSpec().model_dump(mode="json"),
        target_blocks=blocks,
        tokens=list(TOKENIZER.tokens),
        boundary_token=DEFAULT_BOUNDARY,
        methods=[method.value for method in FiltrationMethod],
        fields=[field.value for field in CoefficientField],
        model_defaults=ModelSpec().model_dump(mode="json"),
        data_defaults=DataBuildSpec().model_dump(mode="json"),
        training_defaults=TrainConfig().model_dump(mode="json"),
        analysis_defaults=AnalysisSpec().model_dump(mode="json"),
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"catalog": _catalog().model_dump(mode="json")},
    )


@app.get("/api/health")
async def health() -> dict[str, object]:
    import platform

    import gudhi
    import numpy
    import torch

    snapshot = system_snapshot()
    return {
        "status": "ok",
        "version": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "gudhi": gudhi.__version__,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "phase": "Phase 1/2 - Model & Interpretability",
        **snapshot,
    }


@app.get("/api/catalog", response_model=CatalogResponse)
async def catalog() -> CatalogResponse:
    return _catalog()


@app.post("/api/generate", response_model=DatasetResponse)
async def generate(spec: DatasetSpec) -> DatasetResponse:
    try:
        return generate_dataset(spec)
    except PipelineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/generate/batch", response_model=BatchDatasetResponse)
async def generate_batch(payload: BatchGenerateRequest) -> BatchDatasetResponse:
    plausibility = estimate_generation(payload.spec, count=payload.count)
    if plausibility["level"] == "block":
        raise HTTPException(status_code=422, detail="; ".join(plausibility["warnings"]))
    items: list[DatasetResponse] = []
    warnings: list[str] = []
    if payload.mode == "mix":
        if not payload.structures:
            raise HTTPException(
                status_code=422,
                detail="mixed batch mode requires at least one structure",
            )
        structures = payload.structures
    else:
        structures = [payload.spec.structure]
    for index in range(payload.count):
        structure = structures[index % len(structures)]
        seed = payload.spec.seed + index
        transform = (
            _random_batch_transform(seed, payload.variation_strength)
            if payload.vary_transform
            else payload.spec.transform
        )
        spec = payload.spec.model_copy(
            update={
                "seed": seed,
                "structure": structure,
                "transform": transform,
            }
        )
        try:
            items.append(generate_dataset(spec))
        except (PipelineError, ValueError) as exc:
            warnings.append(f"Cloud {index + 1} failed: {exc}")
    if not items:
        raise HTTPException(
            status_code=422, detail="; ".join(warnings) or "batch generation failed"
        )
    return BatchDatasetResponse(
        items=items,
        warnings=warnings,
        count=len(items),
        mode=payload.mode,
        structures=[structure.value for structure in structures],
    )


@app.post("/api/tokenize", response_model=TokenizeResponse)
async def tokenize(payload: TokenizeRequest) -> TokenizeResponse:
    try:
        tokens = TOKENIZER.tokenize(payload.text, strict=payload.strict)
        ids = [TOKENIZER.token_to_id[token] for token in tokens]
    except TokenizerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TokenizeResponse(
        text=payload.text,
        tokens=tokens,
        ids=ids,
        vocab_size=TOKENIZER.vocab_size,
    )


@app.post("/api/targets/parse", response_model=ParseTargetResponse)
async def parse_targets(payload: ParseTargetRequest) -> ParseTargetResponse:
    try:
        groups = parse_target_groups(payload.text, payload.target_schema)
        canonical = serialize_target_groups(groups)
    except TargetParseError as exc:
        detail: dict[str, object] = {"message": str(exc)}
        if exc.expected is not None:
            detail["expected"] = exc.expected
        if exc.actual is not None:
            detail["actual"] = exc.actual
        raise HTTPException(status_code=422, detail=detail) from exc
    except TokenizerError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ParseTargetResponse(
        groups=groups,
        target_text=canonical,
        target_schema=payload.target_schema,
    )


@app.post("/api/config/export")
async def export_config(payload: ConfigExportRequest) -> dict[str, object]:
    config = config_from_training_spec(
        payload.spec,
        name=payload.name,
        notes=payload.notes,
        run_analysis=payload.run_analysis,
    )
    return {"yaml": config.to_yaml(), "json": config.model_dump(mode="json")}


@app.post("/api/config/import")
async def import_config(payload: ConfigImportRequest) -> dict[str, object]:
    try:
        config = ExperimentConfig.from_yaml(payload.yaml)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Config ungueltig: {exc}") from exc
    return {"config": config.model_dump(mode="json")}


@app.post("/api/estimate/generate")
async def estimate_generate(payload: BatchGenerateRequest) -> dict[str, object]:
    return estimate_generation(payload.spec, count=payload.count)


@app.post("/api/estimate/dataset")
async def estimate_dataset_endpoint(payload: DatasetCreateRequest) -> dict[str, object]:
    return estimate_dataset(payload.data_spec)


@app.post("/api/estimate/training")
async def estimate_training_endpoint(spec: TrainingSpec) -> dict[str, object]:
    meta = None
    if spec.dataset_id:
        try:
            meta = DATASET_STORE.get_meta(spec.dataset_id)
        except DatasetStoreError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return estimate_training(spec, dataset_meta=meta)


@app.get("/api/datasets")
async def list_datasets() -> list[dict[str, object]]:
    return DATASET_STORE.list()


@app.post("/api/datasets")
async def create_dataset(payload: DatasetCreateRequest) -> dict[str, object]:
    plausibility = estimate_dataset(payload.data_spec)
    if plausibility["level"] == "block":
        raise HTTPException(status_code=422, detail="; ".join(plausibility["warnings"]))
    try:
        return DATASET_STORE.save(payload.name, payload.data_spec)
    except DatasetStoreError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/api/datasets/{dataset_id}/export")
async def export_dataset(dataset_id: str) -> FileResponse:
    try:
        DATASET_STORE.get_meta(dataset_id)
    except DatasetStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    directory = DATASET_STORE.root / dataset_id
    tmp = Path(tempfile.mkdtemp(prefix="topology-dataset-"))
    archive = tmp / f"{dataset_id}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in directory.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(DATASET_STORE.root))
    return FileResponse(archive, filename=f"{dataset_id}.zip", media_type="application/zip")


@app.post("/api/datasets/import")
async def import_dataset(file: Annotated[UploadFile, File()]) -> dict[str, object]:
    tmp = Path(tempfile.mkdtemp(prefix="topology-dataset-import-"))
    archive = tmp / "upload.zip"
    with archive.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (tmp / member.filename).resolve()
                if not target.is_relative_to(tmp.resolve()):
                    raise HTTPException(status_code=422, detail="ZIP enthaelt unsichere Pfade.")
            bundle.extractall(tmp)
        metas = list(tmp.rglob("meta.json"))
        if not metas:
            raise HTTPException(status_code=422, detail="ZIP enthaelt keine meta.json.")
        source = metas[0].parent
        dataset_id = source.name
        target = DATASET_STORE.root / dataset_id
        if target.exists():
            raise HTTPException(status_code=409, detail=f"Dataset {dataset_id} existiert bereits.")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, target)
        return {"imported": dataset_id, "path": str(target)}
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=422, detail="Datei ist kein gueltiges ZIP.") from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.get("/api/datasets/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: str,
    split: str = "train",
    limit: int = 6,
) -> dict[str, object]:
    if split not in {"train", "val", "test"}:
        raise HTTPException(status_code=422, detail="split must be train, val or test")
    try:
        return DATASET_STORE.preview(dataset_id, split=split, limit=min(max(limit, 1), 24))
    except DatasetStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str) -> dict[str, object]:
    try:
        return DATASET_STORE.get_meta(dataset_id)
    except DatasetStoreError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/train/start")
async def start_training(spec: TrainingSpec) -> dict[str, object]:
    meta = None
    if spec.dataset_id:
        try:
            meta = DATASET_STORE.get_meta(spec.dataset_id)
        except DatasetStoreError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    plausibility = estimate_training(spec, dataset_meta=meta)
    if plausibility["level"] == "block":
        raise HTTPException(status_code=422, detail="; ".join(plausibility["warnings"]))
    job = MANAGER.start(spec)
    return job.snapshot(include_history=False)


@app.get("/api/train/list")
async def list_training_jobs() -> list[dict[str, object]]:
    return MANAGER.list()


@app.get("/api/train/status/{job_id}")
async def training_status(job_id: str) -> dict[str, object]:
    try:
        return MANAGER.get(job_id).snapshot(include_history=True)
    except JobError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/train/stop/{job_id}")
async def stop_training(job_id: str) -> dict[str, object]:
    try:
        job = MANAGER.get(job_id)
    except JobError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    job.stop()
    return job.snapshot(include_history=False)


@app.post("/api/analyze/{job_id}")
async def analyze_job(job_id: str, spec: AnalysisSpec | None = None) -> dict[str, object]:
    try:
        return MANAGER.analyze(job_id, spec or AnalysisSpec())
    except JobError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to the UI
        status = 422 if type(exc).__name__ == "AnalysisError" else 500
        raise HTTPException(status_code=status, detail=f"{type(exc).__name__}: {exc}") from exc


@app.post("/api/predict/{job_id}")
async def predict_job(job_id: str, payload: BatchGenerateRequest) -> dict[str, object]:
    try:
        return MANAGER.predict(
            job_id,
            payload.spec,
            count=payload.count,
            structures=payload.structures,
            mode=payload.mode,
        )
    except JobError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced to the UI
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc


@app.get("/api/runs/{run_id}/export")
async def export_run(run_id: str) -> FileResponse:
    try:
        job = MANAGER.get(run_id)
    except JobError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    run_dir = Path(job.run_dir)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run directory missing: {run_dir}")
    tmp = Path(tempfile.mkdtemp(prefix="topology-run-"))
    archive = tmp / f"{run_id}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in run_dir.rglob("*"):
            if path.is_file():
                bundle.write(path, path.relative_to(run_dir.parent))
    return FileResponse(archive, filename=f"{run_id}.zip", media_type="application/zip")


@app.post("/api/runs/import")
async def import_run(file: Annotated[UploadFile, File()]) -> dict[str, object]:
    tmp = Path(tempfile.mkdtemp(prefix="topology-import-"))
    archive = tmp / "upload.zip"
    with archive.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    try:
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (tmp / member.filename).resolve()
                if not str(target).startswith(str(tmp.resolve())):
                    raise HTTPException(status_code=422, detail="ZIP enthaelt unsichere Pfade.")
            bundle.extractall(tmp)
        job_files = list(tmp.rglob("job.json"))
        if not job_files:
            raise HTTPException(status_code=422, detail="ZIP enthaelt keine job.json.")
        source = job_files[0].parent
        run_id = source.name
        target = Path.cwd() / "runs" / run_id
        if target.exists():
            raise HTTPException(status_code=409, detail=f"Run {run_id} existiert bereits.")
        shutil.copytree(source, target)
        MANAGER.reload()
        return {"imported": run_id, "path": str(target)}
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=422, detail="Datei ist kein gueltiges ZIP.") from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@app.get("/lab", response_class=HTMLResponse)
async def lab(request: Request) -> HTMLResponse:
    # Single-page app: /lab serves the same shell and the client switches tabs.
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"catalog": _catalog().model_dump(mode="json")},
    )


@app.get("/api/ui-state")
async def get_ui_state() -> dict[str, Any]:
    return STORE.get()


@app.post("/api/ui-state")
async def save_ui_state(patch: dict[str, Any]) -> dict[str, Any]:
    return STORE.save(patch)
