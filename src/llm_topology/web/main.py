from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ConfigDict, Field

from .. import __version__
from ..config import (
    CoefficientField,
    DatasetResponse,
    DatasetSpec,
    FiltrationMethod,
    StructureKind,
    TargetSchema,
    default_target_blocks,
    optional_target_blocks,
)
from ..pipeline import PipelineError, generate_dataset
from ..tokenizer import (
    DEFAULT_BOUNDARY,
    NumericTokenizer,
    TargetParseError,
    TokenizerError,
    parse_target_groups,
    serialize_target_groups,
)
from ..topology.generators import STRUCTURE_IDS, STRUCTURE_LABELS

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
        for block in default_target_blocks(max_dim=2)
    ]
    blocks.extend(
        {"name": block.name, "fields": block.fields, "optional": True}
        for block in optional_target_blocks()
    )
    return CatalogResponse(
        phase="Phase 0 - Data Foundation",
        version=__version__,
        structures=structures,
        default_spec=DatasetSpec().model_dump(mode="json"),
        target_blocks=blocks,
        tokens=list(TOKENIZER.tokens),
        boundary_token=DEFAULT_BOUNDARY,
        methods=[method.value for method in FiltrationMethod],
        fields=[field.value for field in CoefficientField],
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

    return {
        "status": "ok",
        "version": __version__,
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "gudhi": gudhi.__version__,
        "phase": "Phase 0 - Data Foundation",
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
