from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

from .config import TargetSchema

SPECIAL_TOKENS: tuple[str, ...] = (
    "<pad>",
    "<bos>",
    "<eos>",
    "<endoftext>",
    "<infer>",
    "<generate>",
    "<unk>",
)

WORD_TOKENS: tuple[str, ...] = (
    "betti",
    "ph",
    "genus",
    "meta",
)

CHAR_TOKENS: tuple[str, ...] = tuple("0123456789.,;-()=")

DEFAULT_BOUNDARY = "<endoftext>"


class TokenizerError(ValueError):
    pass


class TargetParseError(ValueError):
    def __init__(self, message: str, *, expected: int | None = None, actual: int | None = None):
        super().__init__(message)
        self.expected = expected
        self.actual = actual


class NumericTokenizer:
    """A deliberately tiny, domain-specific tokenizer.

    Every digit is a token. Numeric punctuation is also a token. The only
    multi-character tokens are the requested words and special tokens.
    """

    def __init__(self) -> None:
        self.special_tokens = SPECIAL_TOKENS
        self.word_tokens = WORD_TOKENS
        self.char_tokens = CHAR_TOKENS
        tokens = [*self.special_tokens, *self.word_tokens, *self.char_tokens]
        self.tokens = tuple(dict.fromkeys(tokens))
        self.token_to_id = {token: i for i, token in enumerate(self.tokens)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<pad>"]
        self.bos_id = self.token_to_id["<bos>"]
        self.eos_id = self.token_to_id["<eos>"]
        self.eot_id = self.token_to_id["<endoftext>"]
        self.unk_id = self.token_to_id["<unk>"]
        alternatives = sorted(self.tokens, key=len, reverse=True)
        self._scanner = re.compile("|".join(re.escape(token) for token in alternatives))

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def tokenize(self, text: str, *, strict: bool = True) -> list[str]:
        out: list[str] = []
        position = 0
        while position < len(text):
            match = self._scanner.match(text, position)
            if match is None:
                if strict:
                    raise TokenizerError(f"unsupported character at position {position}: {text[position]!r}")
                out.append("<unk>")
                position += 1
                continue
            token = match.group(0)
            out.append(token)
            position = match.end()
        return out

    def encode(self, text: str, *, strict: bool = True) -> list[int]:
        return [self.token_to_id[token] for token in self.tokenize(text, strict=strict)]

    def decode(self, ids_or_tokens: Iterable[int | str]) -> str:
        pieces: list[str] = []
        for item in ids_or_tokens:
            if isinstance(item, int):
                pieces.append(self.id_to_token[item])
            else:
                pieces.append(item)
        return "".join(pieces)

    def encode_points(self, points: Sequence[Sequence[float]], *, precision: int = 6) -> list[int]:
        return self.encode(serialize_points(points, precision=precision))

    def point_text(self, points: Sequence[Sequence[float]], *, precision: int = 6) -> str:
        return serialize_points(points, precision=precision)

    def target_text(self, groups: Sequence[Sequence[float]], *, precision: int = 6) -> str:
        return serialize_target_groups(groups, precision=precision)

    def sequence(
        self,
        points: Sequence[Sequence[float]],
        groups: Sequence[Sequence[float]],
        *,
        precision: int = 6,
        boundary: str = DEFAULT_BOUNDARY,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> str:
        text = f"{serialize_points(points, precision=precision)}{boundary}"
        text += serialize_target_groups(groups, precision=precision)
        if add_bos:
            text = f"<bos>{text}"
        if add_eos:
            text = f"{text}<eos>"
        return text


def format_number(value: float, *, precision: int = 6) -> str:
    if not math.isfinite(float(value)):
        raise TokenizerError(f"cannot serialize non-finite number: {value!r}")
    rounded = round(float(value), precision)
    if abs(rounded) < 10 ** (-(precision + 2)):
        rounded = 0.0
    text = f"{rounded:.{precision}f}".rstrip("0").rstrip(".")
    if text in {"", "-0"}:
        text = "0"
    return text


def serialize_points(points: Sequence[Sequence[float]], *, precision: int = 6) -> str:
    if not points:
        raise TokenizerError("point cloud must contain at least one point")
    point_strings: list[str] = []
    for point in points:
        if not point:
            raise TokenizerError("each point must contain at least one coordinate")
        point_strings.append(",".join(format_number(value, precision=precision) for value in point))
    return ";".join(point_strings)


def serialize_target_groups(groups: Sequence[Sequence[float]], *, precision: int = 6) -> str:
    if not groups:
        raise TokenizerError("target must contain at least one group")
    return ";".join(
        ",".join(format_number(value, precision=precision) for value in group)
        for group in groups
    )


def parse_target_groups(text: str, schema: TargetSchema) -> list[list[float]]:
    groups = [group for group in text.split(";")]
    if len(groups) != len(schema.blocks):
        raise TargetParseError(
            f"expected {len(schema.blocks)} target groups, got {len(groups)}",
            expected=len(schema.blocks),
            actual=len(groups),
        )
    parsed: list[list[float]] = []
    for block, group_text in zip(schema.blocks, groups, strict=True):
        values = group_text.split(",")
        if len(values) != len(block.fields):
            raise TargetParseError(
                f"group {block.name!r}: expected {len(block.fields)} values, got {len(values)}",
                expected=len(block.fields),
                actual=len(values),
            )
        row: list[float] = []
        for field_name, raw in zip(block.fields, values, strict=True):
            try:
                value = float(raw)
            except ValueError as exc:
                raise TargetParseError(
                    f"group {block.name!r}, field {field_name!r}: {raw!r} is not a number"
                ) from exc
            if not math.isfinite(value):
                raise TargetParseError(
                    f"group {block.name!r}, field {field_name!r}: value must be finite"
                )
            row.append(value)
        parsed.append(row)
    return parsed
