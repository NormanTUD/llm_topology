from __future__ import annotations

import pytest

from llm_topology.config import TargetBlock, TargetSchema, all_target_blocks
from llm_topology.tokenizer import (
    DEFAULT_BOUNDARY,
    NumericTokenizer,
    TargetParseError,
    TokenizerError,
    format_number,
    parse_target_groups,
    serialize_points,
    serialize_target_groups,
)


def test_vocabulary_contains_requested_tokens() -> None:
    tokenizer = NumericTokenizer()
    for token in "0123456789.,;-()=":
        assert token in tokenizer.tokens
    for token in ("betti", "<endoftext>", "<bos>", "<eos>"):
        assert token in tokenizer.tokens


def test_digits_are_individual_tokens() -> None:
    tokenizer = NumericTokenizer()
    assert tokenizer.tokenize("10.5") == ["1", "0", ".", "5"]
    assert tokenizer.encode("10.5") == [tokenizer.token_to_id[token] for token in "10.5"]


def test_roundtrip_text() -> None:
    tokenizer = NumericTokenizer()
    text = "1.2,-3.4;5.0<endoftext>1,4,1;1,3,1;2;1;0,2,0.5,2"
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text


def test_unknown_character_is_graceful() -> None:
    tokenizer = NumericTokenizer()
    with pytest.raises(TokenizerError):
        tokenizer.encode("1.0;2.0?")
    assert "<unk>" in tokenizer.tokenize("1.0;2.0?", strict=False)


def test_format_number_is_stable() -> None:
    assert format_number(1.0) == "1"
    assert format_number(-0.0) == "0"
    assert format_number(1.23456789, precision=3) == "1.235"
    assert format_number(1e-12) == "0"
    with pytest.raises(TokenizerError):
        format_number(float("nan"))


def test_serialize_points() -> None:
    text = serialize_points([[1.0, -2.5, 0.0], [3.0, 4.0, 5.0]], precision=3)
    assert text == "1,-2.5,0;3,4,5"


def test_target_roundtrip_default_schema() -> None:
    schema = TargetSchema(blocks=all_target_blocks(max_dim=2))
    groups = [[1, 4, 1], [1, 3, 1], [2], [1], [0, 2, 0.5, 2]]
    text = serialize_target_groups(groups)
    assert text == "1,4,1;1,3,1;2;1;0,2,0.5,2"
    assert parse_target_groups(text, schema) == [
        [1.0, 4.0, 1.0],
        [1.0, 3.0, 1.0],
        [2.0],
        [1.0],
        [0.0, 2.0, 0.5, 2.0],
    ]


def test_target_parser_reports_group_errors() -> None:
    schema = TargetSchema(blocks=all_target_blocks(max_dim=2))
    with pytest.raises(TargetParseError) as exc_info:
        parse_target_groups("1,4,1;1,3,1", schema)
    assert exc_info.value.expected == 5
    assert exc_info.value.actual == 2

    with pytest.raises(TargetParseError):
        parse_target_groups("1,4,1;1,3;2;1;0,2,0.5,2", schema)


def test_target_parser_reports_non_numeric_values() -> None:
    schema = TargetSchema(
        blocks=[
            TargetBlock(name="betti", fields=["b0", "b1", "b2"]),
        ]
    )
    with pytest.raises(TargetParseError):
        parse_target_groups("1,nope,1", schema)


def test_sequence_uses_explicit_boundary() -> None:
    tokenizer = NumericTokenizer()
    groups = [[1, 1, 0], [1, 1, 0], [0], [1], [0, 2, 0.5, 2]]
    text = tokenizer.sequence([[1, 2], [3, 4]], groups, precision=2)
    before, after = text.split(DEFAULT_BOUNDARY)
    assert before == "1,2;3,4"
    assert after == "1,1,0;1,1,0;0;1;0,2,0.5,2"
    assert tokenizer.decode(tokenizer.encode(text)) == text
