"""Tests for the optional bit-exact float fidelity mode."""

from __future__ import annotations

import math
from binascii import hexlify, unhexlify
from io import BytesIO

import pytest


@pytest.fixture
def fidelity(impl):
    if not hasattr(impl, "CBORFloat"):
        pytest.skip("float fidelity is only implemented in the pure Python module")

    return impl


def roundtrip(impl, payload: str, **encode_kwargs):
    """Decode with fidelity and re-encode; return (value, encoded_bytes)."""
    raw = unhexlify(payload)
    value = impl.loads(raw, float_fidelity=True)
    return value, impl.dumps(value, **encode_kwargs)


# ---------------------------------------------------------------------------
# Hand-written vectors: the same numeric value carried at three widths
# ---------------------------------------------------------------------------

SAME_VALUE_DIFFERENT_WIDTHS = [
    # 1.0
    ("f93c00", 16, 0x3C00),
    ("fa3f800000", 32, 0x3F800000),
    ("fb3ff0000000000000", 64, 0x3FF0000000000000),
    # -2.0
    ("f9c000", 16, 0xC000),
    ("fac0000000", 32, 0xC0000000),
    ("fbc000000000000000", 64, 0xC000000000000000),
    # +infinity
    ("f97c00", 16, 0x7C00),
    ("fa7f800000", 32, 0x7F800000),
    ("fb7ff0000000000000", 64, 0x7FF0000000000000),
]


@pytest.mark.parametrize("payload,width,bits", SAME_VALUE_DIFFERENT_WIDTHS)
def test_same_value_different_widths_byte_exact(fidelity, payload, width, bits):
    value, encoded = roundtrip(fidelity, payload)

    assert isinstance(value, fidelity.CBORFloat)
    assert value.width == width
    assert value.bits == bits
    # The whole point of the wrapper: identical bytes survive the round trip.
    assert encoded == unhexlify(payload)
    assert hexlify(encoded).decode() == payload


@pytest.mark.parametrize("payload,width,bits", SAME_VALUE_DIFFERENT_WIDTHS)
def test_same_value_compares_equal_across_widths(fidelity, payload, width, bits):
    value, _ = roundtrip(fidelity, payload)

    # ...but numerically all widths of the same value compare equal.
    if math.isinf(value):
        assert math.isinf(value)
    else:
        assert value == float(value)


def test_one_point_zero_three_widths_numerically_equal(fidelity):
    variants = [roundtrip(fidelity, p)[0] for p in ("f93c00", "fa3f800000", "fb3ff0000000000000")]
    assert all(v == 1.0 for v in variants)
    assert variants[0] == variants[1] == variants[2]
    assert {v.width for v in variants} == {16, 32, 64}
    # Numeric identity must not erase the distinct bit patterns.
    assert len({v.bits for v in variants}) == 3


# ---------------------------------------------------------------------------
# Hand-written vectors: positive and negative zero
# ---------------------------------------------------------------------------

SIGNED_ZEROES = [
    ("f90000", 16, 0x0000, +1.0),
    ("f98000", 16, 0x8000, -1.0),
    ("fa00000000", 32, 0x00000000, +1.0),
    ("fa80000000", 32, 0x80000000, -1.0),
    ("fb0000000000000000", 64, 0x0000000000000000, +1.0),
    ("fb8000000000000000", 64, 0x8000000000000000, -1.0),
]


@pytest.mark.parametrize("payload,width,bits,sign", SIGNED_ZEROES)
def test_signed_zeroes_byte_exact(fidelity, payload, width, bits, sign):
    value, encoded = roundtrip(fidelity, payload)

    assert value == 0.0
    assert value.width == width
    assert value.bits == bits
    assert math.copysign(1.0, value) == sign
    assert encoded == unhexlify(payload)


# ---------------------------------------------------------------------------
# Hand-written vectors: several NaN payloads (quiet, signaling, max, negative)
# ---------------------------------------------------------------------------

NAN_VECTORS = [
    # half precision
    ("f97e00", 16, 0x7E00),   # canonical quiet NaN
    ("f97e01", 16, 0x7E01),   # quiet NaN, payload 1
    ("f97d55", 16, 0x7D55),   # signaling NaN, payload
    ("f97fff", 16, 0x7FFF),   # all-bits payload
    # single precision
    ("fa7fc00000", 32, 0x7FC00000),  # canonical quiet NaN
    ("fa7fc00001", 32, 0x7FC00001),  # quiet NaN, payload 1
    ("fa7f800001", 32, 0x7F800001),  # signaling NaN, payload 1
    ("fa7fffffff", 32, 0x7FFFFFFF),  # maximum payload
    ("fafff80000", 32, 0xFFF80000),  # negative quiet NaN
    # double precision
    ("fb7ff8000000000000", 64, 0x7FF8000000000000),  # canonical quiet NaN
    ("fb7ff8000000000001", 64, 0x7FF8000000000001),  # quiet NaN, payload 1
    ("fb7ff0000000000001", 64, 0x7FF0000000000001),  # signaling NaN, payload 1
    ("fbffffffffffffffff", 64, 0xFFFFFFFFFFFFFFFF),  # negative, maximum payload
]


@pytest.mark.parametrize("payload,width,bits", NAN_VECTORS)
def test_nan_payloads_byte_exact(fidelity, payload, width, bits):
    value, encoded = roundtrip(fidelity, payload)

    assert math.isnan(value)
    assert value.width == width
    assert value.bits == bits
    # No NaN normalization: every payload is reproduced byte for byte.
    assert encoded == unhexlify(payload)


def test_distinct_nan_payloads_remain_distinct(fidelity):
    values = [roundtrip(fidelity, p)[0] for p, _, _ in NAN_VECTORS]
    assert all(math.isnan(v) for v in values)
    assert len({v.bits for v in values}) == len(NAN_VECTORS)


# ---------------------------------------------------------------------------
# Nested structures and the file-based entry point
# ---------------------------------------------------------------------------

def test_nested_structure_roundtrip(fidelity):
    raw = unhexlify("83f97e01f98000fa7f800001")  # [nan(payload), -0.0, sNaN(payload)]
    value = fidelity.loads(raw, float_fidelity=True)
    assert fidelity.dumps(value) == raw


def test_load_from_stream(fidelity):
    raw = unhexlify("fb7ff0000000000001")
    with BytesIO(raw) as stream:
        value = fidelity.load(stream, float_fidelity=True)

    assert isinstance(value, fidelity.CBORFloat)
    assert value.width == 64
    assert value.bits == 0x7FF0000000000001
    assert fidelity.dumps(value) == raw


def test_decoder_attribute(fidelity):
    with BytesIO(b"") as stream:
        decoder = fidelity.CBORDecoder(stream, float_fidelity=True)
        assert decoder.float_fidelity is True

    with BytesIO(b"") as stream:
        decoder = fidelity.CBORDecoder(stream)
        assert decoder.float_fidelity is False
        decoder.float_fidelity = True
        assert decoder.float_fidelity is True


# ---------------------------------------------------------------------------
# Explicit construction and re-validation after a numeric change
# ---------------------------------------------------------------------------

def test_explicit_constructor_encodes_width(fidelity):
    value = fidelity.CBORFloat(1.5, 16)
    assert value.width == 16
    assert value.bits == 0x3E00
    assert fidelity.dumps(value) == unhexlify("f93e00")


def test_from_bits(fidelity):
    value = fidelity.CBORFloat.from_bits(0x7FC00001, 32)
    assert math.isnan(value)
    assert value.width == 32
    assert value.bits == 0x7FC00001
    assert fidelity.dumps(value) == unhexlify("fa7fc00001")


def test_unrepresentable_value_raises(fidelity):
    # 1.1 cannot be expressed exactly as a 32-bit float
    with pytest.raises(ValueError):
        fidelity.CBORFloat(1.1, 32)

    # 1e40 overflows a half precision float
    with pytest.raises(ValueError):
        fidelity.CBORFloat(1e40, 16)


def test_invalid_width(fidelity):
    with pytest.raises(ValueError):
        fidelity.CBORFloat(1.0, 8)

    with pytest.raises(ValueError):
        fidelity.CBORFloat.from_bits(0x3C00, 8)


def test_from_bits_out_of_range(fidelity):
    with pytest.raises(ValueError):
        fidelity.CBORFloat.from_bits(1 << 16, 16)

    with pytest.raises(ValueError):
        fidelity.CBORFloat.from_bits(-1, 64)


def test_tampered_bits_rejected_on_encode(fidelity):
    value = fidelity.CBORFloat(1.0, 32)
    value.bits = 0x3F800001  # no longer the bit pattern of 1.0
    with pytest.raises(fidelity.CBOREncodeValueError):
        fidelity.dumps(value)


def test_nan_sign_mismatch_rejected_on_encode(fidelity):
    value = fidelity.CBORFloat(float("nan"), 64)  # positive canonical NaN
    value.bits = 0xFFF8000000000000  # tampered into a negative NaN payload
    with pytest.raises(fidelity.CBOREncodeValueError):
        fidelity.dumps(value)


# ---------------------------------------------------------------------------
# Canonical mode vs. fidelity: fidelity explicitly wins
# ---------------------------------------------------------------------------

def test_canonical_does_not_shrink_or_normalize_wrapper(fidelity):
    for payload in ("fa3f800000", "fb3ff0000000000000", "fa7fc00001"):
        value = fidelity.loads(unhexlify(payload), float_fidelity=True)
        assert fidelity.dumps(value, canonical=True) == unhexlify(payload)


def test_canonical_still_minimizes_plain_floats(fidelity):
    # Plain floats are unaffected: canonical mode still narrows them.
    assert fidelity.dumps(1.0, canonical=True) == unhexlify("f93c00")
    assert fidelity.dumps(float("nan"), canonical=True) == unhexlify("f97e00")


# ---------------------------------------------------------------------------
# The old float decode path and numeric comparisons are untouched
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", ["f93c00", "fa3f800000", "fb3ff0000000000000"])
def test_default_decode_still_plain_float(fidelity, payload):
    value = fidelity.loads(unhexlify(payload))
    assert type(value) is float
    assert value == 1.0


def test_default_decode_normalizes_nan(fidelity):
    value = fidelity.loads(unhexlify("fa7fc00001"))
    assert type(value) is float
    assert math.isnan(value)
    # Existing behavior: a plain float NaN re-encodes as half canonical NaN.
    assert fidelity.dumps(value) == unhexlify("f97e00")


def test_wrapper_numeric_comparison(fidelity):
    half = fidelity.loads(unhexlify("f93c00"), float_fidelity=True)
    single = fidelity.loads(unhexlify("fa3f800000"), float_fidelity=True)
    assert half == single == 1.0
    assert half < 2.0
    assert sorted([half, single, 0.5]) == [0.5, half, single]
    assert half + single == 2.0
    assert isinstance(half, float)
