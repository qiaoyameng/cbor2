"""
Tests for the float fidelity mode (``float_fidelity=True``).

These tests exercise the pure Python implementation directly, as the C
extension does not implement this option.
"""

from __future__ import annotations

import math
from binascii import unhexlify
from io import BytesIO

import pytest

from cbor2._decoder import CBORDecoder, loads
from cbor2._encoder import CBOREncoder, dumps
from cbor2._types import CBOREncodeValueError, FloatValue

# Hand-written vectors: (hex encoding, width, expected numeric value).
# The same value (1.5) encoded at each of the three float widths.
SAME_VALUE_VECTORS = [
    ("f93e00", 16, 1.5),
    ("fa3fc00000", 32, 1.5),
    ("fb3ff8000000000000", 64, 1.5),
]

# Positive and negative zero at each width.
SIGNED_ZERO_VECTORS = [
    ("f90000", 16, 0.0),
    ("f98000", 16, -0.0),
    ("fa00000000", 32, 0.0),
    ("fa80000000", 32, -0.0),
    ("fb0000000000000000", 64, 0.0),
    ("fb8000000000000000", 64, -0.0),
]

# NaN values with distinct signs and payloads at each width.
NAN_VECTORS = [
    "f97e00",  # half, canonical quiet NaN
    "f97e01",  # half, payload 1
    "f9fe00",  # half, negative NaN
    "fa7fc00000",  # single, canonical quiet NaN
    "fa7fc00001",  # single, payload 1
    "faffc12345",  # single, negative NaN with payload 0x412345
    "fb7ff8000000000000",  # double, canonical quiet NaN
    "fb7ff8000000000001",  # double, payload 1
    "fb7ff0000000000001",  # double, signaling NaN with payload 1
    "fbfff8000000000000",  # double, negative NaN
]

ROUNDTRIP_VECTORS = SAME_VALUE_VECTORS + SIGNED_ZERO_VECTORS


@pytest.mark.parametrize("hex_payload, width, expected", ROUNDTRIP_VECTORS)
def test_roundtrip_byte_exact(hex_payload, width, expected):
    """Decoded FloatValues re-encode to the exact original bytes."""
    data = unhexlify(hex_payload)
    value = loads(data, float_fidelity=True)
    assert isinstance(value, FloatValue)
    assert value.width == width
    assert value.bits == data[1:]
    assert value.value == expected
    assert float(value) == expected
    assert dumps(value) == data


@pytest.mark.parametrize("hex_payload", NAN_VECTORS)
def test_nan_payloads_roundtrip_byte_exact(hex_payload):
    """NaN payloads and signs survive a decode/encode round trip."""
    data = unhexlify(hex_payload)
    value = loads(data, float_fidelity=True)
    assert isinstance(value, FloatValue)
    assert math.isnan(value.value)
    assert value.bits == data[1:]
    assert dumps(value) == data


@pytest.mark.parametrize("hex_payload", NAN_VECTORS)
def test_nan_payloads_not_normalized_by_canonical(hex_payload):
    """Fidelity takes precedence over canonical mode: NaNs are kept as-is."""
    data = unhexlify(hex_payload)
    value = loads(data, float_fidelity=True)
    assert dumps(value, canonical=True) == data


def test_same_value_different_widths():
    """The same numeric value at different widths keeps its own wire form."""
    values = [loads(unhexlify(h), float_fidelity=True) for h, _, _ in SAME_VALUE_VECTORS]
    assert [v.width for v in values] == [16, 32, 64]
    assert all(v == 1.5 for v in values)
    # Each value re-encodes to its original width, not the minimal one
    assert [dumps(v).hex() for v in values] == [h for h, _, _ in SAME_VALUE_VECTORS]


def test_signed_zero():
    """The sign of zero is preserved in the bit pattern."""
    pos = loads(unhexlify("f90000"), float_fidelity=True)
    neg = loads(unhexlify("f98000"), float_fidelity=True)
    assert pos.value == neg.value == 0.0
    assert math.copysign(1.0, pos.value) == 1.0
    assert math.copysign(1.0, neg.value) == -1.0
    assert pos.bits == b"\x00\x00"
    assert neg.bits == b"\x80\x00"
    assert dumps(pos) == unhexlify("f90000")
    assert dumps(neg) == unhexlify("f98000")


def test_modified_value_reencoded_at_same_width():
    value = loads(unhexlify("f93e00"), float_fidelity=True)
    value.value = 2.0
    assert value.bits == unhexlify("4000")
    assert dumps(value) == unhexlify("f94000")


def test_modified_value_not_representable():
    value = loads(unhexlify("f93e00"), float_fidelity=True)
    value.value = 1.1  # not exactly representable in half precision
    with pytest.raises(CBOREncodeValueError, match="cannot be represented exactly"):
        dumps(value)


def test_modified_value_out_of_range():
    value = loads(unhexlify("f93e00"), float_fidelity=True)
    value.value = 1e10  # overflows half precision
    with pytest.raises(CBOREncodeValueError, match="out of range"):
        dumps(value)


def test_modified_value_to_nan():
    """A modified NaN is re-encoded as the canonical NaN of its width."""
    value = loads(unhexlify("f97e01"), float_fidelity=True)
    value.value = 1.5
    assert dumps(value) == unhexlify("f93e00")
    value.value = float("nan")
    assert dumps(value) == unhexlify("f97e00")


def test_construct_from_float():
    assert dumps(FloatValue(1.5, 16)) == unhexlify("f93e00")
    assert dumps(FloatValue(1.5, 32)) == unhexlify("fa3fc00000")
    assert dumps(FloatValue(1.5)) == unhexlify("fb3ff8000000000000")
    assert dumps(FloatValue(3.14)) == unhexlify("fb40091eb851eb851f")
    assert dumps(FloatValue(float("nan"), 32)) == unhexlify("fa7fc00000")
    assert dumps(FloatValue(float("inf"), 16)) == unhexlify("f97c00")


def test_construct_from_float_not_representable():
    with pytest.raises(ValueError, match="cannot be represented exactly"):
        FloatValue(1.1, 16)
    with pytest.raises(ValueError, match="out of range"):
        FloatValue(1e10, 16)
    with pytest.raises(ValueError, match="invalid float width"):
        FloatValue(1.5, 8)


def test_from_bits_validation():
    with pytest.raises(ValueError, match="invalid float width"):
        FloatValue.from_bits(b"\x00\x00", 8)
    with pytest.raises(ValueError, match="requires 4 bytes"):
        FloatValue.from_bits(b"\x00\x00", 32)


def test_equality_and_hash():
    value = loads(unhexlify("f93e00"), float_fidelity=True)
    assert value == 1.5
    assert value == FloatValue(1.5, 16)
    assert value != FloatValue(1.5, 32)  # same value, different width
    assert hash(value) == hash(1.5)
    assert repr(value) == "FloatValue(1.5, width=16, bits=b'>\\x00')"


def test_as_map_key():
    """FloatValues are hashable and can be decoded as map keys."""
    value = loads(unhexlify("a1f93e0001"), float_fidelity=True)
    assert value == {FloatValue(1.5, 16): 1}
    assert value == {1.5: 1}


def test_nested_structures():
    data = unhexlify("82f93e00fb3ff8000000000000")
    value = loads(data, float_fidelity=True)
    assert [v.width for v in value] == [16, 64]
    assert dumps(value) == data


def test_decoder_option_must_be_bool():
    with pytest.raises(ValueError, match="float_fidelity must be a bool"):
        CBORDecoder(BytesIO(b""), float_fidelity=1)


def test_encoder_via_class():
    stream = BytesIO()
    encoder = CBOREncoder(stream)
    encoder.encode(FloatValue(1.5, 16))
    assert stream.getvalue() == unhexlify("f93e00")


#
# The legacy decode path and numeric comparisons are unaffected by the option
#


def test_legacy_decode_path_unaffected():
    assert type(loads(unhexlify("f93e00"))) is float
    assert loads(unhexlify("f93e00")) == 1.5
    assert loads(unhexlify("fb3ff8000000000000")) == 1.5
    # NaN payloads collapse to a plain float NaN without the option
    legacy_nan = loads(unhexlify("fa7fc00001"))
    assert type(legacy_nan) is float
    assert math.isnan(legacy_nan)
    # Signed zero still decodes to a plain (signed) zero float
    assert loads(unhexlify("f98000")) == 0.0


def test_legacy_encode_path_unaffected():
    assert dumps(1.5) == unhexlify("fb3ff8000000000000")
    assert dumps(float("nan")) == unhexlify("f97e00")
    assert dumps(1.5, canonical=True) == unhexlify("f93e00")
    assert dumps(float("nan"), canonical=True) == unhexlify("f97e00")
