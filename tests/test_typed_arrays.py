from __future__ import annotations

import math
import struct
import sys
from array import array
from binascii import unhexlify

import pytest

import cbor2._decoder as decoder
import cbor2._encoder as encoder
from cbor2 import CBORTag, CBORTypedArray


def test_known_uint32_endianness_vectors():
    big = decoder.loads(unhexlify("d8424412345678"))
    little = decoder.loads(unhexlify("d8464478563412"))

    assert isinstance(big, CBORTypedArray)
    assert isinstance(little, CBORTypedArray)
    assert big.tag == 66
    assert little.tag == 70
    assert big.itemsize == little.itemsize == 4
    assert big.byteorder == "big"
    assert little.byteorder == "little"
    assert big[0] == little[0] == 305419896
    assert big[-1] == little[-1] == 305419896
    assert list(big) == list(little) == [305419896]
    assert big.to_array() == array("I", [305419896])
    assert little.to_array() == array("I", [305419896])


def test_known_float_vectors():
    float32 = decoder.loads(unhexlify("d8514440490fdb"))
    float32_le = decoder.loads(unhexlify("d85544db0f4940"))
    float64 = decoder.loads(unhexlify("d85248400921fb54442d18"))
    float64_le = decoder.loads(unhexlify("d85648182d4454fb210940"))

    assert float32.tag == 81
    assert float32_le.tag == 85
    assert float64.tag == 82
    assert float64_le.tag == 86
    assert float32[0] == pytest.approx(math.pi, abs=1e-7)
    assert float64[0] == math.pi
    assert float32_le[0] == float32[0]
    assert float64_le[0] == float64[0]
    assert isinstance(float32.to_array(), array)
    assert float64.to_array().typecode == "d"


@pytest.mark.parametrize("tag", range(64, 80))
def test_integer_limits_roundtrip(tag):
    itemsize = 1 << ((tag - 64) % 4)
    signed = tag >= 72
    endian = "little" if (tag - 64) % 8 >= 4 else "big"
    prefix = "<" if endian == "little" else ">"

    if signed:
        values = [-(2 ** (itemsize * 8 - 1)), 2 ** (itemsize * 8 - 1) - 1]
        fmt = {1: "b", 2: "h", 4: "i", 8: "q"}[itemsize]
    else:
        values = [0, 2 ** (itemsize * 8) - 1]
        fmt = {1: "B", 2: "H", 4: "I", 8: "Q"}[itemsize]

    payload = struct.pack(f"{prefix}2{fmt}", *values)
    expected = CBORTypedArray(tag, payload)
    decoded = decoder.loads(encoder.dumps(expected))

    assert decoded == expected
    assert decoded.tag == tag
    assert decoded.itemsize == itemsize
    assert decoded.byteorder == endian
    assert list(decoded) == values
    assert len(decoded) == 2
    assert decoded.to_array().typecode == expected.to_array().typecode


@pytest.mark.parametrize("tag", (81, 82, 85, 86))
def test_special_float_values_roundtrip(tag):
    itemsize = 4 if tag in (81, 85) else 8
    endian = "big" if tag in (81, 82) else "little"
    fmt = "f" if itemsize == 4 else "d"
    values = [math.inf, -math.inf, math.nan, -0.0]
    payload = CBORTypedArray.from_values(tag, values).value

    decoded = decoder.loads(encoder.dumps(CBORTypedArray(tag, payload)))
    native_payload = payload
    if sys.byteorder != endian:
        native_payload = b"".join(
            payload[offset : offset + itemsize][::-1]
            for offset in range(0, len(payload), itemsize)
        )

    assert decoded.value == payload
    assert math.isinf(decoded[0]) and decoded[0] > 0
    assert math.isinf(decoded[1]) and decoded[1] < 0
    assert math.isnan(decoded[2])
    assert math.copysign(1.0, decoded[3]) == -1.0
    assert decoded.to_array().tobytes() == native_payload
    assert decoded.as_array().tobytes() == native_payload


def test_empty_typed_array_roundtrip():
    expected = CBORTypedArray(65, b"")
    encoded = encoder.dumps(expected)
    assert encoded == unhexlify("d84140")
    decoded = decoder.loads(encoded)

    assert decoded == expected
    assert len(decoded) == 0
    assert list(decoded) == []
    assert decoded.to_array() == array("H")


def test_constructor_rejects_misaligned_payload():
    with pytest.raises(ValueError, match="not a multiple"):
        CBORTypedArray(65, b"\x00")


def test_decoder_rejects_misaligned_payload():
    with pytest.raises(decoder.CBORDecodeValueError, match="not a multiple"):
        decoder.loads(unhexlify("d8414100"))


def test_decoder_rejects_non_bytestring_payload():
    with pytest.raises(decoder.CBORDecodeValueError, match="byte string"):
        decoder.loads(unhexlify("d84101"))


@pytest.mark.parametrize("tag", (80, 83, 84, 87))
def test_unsupported_typed_array_tag_uses_generic_tag(tag):
    encoded = encoder.dumps(CBORTag(tag, b"\x00"))
    value = decoder.loads(encoded)
    assert value == CBORTag(tag, b"\x00")

    observed = []

    def tag_hook(decoder, tag):
        observed.append(tag)
        return tag.value

    value = decoder.loads(encoded, tag_hook=tag_hook)
    assert value == b"\x00"
    assert observed == [CBORTag(tag, b"\x00")]


def test_encode_uses_object_endianness():
    big = CBORTypedArray(65, unhexlify("01020304"))
    little = CBORTypedArray(69, unhexlify("02010403"))

    assert encoder.dumps(big) == unhexlify("d8414401020304")
    assert encoder.dumps(little) == unhexlify("d8454402010403")
    assert list(big) == list(little) == [258, 772]


def test_from_array_and_properties():
    values = array("i", [-1, 1])
    big = CBORTypedArray.from_array(74, values)
    little = CBORTypedArray.from_array(78, values)

    assert big.data == unhexlify("ffffffff00000001")
    assert little.data == unhexlify("ffffffff01000000")
    assert big.width == 4
    assert big.element_width == 4
    assert big.endianness == "big"
    assert big.endian == "big"
    assert little.endianness == "little"
    assert little.endian == "little"
