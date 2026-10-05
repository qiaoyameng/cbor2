import math
import struct
from array import array
from binascii import unhexlify

import pytest

import cbor2._decoder as decoder
import cbor2._encoder as encoder
import cbor2._types as types

integer_tags = [
    pytest.param(64, 1, "B", 2**8 - 1, False),
    pytest.param(65, 2, "<H", 2**16 - 1, False),
    pytest.param(66, 4, ">I", 2**32 - 1, False),
    pytest.param(67, 8, ">Q", 2**64 - 1, False),
    pytest.param(68, 1, "b", 2**7 - 1, True),
    pytest.param(69, 2, "<h", 2**15 - 1, True),
    pytest.param(70, 4, ">i", 2**31 - 1, True),
    pytest.param(71, 8, ">q", 2**63 - 1, True),
    pytest.param(72, 2, ">H", 2**16 - 1, False),
    pytest.param(73, 4, "<I", 2**32 - 1, False),
    pytest.param(74, 8, "<Q", 2**64 - 1, False),
    pytest.param(75, 2, ">h", 2**15 - 1, True),
    pytest.param(76, 4, "<i", 2**31 - 1, True),
    pytest.param(77, 8, "<q", 2**63 - 1, True),
]

float_tags = [
    pytest.param(80, 4, "<f", "little"),
    pytest.param(81, 4, ">f", "big"),
    pytest.param(82, 8, ">d", "big"),
    pytest.param(83, 8, "<d", "little"),
]


def test_decodes_known_little_and_big_endian_uint32_vectors():
    little = decoder.loads(unhexlify("d849480100000002000000"))
    big = decoder.loads(unhexlify("d842480000000100000002"))

    assert little.tag == 73
    assert little.itemsize == 4
    assert little.byteorder == "little"
    assert list(little) == [1, 2]
    assert little.data == struct.pack("<II", 1, 2)

    assert big.tag == 66
    assert big.itemsize == 4
    assert big.byteorder == "big"
    assert list(big) == [1, 2]
    assert big.data == struct.pack(">II", 1, 2)


@pytest.mark.parametrize("tag,width,format_,maximum,signed", integer_tags)
def test_integer_maximum_roundtrip_and_array_conversion(tag, width, format_, maximum, signed):
    minimum = -(maximum + 1) if signed else 0
    payload = struct.pack(format_, minimum) + struct.pack(format_, maximum)
    encoded = encoder.dumps(types.CBORTag(tag, payload))

    decoded = decoder.loads(encoded)

    assert isinstance(decoded, types.TypedArray)
    assert decoded.tag == tag
    assert decoded.itemsize == width
    assert decoded[0] == minimum
    assert decoded[1] == maximum
    assert encoder.dumps(decoded) == encoded

    values = decoded.to_array()
    assert values.typecode == decoded.typecode
    assert values.itemsize == width
    assert list(values) == [minimum, maximum]


@pytest.mark.parametrize("tag,width,format_,byteorder", float_tags)
@pytest.mark.parametrize(
    "value",
    [
        -0.0,
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_special_float_values_roundtrip(tag, width, format_, byteorder, value):
    payload = struct.pack(format_, value)
    decoded = decoder.loads(encoder.dumps(types.CBORTag(tag, payload)))

    assert decoded.tag == tag
    assert decoded.itemsize == width
    assert decoded.byteorder == byteorder
    assert decoded.data == payload
    if math.isnan(value):
        assert math.isnan(decoded[0])
    else:
        assert decoded[0] == value
        assert math.copysign(1, decoded[0]) == math.copysign(1, value)

    converted = decoded.to_array()
    if math.isnan(value):
        assert math.isnan(converted[0])
    else:
        assert converted[0] == value
        assert math.copysign(1, converted[0]) == math.copysign(1, value)


@pytest.mark.parametrize("tag", [tag.values[0] for tag in integer_tags + float_tags])
def test_empty_typed_array_roundtrip(tag):
    decoded = decoder.loads(encoder.dumps(types.CBORTag(tag, b"")))

    assert isinstance(decoded, types.TypedArray)
    assert decoded.tag == tag
    assert len(decoded) == 0
    assert list(decoded) == []
    assert list(decoded.to_array()) == []
    assert encoder.dumps(decoded) == encoder.dumps(types.CBORTag(tag, b""))


@pytest.mark.parametrize(
    "tag,width", [(param.values[0], param.values[1]) for param in integer_tags]
)
def test_rejects_payload_not_multiple_of_element_width(tag, width):
    if width == 1:
        pytest.skip("one-byte elements cannot have a partial element")

    with pytest.raises(types.CBORDecodeValueError, match="invalid typed array value"):
        decoder.loads(encoder.dumps(types.CBORTag(tag, b"\x00" * (width + 1))))


def test_rejects_non_bytestring_typed_array_payload():
    with pytest.raises(types.CBORDecodeValueError, match="invalid typed array value"):
        decoder.loads(unhexlify("d8428101"))


def test_unknown_typed_array_tag_uses_generic_tag_mechanism():
    def tag_hook(decoder, tag):
        assert tag == types.CBORTag(78, b"")
        return "handled"

    assert decoder.loads(unhexlify("d84e40"), tag_hook=tag_hook) == "handled"


def test_negative_and_out_of_range_indexes():
    typed_array = types.TypedArray(66, struct.pack(">II", 1, 2))

    assert typed_array[-1] == 2
    with pytest.raises(IndexError):
        typed_array[2]
    with pytest.raises(IndexError):
        typed_array[-3]


def test_from_array_preserves_wire_endianness():
    little = types.TypedArray.from_array(73, array("I", [1, 2]))
    big = types.TypedArray.from_array(66, array("I", [1, 2]))

    assert little.data == struct.pack("<II", 1, 2)
    assert big.data == struct.pack(">II", 1, 2)
    assert encoder.dumps(little) == unhexlify("d849480100000002000000")
    assert encoder.dumps(big) == unhexlify("d842480000000100000002")
