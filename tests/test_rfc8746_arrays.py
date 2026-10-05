"""Tests for RFC 8746 typed numeric arrays and multidimensional arrays."""

from __future__ import annotations

import struct
from array import array
from binascii import unhexlify

import pytest

from cbor2 import C_CONTIGUOUS, F_CONTIGUOUS, CBORDecodeValueError, NDArray, TypedArray


# --------------------------------------------------------------------------
# Typed numeric arrays (tags 64-86)
# --------------------------------------------------------------------------

def test_typed_array_integer_endianness(impl):
    big = impl.loads(unhexlify("d84146000100020003"))
    assert isinstance(big, TypedArray)
    assert big.tag == 65
    assert list(big) == [1, 2, 3]

    little = impl.loads(unhexlify("d84746010002000300"))
    assert little.tag == 71
    assert list(little) == [1, 2, 3]

    native = impl.loads(unhexlify("d84446010002000300"))
    assert native.tag == 68
    assert list(native) == [1, 2, 3]


def test_typed_array_float_tags(impl):
    # Appendix A.1 of RFC 8746: tag 85 float32 [-2.0]
    value = impl.loads(unhexlify("d85544c0000000"))
    assert value.tag == 85
    assert list(value) == [-2.0]

    f64 = TypedArray(86, [1.5])
    assert list(impl.loads(impl.dumps(f64))) == [1.5]


def test_typed_array_float16(impl):
    value = impl.loads(
        bytes([0xd8, 84, 0x44]) + struct.pack(">ee", 1.0, 2.5)
    )
    assert value.tag == 84
    assert list(value) == [1.0, 2.5]
    assert value.to_wire_bytes() == struct.pack(">ee", 1.0, 2.5)


def test_typed_array_round_trips_all_integer_tags(impl):
    for tag, typecode, wire in (
        (64, "B", "B"),
        (65, "H", ">H"),
        (71, "H", "<H"),
        (74, "b", "b"),
        (77, "q", ">q"),
        (83, "q", "<q"),
    ):
        arr = TypedArray(tag, list(range(4)))
        decoded = impl.loads(impl.dumps(arr))
        assert decoded.tag == tag
        assert list(decoded) == [0, 1, 2, 3]


def test_typed_array_slice_is_view(impl):
    value = impl.loads(unhexlify("d84146000100020003"))
    part = value[1:3]
    assert isinstance(part, TypedArray)
    assert list(part) == [2, 3]
    assert part.tag == value.tag
    with pytest.raises(ValueError):
        value[::2]


def test_typed_array_unaligned_length(impl):
    with pytest.raises(CBORDecodeValueError):
        impl.loads(unhexlify("d8414100"))


def test_typed_array_requires_bytestring(impl):
    # tag 64 wrapping a plain integer instead of a byte string
    with pytest.raises(CBORDecodeValueError):
        impl.loads(unhexlify("d84001"))


def test_typed_array_element_cap(impl):
    # 3 uint16 elements but a cap of 2
    payload = bytes([0xd8, 65, 0x46]) + struct.pack(">HHH", 1, 2, 3)
    with pytest.raises(CBORDecodeValueError):
        impl.loads(payload, max_elements=2)


def test_native_array_array_encoded(impl):
    decoded = impl.loads(impl.dumps(array("i", [1, -2, 3])))
    assert isinstance(decoded, TypedArray)
    assert list(decoded) == [1, -2, 3]


# --------------------------------------------------------------------------
# Multidimensional arrays (tags 40 and 1040)
# --------------------------------------------------------------------------

def test_non_square_layouts_same_coordinates(impl):
    # A non-square 3x2 matrix. C order stores rows contiguously, F order
    # stores columns contiguously; coordinate indexing must agree.
    c_order = impl.loads(
        unhexlify("d904108282030286000102030405")
    )
    f_order = impl.loads(unhexlify("d8288282030286000204010305"))
    assert c_order.order == C_CONTIGUOUS
    assert f_order.order == F_CONTIGUOUS
    assert c_order.shape == f_order.shape == (3, 2)

    expected = [[0, 1], [2, 3], [4, 5]]
    for row in range(3):
        for col in range(2):
            assert c_order[row, col] == f_order[row, col]
            assert c_order[row, col] == expected[row][col]

    assert c_order.tolist() == f_order.tolist() == expected


def test_three_dimensional_column_order(impl):
    values = [0, 4, 2, 6, 1, 5, 3, 7]
    arr = NDArray((2, 2, 2), values, F_CONTIGUOUS)
    decoded = impl.loads(impl.dumps(arr))
    assert decoded.shape == (2, 2, 2)
    assert decoded.tolist() == [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]


def test_slice_view_shares_backing(impl):
    arr = impl.loads(impl.dumps(NDArray((3, 2), list(range(6)), C_CONTIGUOUS)))
    view = arr[1:3, :]
    assert isinstance(view, NDArray)
    assert view.data is arr.data
    assert view.shape == (2, 2)
    assert view.tolist() == [[2, 3], [4, 5]]

    column = arr[:, 1]
    assert column.data is arr.data
    assert column.tolist() == [1, 3, 5]

    assert arr[2, 1] == 5
    assert arr[-1, -1] == 5


def test_slice_view_with_typed_array_backing(impl):
    backing = TypedArray(65, [0, 1, 2, 3, 4, 5])
    arr = impl.loads(impl.dumps(NDArray((3, 2), backing, C_CONTIGUOUS)))
    assert isinstance(arr.data, TypedArray)
    view = arr[0:2, :]
    assert view.data is arr.data
    assert view.tolist() == [[0, 1], [2, 3]]


def test_scalar_array(impl):
    decoded = impl.loads(impl.dumps(NDArray((), [42], C_CONTIGUOUS)))
    assert decoded.shape == ()
    assert decoded.ndim == 0
    assert decoded.size == 1
    assert decoded[()] == 42
    assert decoded.tolist() == 42
    with pytest.raises(TypeError):
        len(decoded)
    with pytest.raises(TypeError):
        iter(decoded)


def test_zero_length_dimension(impl, order=C_CONTIGUOUS):
    decoded = impl.loads(impl.dumps(NDArray((0, 5), [], order)))
    assert decoded.shape == (0, 5)
    assert decoded.size == 0
    assert decoded.tolist() == []
    assert len(decoded) == 0

    nested = impl.loads(impl.dumps(NDArray((3, 0, 2), [], order)))
    assert nested.tolist() == [[], [], []]
    assert nested.size == 0


def test_zero_length_dimension_fortran(impl):
    test_zero_length_dimension(impl, F_CONTIGUOUS)


def test_shape_product_mismatch(impl):
    # shape [3,3] (product 9) but only 6 elements
    bad = unhexlify("d904108282030386000102030405")
    with pytest.raises(CBORDecodeValueError):
        impl.loads(bad)


def test_negative_dimension_rejected(impl):
    # tag 1040, [[-1], []]
    bad = unhexlify("d904108282812080")
    with pytest.raises(CBORDecodeValueError):
        impl.loads(bad)


def test_non_integer_dimension_rejected(impl):
    # tag 1040, [[true], []]
    bad = unhexlify("d90410828281f580")
    with pytest.raises(CBORDecodeValueError):
        impl.loads(bad)


def test_max_ndim_cap(impl):
    payload = impl.dumps(NDArray((1, 1, 1), [1], C_CONTIGUOUS))
    with pytest.raises(CBORDecodeValueError):
        impl.loads(payload, max_ndim=2)


def test_max_elements_cap(impl):
    payload = impl.dumps(NDArray((100, 100), list(range(10000)), C_CONTIGUOUS))
    with pytest.raises(CBORDecodeValueError):
        impl.loads(payload, max_elements=9999)


def _uint(n: int) -> bytes:
    if n < 24:
        return bytes([n])
    if n < 2**16:
        return bytes([25]) + struct.pack(">H", n)
    if n < 2**32:
        return bytes([26]) + struct.pack(">L", n)
    return bytes([27]) + struct.pack(">Q", n)


def test_huge_shape_rejected_without_allocation():
    # Declares an enormous product but carries no backing data; the decoder
    # must reject the dimensions before attempting to read or allocate the
    # payload.
    stream = unhexlify("d904108282") + _uint(2**40) + _uint(2**24)
    with pytest.raises(CBORDecodeValueError):
        __import__("cbor2").loads(stream, max_elements=1_000_000)


def test_huge_typed_array_rejected_before_body():
    for tag, itemsize in ((64, 1), (65, 2), (86, 8)):
        head = bytes([0xd8, tag, 0x5b]) + struct.pack(
            ">Q", (2**40) * itemsize
        )
        with pytest.raises(CBORDecodeValueError):
            __import__("cbor2").loads(head, max_elements=1_000_000)


def test_nested_backing_rejected(impl):
    nested = impl.dumps(
        __import__("cbor2").CBORTag(1040, [[1, 2], [[1], [2]]])
    )
    with pytest.raises(CBORDecodeValueError):
        impl.loads(nested)


def test_zero_dimension_typed_array_backing(impl):
    from cbor2 import CBORTag

    empty_typed = impl.loads(impl.dumps(CBORTag(1040, [[0, 5], TypedArray(65, [])])))
    assert empty_typed.shape == (0, 5)
    assert isinstance(empty_typed.data, TypedArray)
    assert empty_typed.tolist() == []

    scalar_typed = impl.loads(impl.dumps(CBORTag(40, [[], TypedArray(85, [-2.0])])))
    assert scalar_typed.scalar
    assert scalar_typed.order == F_CONTIGUOUS
    assert scalar_typed[()] == -2.0


def test_malformed_multidim_wrappers(impl):
    for hexstr in (
        "d9041001",       # not an array
        "d9041083010203", # 3-element wrapper
    ):
        with pytest.raises(CBORDecodeValueError):
            impl.loads(unhexlify(hexstr))


def test_decoder_limit_validation(impl):
    from io import BytesIO

    with pytest.raises(ValueError):
        impl.CBORDecoder(BytesIO(b""), max_ndim=-1)
    with pytest.raises(ValueError):
        impl.CBORDecoder(BytesIO(b""), max_elements=-2)


def test_constructor_validates_shape():
    with pytest.raises(ValueError):
        NDArray((-1,), [], C_CONTIGUOUS)
    with pytest.raises(ValueError):
        NDArray((3, 2), list(range(5)), C_CONTIGUOUS)
    with pytest.raises(ValueError):
        NDArray((), [], C_CONTIGUOUS)


def test_ndarray_equality(impl):
    arr = NDArray((2, 2), [1, 2, 3, 4], C_CONTIGUOUS)
    assert arr == [[1, 2], [3, 4]]
    assert arr != [[1, 2], [3, 5]]
    assert arr != [[1, 2]]
    decoded = impl.loads(impl.dumps(arr))
    assert decoded == arr
