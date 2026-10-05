from __future__ import annotations

import sys
from array import array
from binascii import unhexlify

import pytest

from cbor2._decoder import loads
from cbor2._encoder import dumps
from cbor2._types import (
    CBORDecodeValueError,
    CBOREncodeTypeError,
    CBORTag,
    MultiDimArray,
    MultiDimArrayLayout,
)

#: tag 40, [[2, 3], [1, 2, 3, 4, 5, 6]] (row-major 2x3)
ROW_MAJOR_HEX = "d8288282020386010203040506"
#: tag 1040, [[2, 3], [1, 4, 2, 5, 3, 6]] (column-major 2x3, same logical matrix)
COLUMN_MAJOR_HEX = "d904108282020386010402050306"


def test_decode_row_major():
    arr = loads(unhexlify(ROW_MAJOR_HEX))
    assert isinstance(arr, MultiDimArray)
    assert arr.layout is MultiDimArrayLayout.ROW_MAJOR
    assert arr.dimensions == (2, 3)
    assert arr.ndim == 2
    assert arr.size == 6
    assert arr.data == [1, 2, 3, 4, 5, 6]
    assert arr.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_decode_column_major():
    arr = loads(unhexlify(COLUMN_MAJOR_HEX))
    assert isinstance(arr, MultiDimArray)
    assert arr.layout is MultiDimArrayLayout.COLUMN_MAJOR
    assert arr.dimensions == (2, 3)
    assert arr.data == [1, 4, 2, 5, 3, 6]
    assert arr.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_layouts_agree_on_coordinates():
    """Both layouts of the same non-square matrix yield the same values."""
    row_major = loads(unhexlify(ROW_MAJOR_HEX))
    column_major = loads(unhexlify(COLUMN_MAJOR_HEX))
    assert row_major.dimensions == column_major.dimensions
    for i in range(2):
        for j in range(3):
            assert row_major[i, j] == column_major[i, j]


def test_coordinate_indexing():
    arr = loads(unhexlify(ROW_MAJOR_HEX))
    assert arr[0, 0] == 1
    assert arr[0, 2] == 3
    assert arr[1, 0] == 4
    assert arr[1, 2] == 6
    # negative indices count from the end of the axis
    assert arr[-1, -1] == 6
    assert arr[-2, 1] == 2
    with pytest.raises(IndexError):
        arr[2, 0]
    with pytest.raises(IndexError):
        arr[0, -4]
    with pytest.raises(IndexError):
        arr[0, 0, 0]
    with pytest.raises(TypeError):
        arr[0, "x"]


def test_decode_typed_array_backing():
    # tag 40, [[2, 3], tag 64 (uint8) h'010203040506']
    arr = loads(unhexlify("d82882820203d84046010203040506"))
    assert isinstance(arr.data, array)
    assert arr.data.typecode == "B"
    assert arr[1, 2] == 6
    assert arr.tolist() == [[1, 2, 3], [4, 5, 6]]


@pytest.mark.parametrize(
    "payload, typecode, values",
    [
        ("d84043010203", "B", [1, 2, 3]),  # tag 64: uint8
        ("d84443010203", "B", [1, 2, 3]),  # tag 68: uint8 (clamped)
        ("d8414400030007", "H", [3, 7]),  # tag 65: uint16be
        ("d8454401000200", "H", [1, 2]),  # tag 69: uint16le
        ("d842480000000300000007", "I", [3, 7]),  # tag 66: uint32be
        ("d8484203ff", "b", [3, -1]),  # tag 72: sint8
        ("d84944fffe0007", "h", [-2, 7]),  # tag 73: sint16be
        ("d84d44feff0700", "h", [-2, 7]),  # tag 77: sint16le
        ("d850443c00c000", "f", [1.0, -2.0]),  # tag 80: float16be
        ("d851483f800000c0000000", "f", [1.0, -2.0]),  # tag 81: float32be
        ("d852503ff0000000000000c000000000000000", "d", [1.0, -2.0]),  # tag 82
        ("d856480000803f000000c0", "f", [1.0, -2.0]),  # tag 86: float32le
    ],
)
def test_decode_typed_arrays(payload, typecode, values):
    decoded = loads(unhexlify(payload))
    assert isinstance(decoded, array)
    assert decoded.typecode == typecode
    assert list(decoded) == values


def test_decode_typed_array_invalid_payload():
    # payload is not a byte string
    with pytest.raises(CBORDecodeValueError, match="invalid typed array payload"):
        loads(unhexlify("d84001"))
    # payload length is not a multiple of the element size
    with pytest.raises(CBORDecodeValueError, match="not a multiple"):
        loads(unhexlify("d84143010203"))


def test_decode_unsupported_typed_array_tag():
    # tag 83 (float128) has no built-in decoder and stays a CBORTag
    decoded = loads(unhexlify("d85350" + "00" * 16))
    assert isinstance(decoded, CBORTag)
    assert decoded.tag == 83


def test_slice_view_shares_data():
    arr = loads(unhexlify(ROW_MAJOR_HEX))
    view = arr[1]
    assert isinstance(view, MultiDimArray)
    assert view.dimensions == (3,)
    assert view.data is arr.data  # no copy of the underlying data
    assert list(view) == [4, 5, 6]
    assert view.tolist() == [4, 5, 6]

    view2d = arr[0, :]
    assert view2d.dimensions == (3,)
    assert view2d.data is arr.data
    assert view2d.tolist() == [1, 2, 3]

    col = loads(unhexlify(COLUMN_MAJOR_HEX))
    col_view = col[:, 1]
    assert col_view.dimensions == (2,)
    assert col_view.data is col.data
    assert col_view.tolist() == [2, 5]


def test_views_are_read_only():
    arr = loads(unhexlify(ROW_MAJOR_HEX))
    view = arr[0]
    with pytest.raises(TypeError):
        view[0] = 42
    with pytest.raises(TypeError):
        arr[0, 0] = 42


def test_non_contiguous_slices_rejected():
    row_major = loads(unhexlify(ROW_MAJOR_HEX))
    with pytest.raises(ValueError, match="non-contiguous"):
        row_major[:, 1]
    column_major = loads(unhexlify(COLUMN_MAJOR_HEX))
    with pytest.raises(ValueError, match="non-contiguous"):
        column_major[1, :]


def test_partial_slices_rejected():
    arr = loads(unhexlify(ROW_MAJOR_HEX))
    with pytest.raises(ValueError, match="full slices"):
        arr[0:1]
    with pytest.raises(ValueError, match="full slices"):
        arr[::2]


def test_iteration_yields_views():
    arr = loads(unhexlify(ROW_MAJOR_HEX))
    rows = list(arr)
    assert len(rows) == 2
    assert all(isinstance(row, MultiDimArray) for row in rows)
    assert rows[0].tolist() == [1, 2, 3]
    assert rows[1].tolist() == [4, 5, 6]
    assert len(arr) == 2


def test_scalar_array():
    # tag 40, [[], [42]]: a 0-dimensional array
    arr = loads(unhexlify("d828828081182a"))
    assert arr.ndim == 0
    assert arr.dimensions == ()
    assert arr.size == 1
    assert arr[()] == 42
    assert arr.tolist() == 42
    with pytest.raises(TypeError):
        len(arr)
    with pytest.raises(IndexError):
        arr[0]
    assert dumps(arr) == unhexlify("d828828081182a")


def test_zero_length_dimension():
    # tag 40, [[0, 3], []]: zero-length axis means zero elements
    arr = loads(unhexlify("d8288282000380"))
    assert arr.dimensions == (0, 3)
    assert arr.size == 0
    assert arr.tolist() == []
    with pytest.raises(IndexError):
        arr[0, 0]
    assert loads(dumps(arr)) == arr


@pytest.mark.parametrize(
    "payload, match",
    [
        ("d82801", "2-element array"),  # content is not an array
        ("d82883010203", "2-element array"),  # 3-element content
        ("d828820186010203040506", "dimensions must be an array"),
        ("d8288282020301", "data must be a sequence"),  # data not a sequence
    ],
)
def test_decode_invalid_payloads(payload, match):
    with pytest.raises(CBORDecodeValueError, match=match):
        loads(unhexlify(payload))


def test_decode_negative_dimension():
    # tag 40, [[-1, 3], [...]]
    with pytest.raises(CBORDecodeValueError, match="non-negative"):
        loads(unhexlify("d8288282200386" + "00" * 6))


def test_decode_non_integer_dimension():
    # tag 40, [[true, 3], [...]] and [["x", 3], [...]]
    with pytest.raises(CBORDecodeValueError, match="must be integers"):
        loads(unhexlify("d8288282f50380"))
    with pytest.raises(CBORDecodeValueError, match="must be integers"):
        loads(unhexlify("d828828261780380"))


def test_decode_product_mismatch():
    # dimensions (2, 3) but only 5 elements
    with pytest.raises(CBORDecodeValueError, match="does not match"):
        loads(unhexlify("d82882820203850102030405"))


def test_decode_too_many_dimensions():
    payload = unhexlify("d828828301020386010203040506")
    with pytest.raises(CBORDecodeValueError, match="too many dimensions"):
        loads(payload, max_multidim_dimensions=2)
    # the default limit accepts 3 dimensions
    assert loads(payload).dimensions == (1, 2, 3)


def test_decode_too_many_elements():
    with pytest.raises(CBORDecodeValueError, match="too many elements"):
        loads(unhexlify(ROW_MAJOR_HEX), max_multidim_elements=5)
    assert loads(unhexlify(ROW_MAJOR_HEX), max_multidim_elements=6).size == 6


def test_decode_huge_dimensions_no_memory_exhaustion():
    # (2**62, 2**62) overflows the default element limit; this must fail
    # immediately without attempting a huge allocation
    payload = unhexlify("d82882" "82" "1b4000000000000000" "1b4000000000000000" "80")
    with pytest.raises(CBORDecodeValueError, match="too many elements"):
        loads(payload)


def test_decoder_limit_validation():
    from io import BytesIO

    from cbor2._decoder import CBORDecoder

    with pytest.raises(ValueError):
        CBORDecoder(BytesIO(b""), max_multidim_dimensions=-1)
    with pytest.raises(ValueError):
        CBORDecoder(BytesIO(b""), max_multidim_elements=-1)


def test_constructor_validation():
    with pytest.raises(TypeError):
        MultiDimArray("not a sequence", (1,))
    with pytest.raises(TypeError):
        MultiDimArray([1], (1,), "row-major")
    with pytest.raises(ValueError, match="non-negative"):
        MultiDimArray([], (-1,))
    with pytest.raises(ValueError, match="does not match"):
        MultiDimArray([1, 2, 3], (2, 2))
    with pytest.raises(ValueError, match="too many dimensions"):
        MultiDimArray([1], (1, 1, 1), max_dimensions=2)
    with pytest.raises(ValueError, match="too many elements"):
        MultiDimArray([], (2**62, 2**62))


def test_encode_roundtrip():
    for payload in (ROW_MAJOR_HEX, COLUMN_MAJOR_HEX):
        arr = loads(unhexlify(payload))
        assert dumps(arr) == unhexlify(payload)


def test_encode_typed_array_backing():
    arr = MultiDimArray(array("B", [1, 2, 3, 4, 5, 6]), (2, 3))
    decoded = loads(dumps(arr))
    assert decoded == arr
    assert isinstance(decoded.data, array)
    assert list(decoded.data) == [1, 2, 3, 4, 5, 6]


def test_encode_typed_array():
    assert dumps(array("B", [1, 2, 3])) == unhexlify("d84043010203")
    if sys.byteorder == "little":
        assert dumps(array("H", [1, 2])) == unhexlify("d8454401000200")
    else:
        assert dumps(array("H", [1, 2])) == unhexlify("d8414400010002")

    for typecode, values in [
        ("H", [1, 2]),
        ("i", [1, -2, 3]),
        ("q", [5]),
        ("f", [0.5]),
        ("d", [1.0, -2.0]),
    ]:
        decoded = loads(dumps(array(typecode, values)))
        assert isinstance(decoded, array)
        assert decoded.itemsize == array(typecode).itemsize
        assert list(decoded) == values


def test_encode_unsupported_typecode():
    with pytest.raises(CBOREncodeTypeError, match="typecode"):
        dumps(array("u", "abc"))


def test_encode_view_materializes_elements():
    arr = loads(unhexlify(ROW_MAJOR_HEX))
    view = arr[1]
    assert dumps(view) == unhexlify("d828828103830405 06".replace(" ", ""))
    assert loads(dumps(view)) == view


def test_equality_and_repr():
    arr = loads(unhexlify(ROW_MAJOR_HEX))
    same = loads(unhexlify(ROW_MAJOR_HEX))
    other = loads(unhexlify(COLUMN_MAJOR_HEX))
    assert arr == same
    assert arr != other
    assert arr != [1, 2, 3, 4, 5, 6]
    assert "MultiDimArray" in repr(arr)
    assert "(2, 3)" in repr(arr)
