"""
Support for RFC 8746 typed numeric arrays and multidimensional arrays.

This module is implemented without NumPy.  Typed numeric arrays (CBOR tags
64 through 86, assigned by RFC 8746) are represented by :class:`TypedArray`,
while multidimensional arrays (CBOR tag 40 for column-major / Fortran-order
arrays and tag 1040 for row-major / C-order arrays) are represented by
:class:`NDArray`.

Decoding never expands an array into nested lists: the dimensions and layout
are retained on the object, and individual elements are only materialized on
demand through indexing, iteration or :meth:`NDArray.tolist`.
"""

from __future__ import annotations

import struct
from array import array
from collections.abc import Sequence
from typing import Any, Iterator

from ._types import CBOREncodeValueError, CBORDecodeValueError

#: Row-major (C order) layout.
C_CONTIGUOUS = "C"

#: Column-major (Fortran order) layout.
F_CONTIGUOUS = "F"

# Mapping of RFC 8746 tag -> tuple of
# (native ``array.array`` typecode used for storage, :mod:`struct` format of
# the wire encoding, itemsize, is_float, endianness).
# ``endianness`` is "big", "little" or "native".  Floating point tags are
# only defined in big-endian form (f16/f32/f64).
TAG_SPECS: dict[int, tuple[str, str, int, bool, str]] = {
    64: ("B", "B", 1, False, "native"),   # uint8
    65: ("H", ">H", 2, False, "big"),     # uint16, big endian
    66: ("I", ">I", 4, False, "big"),     # uint32, big endian
    67: ("Q", ">Q", 8, False, "big"),     # uint64, big endian
    68: ("H", "H", 2, False, "native"),   # uint16, native endian
    69: ("I", "I", 4, False, "native"),   # uint32, native endian
    70: ("Q", "Q", 8, False, "native"),   # uint64, native endian
    71: ("H", "<H", 2, False, "little"),  # uint16, little endian
    72: ("I", "<I", 4, False, "little"),  # uint32, little endian
    73: ("Q", "<Q", 8, False, "little"),  # uint64, little endian
    74: ("b", "b", 1, True, "native"),    # sint8
    75: ("h", ">h", 2, True, "big"),      # sint16, big endian
    76: ("i", ">i", 4, True, "big"),      # sint32, big endian
    77: ("q", ">q", 8, True, "big"),      # sint64, big endian
    78: ("h", "h", 2, True, "native"),    # sint16, native endian
    79: ("i", "i", 4, True, "native"),    # sint32, native endian
    80: ("q", "q", 8, True, "native"),    # sint64, native endian
    81: ("h", "<h", 2, True, "little"),   # sint16, little endian
    82: ("i", "<i", 4, True, "little"),   # sint32, little endian
    83: ("q", "<q", 8, True, "little"),   # sint64, little endian
    84: ("f", ">e", 2, False, "big"),     # IEEE 754 16-bit, big endian
    85: ("f", ">f", 4, False, "big"),     # IEEE 754 32-bit, big endian
    86: ("d", ">d", 8, False, "big"),     # IEEE 754 64-bit, big endian
}

# Mapping from a native ``array.array`` typecode to an RFC 8746 native-endian
# tag (when one is unambiguously available on the current platform).
ARRAY_CODE_TAGS: dict[str, int] = {}
for _code, _tag, _size in (
    ("b", 74, 1),
    ("B", 64, 1),
    ("h", 78, 2),
    ("H", 68, 2),
    ("i", 79, 4),
    ("I", 69, 4),
    ("l", 79, 4),
    ("L", 69, 4),
    ("q", 80, 8),
    ("Q", 70, 8),
):
    if array(_code).itemsize == _size:
        ARRAY_CODE_TAGS[_code] = _tag

ARRAY_CODE_TAGS["f"] = 85
ARRAY_CODE_TAGS["d"] = 86

_NATIVE_LITTLE = struct.pack("H", 1)[0] == 1


def _convert_order(raw: bytes, wire_fmt: str, to_native: bool) -> bytes:
    """Convert element bytes between wire byte order and native byte order."""
    prefix, char = wire_fmt[0], wire_fmt[-1]
    itemsize = struct.calcsize(wire_fmt)
    count = len(raw) // itemsize
    if count == 0 or prefix not in "<>":
        return raw

    if to_native:
        source_little = prefix == "<"
        if source_little == _NATIVE_LITTLE:
            return raw

        values = struct.unpack(prefix + char * count, raw)
        out_prefix = "<" if _NATIVE_LITTLE else ">"
    else:
        target_little = prefix == "<"
        if target_little == _NATIVE_LITTLE:
            return raw

        native_prefix = "<" if _NATIVE_LITTLE else ">"
        values = struct.unpack(native_prefix + char * count, raw)
        out_prefix = prefix

    return struct.pack(out_prefix + char * count, *values)


class TypedArray:
    """
    A read-only, one-dimensional RFC 8746 typed numeric array.

    Elements are backed by an :class:`array.array` and are not copied when
    sliced.  Instances behave like immutable sequences: they support indexing,
    slicing, iteration, ``len()`` and equality against other sequences.

    Half-precision (f16, tag 84) values are stored widened to f32 as no
    native half-float container exists.

    :param tag: the RFC 8746 tag number (64-86) defining the element type
    :param values: an iterable of initial element values
    """

    __slots__ = ("tag", "_values")

    tag: int
    _values: array[Any]

    def __init__(self, tag: int, values: Any = ()) -> None:
        if tag not in TAG_SPECS:
            raise CBOREncodeValueError(f"unknown typed array tag {tag}")

        typecode = TAG_SPECS[tag][0]
        if tag == 84:
            self._values = array("f", values)
        elif isinstance(values, array):
            if values.typecode != typecode:
                raise CBOREncodeValueError(
                    f"typed array tag {tag} requires typecode {typecode!r}, "
                    f"got {values.typecode!r}"
                )
            self._values = values
        else:
            self._values = array(typecode, values)

        self.tag = tag

    @classmethod
    def _from_bytes(cls, tag: int, raw: bytes) -> TypedArray:
        typecode, wire_fmt, itemsize, _is_float, _endian = TAG_SPECS[tag]
        if len(raw) % itemsize:
            raise CBORDecodeValueError(
                f"typed array payload length {len(raw)} is not a multiple of "
                f"the {itemsize}-byte element size for tag {tag}"
            )

        obj = object.__new__(cls)
        obj.tag = tag
        if tag == 84:
            count = len(raw) // 2
            obj._values = (
                array("f", struct.unpack(">" + "e" * count, raw)) if count else array("f")
            )
        else:
            values = array(typecode)
            values.frombytes(_convert_order(raw, wire_fmt, to_native=True))
            obj._values = values

        return obj

    @property
    def itemsize(self) -> int:
        """Size in bytes of a single element."""
        return TAG_SPECS[self.tag][2]

    @property
    def is_float(self) -> bool:
        """``True`` if the element type is a floating point type."""
        return TAG_SPECS[self.tag][3]

    def to_wire_bytes(self) -> bytes:
        """Return the element data encoded according to the tag's wire format."""
        _typecode, wire_fmt, _itemsize, _is_float, _endian = TAG_SPECS[self.tag]
        if self.tag == 84:
            count = len(self._values)
            return struct.pack(">" + "e" * count, *self._values) if count else b""

        return _convert_order(bytes(self._values), wire_fmt, to_native=False)

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self._values))
            if step != 1:
                raise ValueError("typed array slices must have a step of 1")

            return self._make_view(self._values[start:stop])

        return self._values[index]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypedArray):
            other_values: Any = other._values
        elif isinstance(other, (array, Sequence)):
            other_values = other
        else:
            return NotImplemented

        return len(self._values) == len(other_values) and all(
            a == b for a, b in zip(self._values, other_values)
        )

    def __repr__(self) -> str:
        items = ", ".join(repr(v) for v in list(self._values)[:10])
        if len(self._values) > 10:
            items += ", ..."

        return f"TypedArray(tag={self.tag}, [{items}])"

    def _make_view(self, values: array[Any]) -> TypedArray:
        obj = object.__new__(TypedArray)
        obj.tag = self.tag
        obj._values = values
        return obj


def validate_shape(
    shape: Any,
    n_elements: int | None = None,
    max_ndim: int | None = None,
    max_elements: int | None = None,
) -> tuple[int, ...]:
    """
    Validate an array shape.

    :param shape: the raw shape value decoded from the stream
    :param n_elements: if given, the shape product must equal this value
    :param max_ndim: maximum permitted number of dimensions
    :param max_elements: maximum permitted product of the dimensions
    :returns: the shape as a tuple of non-negative ``int`` values
    :raises CBORDecodeValueError: if the shape is invalid or exceeds a limit
    """
    if not isinstance(shape, (list, tuple)):
        raise CBORDecodeValueError("multidimensional array dimensions must be an array")

    if max_ndim is not None and len(shape) > max_ndim:
        raise CBORDecodeValueError(
            f"multidimensional array has {len(shape)} dimensions, exceeding the "
            f"maximum of {max_ndim}"
        )

    normalized: list[int] = []
    product = 1
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise CBORDecodeValueError("array dimensions must be unsigned integers")
        if dim < 0:
            raise CBORDecodeValueError("array dimensions must not be negative")

        normalized.append(dim)
        product *= dim
        if max_elements is not None and product > max_elements:
            raise CBORDecodeValueError(
                f"multidimensional array element count {product} exceeds the "
                f"maximum of {max_elements}"
            )

    if n_elements is not None and product != n_elements:
        raise CBORDecodeValueError(
            f"shape product {product} does not match element count {n_elements}"
        )

    return tuple(normalized)


class NDArray:
    """
    A multidimensional numeric array as defined by RFC 8746.

    The object retains the original dimensions (:attr:`shape`) and layout
    (:attr:`order`) without expanding the data into nested lists.  Elements
    can be accessed by coordinate (``arr[1, 2]``) and restricted slices return
    :class:`NDArray` views that share the same read-only backing data instead
    of copying it.

    An empty shape (``()``) represents a scalar: the product is 1 and the
    backing contains exactly one element.  A shape containing a zero-length
    dimension represents an empty array: the product is 0 and the backing is
    empty, regardless of the other dimensions.

    :param shape: tuple of non-negative dimension sizes
    :param data: the flat backing values (a :class:`TypedArray` or sequence)
    :param order: :data:`C_CONTIGUOUS` (tag 1040) or :data:`F_CONTIGUOUS`
        (tag 40)
    """

    __slots__ = ("shape", "data", "order", "offset", "strides")

    shape: tuple[int, ...]
    data: TypedArray | Sequence[Any]
    order: str
    offset: int
    strides: tuple[int, ...]

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: TypedArray | Sequence[Any],
        order: str = C_CONTIGUOUS,
    ) -> None:
        shape_tuple = validate_shape(tuple(shape))
        if order not in (C_CONTIGUOUS, F_CONTIGUOUS):
            raise CBOREncodeValueError(f"invalid array order {order!r}")
        try:
            n_elements = len(data)
        except TypeError as exc:
            raise CBOREncodeValueError("array data must have a length") from exc

        product = 1
        for dim in shape_tuple:
            product *= dim

        if n_elements != product:
            raise CBOREncodeValueError(
                f"shape product {product} does not match element count {n_elements}"
            )

        self.shape = shape_tuple
        self.data = data
        self.order = order
        self.offset = 0
        self.strides = _contiguous_strides(shape_tuple, order)

    @classmethod
    def _create(
        cls,
        shape: tuple[int, ...],
        data: TypedArray | Sequence[Any],
        order: str,
        offset: int,
        strides: tuple[int, ...],
    ) -> NDArray:
        obj = object.__new__(cls)
        obj.shape = shape
        obj.data = data
        obj.order = order
        obj.offset = offset
        obj.strides = strides
        return obj

    @property
    def ndim(self) -> int:
        """Number of dimensions (0 for a scalar array)."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Total number of elements (1 for a scalar array)."""
        size = 1
        for dim in self.shape:
            size *= dim

        return size

    @property
    def scalar(self) -> bool:
        """``True`` if this is a zero-dimensional (scalar) array."""
        return not self.shape

    def _flat_index(self, coords: tuple[int, ...]) -> int:
        return self.offset + sum(stride * coord for stride, coord in zip(self.strides, coords))

    def __getitem__(self, key: Any) -> Any:
        if self.scalar:
            if key != () and key is not Ellipsis:
                if not isinstance(key, tuple) or key:
                    raise IndexError("too many indices for scalar array")

            return self.data[self.offset]

        if not isinstance(key, tuple):
            key = (key,)

        if Ellipsis in key:
            raise IndexError("ellipsis indexing is not supported")

        if len(key) > len(self.shape):
            raise IndexError(
                f"too many indices for array: array is {len(self.shape)}-dimensional "
                f"but {len(key)} were indexed"
            )

        # Pad missing trailing indices with full slices.
        padded: list[Any] = list(key)
        while len(padded) < len(self.shape):
            padded.append(slice(None))

        new_shape: list[int] = []
        new_strides: list[int] = []
        offset = self.offset
        all_integers = True
        for dim, stride, indexer in zip(self.shape, self.strides, padded):
            if isinstance(indexer, bool) or not isinstance(indexer, (int, slice)):
                raise TypeError("array indices must be integers or slices")

            if isinstance(indexer, int):
                coord = indexer
                if coord < 0:
                    coord += dim
                if coord < 0 or coord >= dim:
                    raise IndexError(
                        f"index {indexer} is out of bounds for axis with size {dim}"
                    )

                offset += coord * stride
            else:
                all_integers = False
                start, stop, step = indexer.indices(dim)
                if step != 1:
                    raise ValueError("array slices must have a step of 1")

                new_shape.append(stop - start)
                new_strides.append(stride)
                offset += start * stride

        if all_integers and len(padded) == len(self.shape):
            return self.data[offset]

        return NDArray._create(
            tuple(new_shape), self.data, self.order, offset, tuple(new_strides)
        )

    def _elements(self) -> Iterator[Any]:
        """Yield elements in C (row-major) logical order, independent of layout."""
        if self.scalar:
            yield self.data[self.offset]
            return

        coords = [0] * len(self.shape)
        total = self.size
        for _ in range(total):
            yield self.data[self._flat_index(tuple(coords))]
            # Advance coordinates in row-major order.
            for axis in range(len(self.shape) - 1, -1, -1):
                coords[axis] += 1
                if coords[axis] < self.shape[axis]:
                    break

                coords[axis] = 0

    def _elements_layout_order(self) -> Iterator[Any]:
        """Yield elements in the array's declared storage order."""
        if self.order == C_CONTIGUOUS:
            yield from self._elements()
            return

        if self.scalar:
            yield self.data[self.offset]
            return

        coords = [0] * len(self.shape)
        for _ in range(self.size):
            yield self.data[self._flat_index(tuple(coords))]
            for axis in range(len(self.shape)):
                coords[axis] += 1
                if coords[axis] < self.shape[axis]:
                    break

                coords[axis] = 0

    def tolist(self) -> Any:
        """
        Materialize the array into nested Python lists.

        Scalar arrays return their single element; zero-length dimensions are
        represented by empty lists at the appropriate nesting level.
        """
        if self.scalar:
            return self.data[self.offset]

        iterator = self._elements()

        def build(level: int) -> Any:
            dim = self.shape[level]
            if level == len(self.shape) - 1:
                return [next(iterator) for _ in range(dim)]

            return [build(level + 1) for _ in range(dim)]

        return build(0)

    def __iter__(self) -> Iterator[Any]:
        if self.scalar:
            raise TypeError("iteration over a 0-d array")

        return _NDAxisIterator(self)

    def __len__(self) -> int:
        if self.scalar:
            raise TypeError("len() of unsized 0-d array")

        return self.shape[0]

    def __eq__(self, other: object) -> bool:
        if isinstance(other, NDArray):
            if other.shape != self.shape:
                return False

            return all(a == b for a, b in zip(self._elements(), other._elements()))
        elif isinstance(other, (list, tuple)):
            def shape_of(value: Any, level: int) -> bool:
                if level == len(self.shape):
                    return not isinstance(value, (list, tuple))
                if not isinstance(value, (list, tuple)) or len(value) != self.shape[level]:
                    return False

                return all(shape_of(v, level + 1) for v in value)

            if not shape_of(other, 0):
                return False

            def leaves(value: Any) -> Iterator[Any]:
                if isinstance(value, (list, tuple)):
                    for v in value:
                        yield from leaves(v)
                else:
                    yield value

            return all(a == b for a, b in zip(self._elements(), leaves(other)))

        return NotImplemented

    def __repr__(self) -> str:
        return (
            f"NDArray(shape={self.shape!r}, order={self.order!r}, "
            f"data={self.data!r})"
        )


class _NDAxisIterator:
    __slots__ = ("_array", "_coord")

    def __init__(self, arr: NDArray) -> None:
        self._array = arr
        self._coord = 0

    def __iter__(self) -> _NDAxisIterator:
        return self

    def __next__(self) -> Any:
        if self._coord >= self._array.shape[0]:
            raise StopIteration

        value = self._array[self._coord]
        self._coord += 1
        return value


def _contiguous_strides(shape: tuple[int, ...], order: str) -> tuple[int, ...]:
    if not shape:
        return ()

    strides = [0] * len(shape)
    if order == C_CONTIGUOUS:
        running = 1
        for axis in range(len(shape) - 1, -1, -1):
            strides[axis] = running
            running *= shape[axis]
    else:
        running = 1
        for axis in range(len(shape)):
            strides[axis] = running
            running *= shape[axis]

    return tuple(strides)
