from __future__ import annotations

import math
import sys
import threading
from collections import namedtuple
from collections.abc import Iterable, Iterator, Sequence
from enum import Enum
from functools import total_ordering
from itertools import product
from reprlib import recursive_repr
from reprlib import repr as limited_repr
from typing import Any, Mapping, TypeVar

KT = TypeVar("KT")
VT_co = TypeVar("VT_co", covariant=True)

thread_locals = threading.local()

#: Default maximum number of dimensions accepted for a :class:`MultiDimArray`
DEFAULT_MAX_MULTIDIM_DIMENSIONS = 32

#: Default maximum number of elements accepted for a :class:`MultiDimArray`
DEFAULT_MAX_MULTIDIM_ELEMENTS = sys.maxsize


class CBORError(Exception):
    """Base class for errors that occur during CBOR encoding or decoding."""


class CBOREncodeError(CBORError):
    """Raised for exceptions occurring during CBOR encoding."""


class CBOREncodeTypeError(CBOREncodeError, TypeError):
    """Raised when attempting to encode a type that cannot be serialized."""


class CBOREncodeValueError(CBOREncodeError, ValueError):
    """Raised when the CBOR encoder encounters an invalid value."""


class CBORDecodeError(CBORError):
    """Raised for exceptions occurring during CBOR decoding."""


class CBORDecodeValueError(CBORDecodeError, ValueError):
    """Raised when the CBOR stream being decoded contains an invalid value."""


class CBORDecodeEOF(CBORDecodeError, EOFError):
    """Raised when decoding unexpectedly reaches EOF."""


@total_ordering
class CBORTag:
    """
    Represents a CBOR semantic tag.

    :param int tag: tag number
    :param value: encapsulated value (any object)
    """

    __slots__ = "tag", "value"

    def __init__(self, tag: str | int, value: Any) -> None:
        if not isinstance(tag, int) or tag not in range(2**64):
            raise TypeError("CBORTag tags must be positive integers less than 2**64")
        self.tag = tag
        self.value = value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CBORTag):
            return (self.tag, self.value) == (other.tag, other.value)

        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, CBORTag):
            return (self.tag, self.value) <= (other.tag, other.value)

        return NotImplemented

    @recursive_repr()
    def __repr__(self) -> str:
        return f"CBORTag({self.tag}, {self.value!r})"

    def __hash__(self) -> int:
        self_id = id(self)
        try:
            running_hashes = thread_locals.running_hashes
        except AttributeError:
            running_hashes = thread_locals.running_hashes = set()

        if self_id in running_hashes:
            raise RuntimeError(
                "This CBORTag is not hashable because it contains a reference to itself"
            )

        running_hashes.add(self_id)
        try:
            return hash((self.tag, self.value))
        finally:
            running_hashes.remove(self_id)
            if not running_hashes:
                del thread_locals.running_hashes


class CBORSimpleValue(namedtuple("CBORSimpleValue", ["value"])):
    """
    Represents a CBOR "simple value".

    :param int value: the value (0-255)
    """

    __slots__ = ()

    value: int

    def __hash__(self) -> int:
        return hash(self.value)

    def __new__(cls, value: int) -> CBORSimpleValue:
        if value < 0 or value > 255 or 23 < value < 32:
            raise TypeError("simple value out of range (0..23, 32..255)")

        return super().__new__(cls, value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.value == other
        elif isinstance(other, CBORSimpleValue):
            return self.value == other.value

        return NotImplemented

    def __ne__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.value != other
        elif isinstance(other, CBORSimpleValue):
            return self.value != other.value

        return NotImplemented

    def __lt__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.value < other
        elif isinstance(other, CBORSimpleValue):
            return self.value < other.value

        return NotImplemented

    def __le__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.value <= other
        elif isinstance(other, CBORSimpleValue):
            return self.value <= other.value

        return NotImplemented

    def __ge__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.value >= other
        elif isinstance(other, CBORSimpleValue):
            return self.value >= other.value

        return NotImplemented

    def __gt__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.value > other
        elif isinstance(other, CBORSimpleValue):
            return self.value > other.value

        return NotImplemented


class FrozenDict(Mapping[KT, VT_co]):
    """
    A hashable, immutable mapping type.

    The arguments to ``FrozenDict`` are processed just like those to ``dict``.
    """

    def __init__(self, *args: Mapping[KT, VT_co] | Iterable[tuple[KT, VT_co]]) -> None:
        self._d: dict[KT, VT_co] = dict(*args)
        self._hash: int | None = None

    def __iter__(self) -> Iterator[KT]:
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __getitem__(self, key: KT) -> VT_co:
        return self._d[key]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._d})"

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = hash((frozenset(self), frozenset(self.values())))

        return self._hash


class MultiDimArrayLayout(Enum):
    """
    Memory layout of an RFC 8746 multi-dimensional array.

    The enum values are the corresponding CBOR tag numbers.
    """

    #: Row-major (C-style) layout; CBOR tag 40
    ROW_MAJOR = 40

    #: Column-major (Fortran-style) layout; CBOR tag 1040
    COLUMN_MAJOR = 1040


class MultiDimArray(Sequence[Any]):
    """
    Represents an `RFC 8746`_ multi-dimensional array (CBOR tags 40 and 1040).

    The array keeps the flat underlying data (either a plain sequence or a
    typed :class:`array.array`) along with its dimensions and layout. The data
    is **not** expanded into nested lists on decoding; use coordinate indexing
    or :meth:`tolist` to access the elements.

    Indexing works with a coordinate tuple of ``ndim`` (or fewer) integers,
    e.g. ``array[1, 2]``. A restricted form of slicing is also supported: full
    slices (``:``) may be used in place of integers as long as the selected
    elements remain contiguous in memory (for row-major arrays, integer
    indices must precede slice axes; for column-major arrays they must follow
    them). Slices return a read-only view that shares the underlying data
    with the original array instead of copying it.

    Zero-length dimensions are allowed (as per RFC 8746); such arrays contain
    no elements. An empty dimensions tuple represents a 0-dimensional (scalar)
    array holding exactly one element.

    :param data: a flat sequence of elements in layout order
    :param dimensions: sequence of non-negative dimension lengths
    :param layout: the memory layout of the data
    :param max_dimensions: maximum allowed number of dimensions
    :param max_elements: maximum allowed number of elements (the product of
        the dimensions)

    .. _RFC 8746: https://www.rfc-editor.org/rfc/rfc8746.html
    """

    __slots__ = ("_data", "_dimensions", "_layout", "_offset")

    def __init__(
        self,
        data: Sequence[Any],
        dimensions: Iterable[int],
        layout: MultiDimArrayLayout = MultiDimArrayLayout.ROW_MAJOR,
        *,
        max_dimensions: int = DEFAULT_MAX_MULTIDIM_DIMENSIONS,
        max_elements: int = DEFAULT_MAX_MULTIDIM_ELEMENTS,
    ) -> None:
        if isinstance(data, (str, bytes, bytearray)) or not (
            hasattr(data, "__len__") and hasattr(data, "__getitem__")
        ):
            raise TypeError("data must be a sequence of elements")
        if not isinstance(layout, MultiDimArrayLayout):
            raise TypeError("layout must be a MultiDimArrayLayout")

        dims = tuple(dimensions)
        for dim in dims:
            if not isinstance(dim, int) or isinstance(dim, bool):
                raise TypeError(f"dimensions must be integers, not {dim!r}")
            if dim < 0:
                raise ValueError(f"dimensions must be non-negative, got {dim}")

        if len(dims) > max_dimensions:
            raise ValueError(
                f"too many dimensions ({len(dims)} > {max_dimensions})"
            )

        # Computed as a Python big integer, so an absurd product can never
        # trigger a huge memory allocation by itself
        size = math.prod(dims)
        if size > max_elements:
            raise ValueError(
                f"too many elements ({size} > {max_elements})"
            )
        if size != len(data):
            raise ValueError(
                f"the product of the dimensions {dims!r} ({size}) does not match "
                f"the number of elements ({len(data)})"
            )

        self._data = data
        self._dimensions = dims
        self._layout = layout
        self._offset = 0

    @classmethod
    def _view(
        cls,
        data: Sequence[Any],
        dimensions: tuple[int, ...],
        layout: MultiDimArrayLayout,
        offset: int,
    ) -> MultiDimArray:
        """Create a view sharing ``data`` without re-validating or copying."""
        view = cls.__new__(cls)
        view._data = data
        view._dimensions = dimensions
        view._layout = layout
        view._offset = offset
        return view

    @property
    def data(self) -> Sequence[Any]:
        """The underlying flat sequence of elements (shared with any views)."""
        return self._data

    @property
    def dimensions(self) -> tuple[int, ...]:
        """The lengths of the array's axes."""
        return self._dimensions

    @property
    def layout(self) -> MultiDimArrayLayout:
        """The memory layout of the array."""
        return self._layout

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return len(self._dimensions)

    @property
    def size(self) -> int:
        """The total number of elements (the product of the dimensions)."""
        return math.prod(self._dimensions)

    @property
    def _strides(self) -> tuple[int, ...]:
        strides = []
        stride = 1
        if self._layout is MultiDimArrayLayout.ROW_MAJOR:
            for dim in reversed(self._dimensions):
                strides.append(stride)
                stride *= dim
            strides.reverse()
        else:
            for dim in self._dimensions:
                strides.append(stride)
                stride *= dim

        return tuple(strides)

    def _memory_offsets(self) -> Iterator[int]:
        """Yield the flat offsets of all elements in memory (layout) order."""
        strides = self._strides
        ranges = [range(dim) for dim in self._dimensions]
        if self._layout is MultiDimArrayLayout.COLUMN_MAJOR:
            ranges.reverse()
            strides = tuple(reversed(strides))

        for coordinates in product(*ranges):
            yield self._offset + sum(
                coordinate * stride for coordinate, stride in zip(coordinates, strides)
            )

    def flat_data(self) -> Sequence[Any]:
        """
        Return the elements as a flat sequence in memory (layout) order.

        For top-level arrays this is the underlying sequence itself; for
        views a new list is materialized.
        """
        if self._offset == 0 and len(self._data) == self.size:
            return self._data

        return [self._data[offset] for offset in self._memory_offsets()]

    def tolist(self) -> Any:
        """
        Expand the array into (possibly nested) lists in logical order.

        A 0-dimensional array is converted to its single element.
        """
        if not self._dimensions:
            return self._data[self._offset]

        strides = self._strides

        def build(axis: int, offset: int) -> Any:
            if axis == len(self._dimensions):
                return self._data[offset]

            return [
                build(axis + 1, offset + index * strides[axis])
                for index in range(self._dimensions[axis])
            ]

        return build(0, self._offset)

    def __len__(self) -> int:
        if not self._dimensions:
            raise TypeError("len() of a 0-dimensional array")

        return self._dimensions[0]

    def __getitem__(self, index: int | slice | tuple[int | slice, ...]) -> Any:
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) > self.ndim:
            raise IndexError(
                f"too many indices for a {self.ndim}-dimensional array"
            )

        # Pad missing trailing axes with full slices
        index = index + (slice(None),) * (self.ndim - len(index))

        offset = self._offset
        new_dimensions: list[int] = []
        seen_slice = False
        seen_int = False
        for axis, (item, dim, stride) in enumerate(
            zip(index, self._dimensions, self._strides)
        ):
            if isinstance(item, slice):
                if item.start is not None or item.stop is not None or item.step is not None:
                    raise ValueError(
                        "only full slices (:) are supported; partial slices would "
                        "require copying data"
                    )
                if seen_int and self._layout is MultiDimArrayLayout.COLUMN_MAJOR:
                    raise ValueError(
                        "integer indices must follow slice axes in a column-major "
                        "array (non-contiguous views are not supported)"
                    )

                seen_slice = True
                new_dimensions.append(dim)
            elif isinstance(item, int):
                if seen_slice and self._layout is MultiDimArrayLayout.ROW_MAJOR:
                    raise ValueError(
                        "integer indices must precede slice axes in a row-major "
                        "array (non-contiguous views are not supported)"
                    )

                seen_int = True
                coordinate = item + dim if item < 0 else item
                if not 0 <= coordinate < dim:
                    raise IndexError(
                        f"index {item} is out of bounds for axis {axis} with "
                        f"dimension {dim}"
                    )

                offset += coordinate * stride
            else:
                raise TypeError(
                    f"indices must be integers or full slices, not {item!r}"
                )

        if not new_dimensions:
            return self._data[offset]

        return MultiDimArray._view(
            self._data, tuple(new_dimensions), self._layout, offset
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MultiDimArray):
            return (
                self._layout == other._layout
                and self._dimensions == other._dimensions
                and self._elements() == other._elements()
            )

        return NotImplemented

    def _elements(self) -> list[Any]:
        return [self._data[offset] for offset in self._memory_offsets()]

    @recursive_repr()
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({limited_repr(self.flat_data())}, "
            f"{self._dimensions!r}, {self._layout!r})"
        )


class UndefinedType:
    __slots__ = ()

    def __new__(cls: type[UndefinedType]) -> UndefinedType:
        try:
            return undefined
        except NameError:
            return super().__new__(cls)

    def __repr__(self) -> str:
        return "undefined"

    def __bool__(self) -> bool:
        return False


class BreakMarkerType:
    __slots__ = ()

    def __new__(cls: type[BreakMarkerType]) -> BreakMarkerType:
        try:
            return break_marker
        except NameError:
            return super().__new__(cls)

    def __repr__(self) -> str:
        return "break_marker"

    def __bool__(self) -> bool:
        return True


#: Represents the "undefined" value.
undefined = UndefinedType()
break_marker = BreakMarkerType()
