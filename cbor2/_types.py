from __future__ import annotations

import struct
import sys
from array import array
import threading
from collections import namedtuple
from collections.abc import Iterable, Iterator
from functools import total_ordering
from reprlib import recursive_repr
from typing import Any, Mapping, TypeVar

KT = TypeVar("KT")
VT_co = TypeVar("VT_co", covariant=True)

thread_locals = threading.local()


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


class TypedArray:
    """
    Represents an RFC 8746 typed array.

    The wire payload bytes, element width, signedness and wire byte order are
    retained without depending on the host machine's byte order.
    """

    __slots__ = ("_tag", "_data", "_width", "_typecode", "_byteorder", "_struct")

    _specs = {
        64: ("B", 1, "B", "little"),
        65: ("H", 2, "H", "little"),
        66: ("I", 4, "I", "big"),
        67: ("Q", 8, "Q", "big"),
        68: ("b", 1, "b", "little"),
        69: ("h", 2, "h", "little"),
        70: ("i", 4, "i", "big"),
        71: ("q", 8, "q", "big"),
        72: ("H", 2, "H", "big"),
        73: ("I", 4, "I", "little"),
        74: ("Q", 8, "Q", "little"),
        75: ("h", 2, "h", "big"),
        76: ("i", 4, "i", "little"),
        77: ("q", 8, "q", "little"),
        80: ("f", 4, "f", "little"),
        81: ("f", 4, "f", "big"),
        82: ("d", 8, "d", "big"),
        83: ("d", 8, "d", "little"),
    }

    def __init__(self, tag: int, data: bytes) -> None:
        try:
            typecode, width, struct_format, byteorder = self._specs[tag]
        except KeyError:
            raise ValueError(f"unknown typed array tag {tag}") from None

        if not isinstance(data, bytes):
            raise TypeError("typed array data must be bytes")
        if len(data) % width:
            raise ValueError("typed array data length is not a multiple of the element width")

        self._tag = tag
        self._data = data
        self._typecode = typecode
        self._width = width
        self._byteorder = byteorder
        self._struct = struct.Struct(self._struct_prefix(byteorder) + struct_format)

    @classmethod
    def from_array(cls, tag: int, values: array) -> TypedArray:
        """Create a typed array from a standard library :class:`array.array`."""
        try:
            typecode, width, _, _ = cls._specs[tag]
        except KeyError:
            raise ValueError(f"unknown typed array tag {tag}") from None

        if values.typecode != typecode or values.itemsize != width:
            raise ValueError("array typecode does not match the typed array tag")

        data = bytes(values)
        target_byteorder = cls._specs[tag][3]
        native_byteorder = "little" if sys.byteorder == "little" else "big"
        if width > 1 and target_byteorder != native_byteorder:
            swapped = array(values.typecode, data)
            swapped.byteswap()
            data = bytes(swapped)

        return cls(tag, data)

    @property
    def tag(self) -> int:
        return self._tag

    @property
    def data(self) -> bytes:
        return self._data

    @property
    def value(self) -> bytes:
        return self._data

    @property
    def typecode(self) -> str:
        return self._typecode

    @property
    def itemsize(self) -> int:
        return self._width

    @property
    def byteorder(self) -> str:
        return self._byteorder

    def __len__(self) -> int:
        return len(self._data) // self._width

    def __getitem__(self, index: int) -> int | float:
        length = len(self)
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise IndexError("typed array index out of range")

        offset = index * self._width
        return self._struct.unpack_from(self._data, offset)[0]

    def __iter__(self) -> Iterator[int | float]:
        for index in range(len(self)):
            yield self[index]

    def to_array(self) -> array:
        """Convert the values to a standard library :class:`array.array`."""
        values = array(self._typecode)
        values.frombytes(self._data)
        native_byteorder = "little" if sys.byteorder == "little" else "big"
        if self._width > 1 and self._byteorder != native_byteorder:
            values.byteswap()

        return values

    def __eq__(self, other: object) -> bool:
        if isinstance(other, TypedArray):
            return self._tag == other._tag and self._data == other._data

        return NotImplemented

    def __repr__(self) -> str:
        values = list(self)
        if len(values) > 6:
            values_text = f"[{', '.join(repr(value) for value in values[:6])}, ...]"
        else:
            values_text = repr(values)

        return f"TypedArray({self._tag}, {values_text})"

    def __hash__(self) -> int:
        return hash((self._tag, self._data))

    @staticmethod
    def _struct_prefix(byteorder: str) -> str:
        return "<" if byteorder == "little" else ">"


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
