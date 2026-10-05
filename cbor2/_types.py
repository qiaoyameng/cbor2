from __future__ import annotations

import struct
import sys
import threading
from array import array
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


class CBORTypedArray:
    """
    Represents an RFC 8746 typed array.

    The raw byte string is retained together with the wire element width and
    byte order described by the semantic tag.

    :param int tag: the RFC 8746 typed array semantic tag
    :param bytes value: the byte string containing the array elements
    """

    __slots__ = ("tag", "value", "itemsize", "byteorder", "_struct", "_typecode")

    _specs: dict[int, tuple[int, str, str]] = {
        64: (1, "big", "B"),
        65: (2, "big", "H"),
        66: (4, "big", "I"),
        67: (8, "big", "Q"),
        68: (1, "little", "B"),
        69: (2, "little", "H"),
        70: (4, "little", "I"),
        71: (8, "little", "Q"),
        72: (1, "big", "b"),
        73: (2, "big", "h"),
        74: (4, "big", "i"),
        75: (8, "big", "q"),
        76: (1, "little", "b"),
        77: (2, "little", "h"),
        78: (4, "little", "i"),
        79: (8, "little", "q"),
        81: (4, "big", "f"),
        82: (8, "big", "d"),
        85: (4, "little", "f"),
        86: (8, "little", "d"),
    }

    _array_typecodes: dict[tuple[int, str], tuple[str, ...]] = {
        (1, "B"): ("B",),
        (2, "H"): ("H",),
        (4, "I"): ("I", "L"),
        (8, "Q"): ("Q",),
        (1, "b"): ("b",),
        (2, "h"): ("h",),
        (4, "i"): ("i", "l"),
        (8, "q"): ("q",),
        (4, "f"): ("f",),
        (8, "d"): ("d",),
    }

    def __init__(self, tag: int, value: bytes) -> None:
        spec = self._specs.get(tag)
        if spec is None:
            raise ValueError(f"unknown RFC 8746 typed array tag {tag}")
        if not isinstance(value, bytes):
            raise TypeError("typed array value must be a byte string")

        itemsize, byteorder, fmt = spec
        if len(value) % itemsize:
            raise ValueError(
                f"typed array payload length ({len(value)}) is not a multiple of the "
                f"element width ({itemsize})"
            )

        self.tag = tag
        self.value = value
        self.itemsize = itemsize
        self.byteorder = byteorder
        self._struct = struct.Struct(("<" if byteorder == "little" else ">") + fmt)
        self._typecode = self._find_typecode(itemsize, fmt)

    @classmethod
    def _find_typecode(cls, itemsize: int, fmt: str) -> str:
        for typecode in cls._array_typecodes[(itemsize, fmt)]:
            if array(typecode).itemsize == itemsize:
                return typecode

        raise TypeError(f"no standard library array type with width {itemsize}")

    @classmethod
    def from_values(cls, tag: int, values: Iterable[int | float]) -> CBORTypedArray:
        """Build a typed array from native Python numeric values."""
        spec = cls._specs.get(tag)
        if spec is None:
            raise ValueError(f"unknown RFC 8746 typed array tag {tag}")

        itemsize, byteorder, fmt = spec
        values = tuple(values)
        byteorder_prefix = "<" if byteorder == "little" else ">"
        data = struct.pack(f"{byteorder_prefix}{len(values)}{fmt}", *values)
        return cls(tag, data)

    @classmethod
    def from_array(cls, tag: int, values: array) -> CBORTypedArray:
        """Build a typed array from a standard library :class:`array.array`."""
        spec = cls._specs.get(tag)
        if spec is None:
            raise ValueError(f"unknown RFC 8746 typed array tag {tag}")

        itemsize, byteorder, fmt = spec
        candidates = cls._array_typecodes[(itemsize, fmt)]
        if values.typecode not in candidates or array(values.typecode).itemsize != itemsize:
            raise TypeError(
                f"array type {values.typecode!r} is not compatible with RFC 8746 tag {tag}"
            )

        wire_value = values.tobytes()
        if byteorder != sys.byteorder:
            wire_value = b"".join(
                wire_value[offset : offset + itemsize][::-1]
                for offset in range(0, len(wire_value), itemsize)
            )

        return cls(tag, wire_value)

    def __len__(self) -> int:
        return len(self.value) // self.itemsize

    def __getitem__(self, index: int | slice) -> int | float | list[int | float]:
        if isinstance(index, slice):
            return [self[position] for position in range(len(self))[index]]

        position = index + len(self) if index < 0 else index
        if not 0 <= position < len(self):
            raise IndexError("typed array index out of range")

        return self._struct.unpack_from(self.value, position * self.itemsize)[0]

    def __iter__(self) -> Iterator[int | float]:
        return (element[0] for element in self._struct.iter_unpack(self.value))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CBORTypedArray):
            return (self.tag, self.value) == (other.tag, other.value)

        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.tag, self.value))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.tag}, {self.value!r})"

    def to_array(self) -> array:
        """Return the elements as a standard library :class:`array.array`."""
        native_value = self.value
        if self.byteorder != sys.byteorder:
            native_value = b"".join(
                self.value[offset : offset + self.itemsize][::-1]
                for offset in range(0, len(self.value), self.itemsize)
            )

        result = array(self._typecode)
        result.frombytes(native_value)
        return result

    def as_array(self) -> array:
        """Alias for :meth:`to_array`."""
        return self.to_array()

    @property
    def data(self) -> bytes:
        return self.value

    @property
    def width(self) -> int:
        return self.itemsize

    @property
    def element_width(self) -> int:
        return self.itemsize

    @property
    def endianness(self) -> str:
        return self.byteorder

    @property
    def endian(self) -> str:
        return self.byteorder


TypedArray = CBORTypedArray


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
