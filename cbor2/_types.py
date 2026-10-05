from __future__ import annotations

import math
import struct
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


_float_formats = {16: ">e", 32: ">f", 64: ">d"}


class FloatValue:
    """
    A float wrapper that preserves the CBOR wire representation of a
    floating-point value: its width (16, 32 or 64 bits) and its exact bit
    pattern (including NaN payloads and the sign of zero).

    Instances of this type are produced by :class:`.CBORDecoder` when the
    ``float_fidelity`` option is enabled, but they can also be constructed
    directly from a plain :class:`float` (the value must be exactly
    representable at the given width).

    When re-encoded, the original wire form (width and bit pattern) is
    reused by default, even when canonical encoding is requested: fidelity
    takes precedence over canonicalization for wrapped values, so NaN
    payloads are never normalized away. If :attr:`value` is numerically
    modified, the new value is re-validated against the stored width and
    re-encoded at that same width; encoding fails with
    :exc:`ValueError` if the new value cannot be expressed at that width.

    :param value: the numeric value (a plain :class:`float`)
    :param width: the float width in bits (16, 32 or 64; default 64)
    """

    __slots__ = ("_value", "_width", "_bits", "_modified")

    def __init__(self, value: float, width: int = 64) -> None:
        if width not in _float_formats:
            raise ValueError(f"invalid float width {width!r} (must be 16, 32 or 64)")

        self._width = width
        self._bits = self._pack(value, width)
        self._value = float(value)
        self._modified = False

    @staticmethod
    def _pack(value: float, width: int) -> bytes:
        fmt = _float_formats[width]
        try:
            packed = struct.pack(fmt, value)
        except OverflowError:
            raise ValueError(f"{value!r} is out of range for a {width}-bit float") from None

        if not math.isnan(value) and struct.unpack(fmt, packed)[0] != value:
            raise ValueError(f"{value!r} cannot be represented exactly as a {width}-bit float")

        return packed

    @classmethod
    def from_bits(cls, bits: bytes, width: int) -> FloatValue:
        """
        Construct a :class:`FloatValue` from a raw big-endian bit pattern.

        :param bits: the raw payload bytes (2, 4 or 8 bytes)
        :param width: the float width in bits (16, 32 or 64)
        """
        if width not in _float_formats:
            raise ValueError(f"invalid float width {width!r} (must be 16, 32 or 64)")
        elif len(bits) != width // 8:
            raise ValueError(
                f"a {width}-bit float requires {width // 8} bytes, got {len(bits)}"
            )

        self = cls.__new__(cls)
        self._width = width
        self._bits = bytes(bits)
        self._value = struct.unpack(_float_formats[width], self._bits)[0]
        self._modified = False
        return self

    @property
    def value(self) -> float:
        """The numeric value, as a plain :class:`float`."""
        return self._value

    @value.setter
    def value(self, new_value: float) -> None:
        self._value = float(new_value)
        self._modified = True

    @property
    def width(self) -> int:
        """The float width in bits (16, 32 or 64)."""
        return self._width

    @property
    def bits(self) -> bytes:
        """
        The raw big-endian bit pattern of the value at its stored width.

        If the value has not been modified since decoding/construction, this
        is the exact bit pattern read from the wire (NaN payloads and the
        sign of zero included). If the value was modified, the new value is
        packed at the stored width, raising :exc:`ValueError` if it cannot
        be expressed at that width.
        """
        if self._modified:
            return self._pack(self._value, self._width)

        return self._bits

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FloatValue):
            return (self._width, self._value) == (other._width, other._value)
        elif isinstance(other, (float, int)):
            return self._value == other

        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)

    def __float__(self) -> float:
        return self._value

    def __repr__(self) -> str:
        return f"FloatValue({self._value!r}, width={self._width}, bits={self.bits!r})"


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
