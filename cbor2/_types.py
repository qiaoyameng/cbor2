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


class CBORFloat(float):
    """
    A float that preserves the exact CBOR floating point representation it was
    decoded from (or was explicitly constructed with).

    The wrapper behaves numerically like an ordinary :class:`float` (so value
    comparisons, arithmetic and ``math`` functions work unchanged) but carries
    two extra pieces of information needed for bit-exact binary audits:

    * :attr:`width` -- the original float width (``16``, ``32`` or ``64``)
    * :attr:`bits` -- the raw, unsigned big-endian bit pattern of the original
      encoding (including the sign bit and any NaN payload)

    Re-encoding a ``CBORFloat`` reuses its original width and bit string, and
    re-checks that the numeric value is still represented by those bits.
    Constructing a ``CBORFloat`` validates that the value can actually be
    expressed in the given width; a value that would overflow or change when
    rounded raises :class:`ValueError`.

    Use :meth:`from_bits` to construct a wrapper from a raw bit pattern, which
    is the only way to preserve non-canonical NaN payloads and signed zeroes.

    :param value: the numeric value
    :param int width: the target float width (16, 32 or 64)
    :raises ValueError: if the value cannot be represented exactly in *width*
    """

    __slots__ = ("width", "bits")

    def __new__(cls, value: Any, width: int = 64) -> CBORFloat:
        if width not in (16, 32, 64):
            raise ValueError("float width must be one of 16, 32 or 64")

        number = float(value)
        float_format = {16: ">e", 32: ">f", 64: ">d"}[width]
        bits_format = {16: ">H", 32: ">I", 64: ">Q"}[width]
        try:
            packed = struct.pack(float_format, number)
        except OverflowError:
            raise ValueError(
                f"value {number!r} cannot be represented as a {width}-bit float"
            ) from None

        roundtrip = struct.unpack(float_format, packed)[0]
        if roundtrip != number and not (
            math.isnan(roundtrip) and math.isnan(number)
        ):
            raise ValueError(
                f"value {number!r} cannot be represented exactly as a "
                f"{width}-bit float"
            )

        bits = struct.unpack(bits_format, packed)[0]
        self = super().__new__(cls, number)
        self.width = width
        self.bits = bits
        return self

    @classmethod
    def from_bits(cls, bits: int, width: int) -> CBORFloat:
        """
        Create a ``CBORFloat`` from the raw unsigned bit pattern *bits* as
        originally encoded in a *width*-bit (16, 32 or 64) CBOR float.

        This preserves information that :class:`float` itself cannot carry,
        such as the original width, signed zero and arbitrary NaN payloads.

        :param int bits: the raw, big-endian unsigned bit pattern
        :param int width: the float width (16, 32 or 64)
        :raises ValueError: if *width* is invalid or *bits* is out of range
        """
        if width not in (16, 32, 64):
            raise ValueError("float width must be one of 16, 32 or 64")
        if not isinstance(bits, int) or bits < 0 or bits >= 1 << width:
            raise ValueError(f"bits 0x{bits:x} do not fit in a {width}-bit float")

        bits_format = {16: ">H", 32: ">I", 64: ">Q"}[width]
        float_format = {16: ">e", 32: ">f", 64: ">d"}[width]
        packed = struct.pack(bits_format, bits)
        value = struct.unpack(float_format, packed)[0]
        self = super().__new__(cls, value)
        self.width = width
        self.bits = bits
        return self

    def __repr__(self) -> str:
        return f"CBORFloat({float(self)!r}, width={self.width}, bits=0x{self.bits:x})"


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
