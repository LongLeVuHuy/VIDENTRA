"""Exact time arithmetic. Never infer frame ordinal from FPS."""
from fractions import Fraction


def microseconds(value: Fraction) -> int:
    return round(value * 1_000_000)


def clip_time(pts: int, num: int, den: int, origin: Fraction) -> Fraction:
    if num <= 0 or den <= 0:
        raise ValueError("time_base must be positive")
    return Fraction(pts * num, den) - origin


def interval(start: int | float, end: int | float) -> None:
    if end <= start:
        raise ValueError("Expected a nonempty half-open interval")


def frame_endpoints(times: list[Fraction], start: Fraction, end: Fraction) -> tuple[int, int]:
    interval(start, end)
    if times != sorted(times):
        raise ValueError("Timeline must stay within one monotonic epoch")
    indices = [i for i, t in enumerate(times) if start <= t < end]
    if not indices:
        raise ValueError("No observed frame in interval")
    return indices[0], indices[-1]

