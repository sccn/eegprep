"""Seeded draws for the nightly fuzzer.

The generator is SHA-256 in counter mode with rejection sampling. It does not
read Hypothesis, ``random``, NumPy, or the set of imported modules, so a case
list depends only on the seed, the target name, and the example index.
"""

from __future__ import annotations

import hashlib


def mix_seed(seed_value: int, function: str, index: int) -> int:
    """Return the 128-bit seed for one example of one target."""
    payload = f"eegprep-fuzz-v1\0{int(seed_value)}\0{function}\0{int(index)}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "little")


class Draw:
    """Uniform integers from an explicit inclusive range.

    A forced value list replays or shrinks a recorded trace. Values past the
    end of that list use the low end of the requested range. A value outside
    the range is clamped into it.
    """

    def __init__(self, seed: int, forced: tuple[int, ...] | None = None):
        self._seed = int(seed) & ((1 << 128) - 1)
        self._counter = 0
        self._pool = 0
        self._pool_bits = 0
        self._forced = () if forced is None else tuple(int(value) for value in forced)
        self._forced_index = 0
        self.trace: list[tuple[int, int, int]] = []

    def integers(self, low: int, high: int) -> int:
        low = int(low)
        high = int(high)
        if high < low:
            raise ValueError(f"integer range {low}..{high} is empty")
        if self._forced_index < len(self._forced):
            raw = self._forced[self._forced_index]
            self._forced_index += 1
            value = min(high, max(low, raw))
        elif self._forced:
            value = low
        else:
            value = low + self._uniform(high - low + 1)
        self.trace.append((value, low, high))
        return value

    def choice(self, options):
        picked = self.integers(0, len(options) - 1)
        return options[picked]

    def _uniform(self, span: int) -> int:
        if span <= 1:
            return 0
        limit = (1 << 64) // span * span
        while True:
            draw = self._u64()
            if draw < limit:
                return draw % span

    def _u64(self) -> int:
        if self._pool_bits < 64:
            block = hashlib.sha256(self._seed.to_bytes(16, "little") + self._counter.to_bytes(8, "little")).digest()
            self._counter += 1
            self._pool = int.from_bytes(block, "little")
            self._pool_bits = 256
        value = self._pool & ((1 << 64) - 1)
        self._pool >>= 64
        self._pool_bits -= 64
        return value
