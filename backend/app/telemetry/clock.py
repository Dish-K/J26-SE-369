from dataclasses import dataclass
from statistics import median

@dataclass(frozen=True)
class ClockSample:
    t0: float  # browser clock when the ping left
    t1: float  # server clock when the ping arrived
    t2: float  # server clock when the reply left
    t3: float  # browser clock when the reply arrived

    @property
    def rtt_ms(self) -> float:
        # Browser clock rounding can push a very fast round trip just below 0.
        return max((self.t3 - self.t0) - (self.t2 - self.t1), 0.0)

    @property
    def offset_ms(self) -> float:
        return ((self.t1 - self.t0) + (self.t2 - self.t3)) / 2


def accepted(samples: list[ClockSample]) -> list[bool]:
    # A short round trip leaves less room for an uneven delay, so keep the faster half.
    keep = (len(samples) + 1) // 2
    fastest = set(sorted(range(len(samples)), key=lambda i: samples[i].rtt_ms)[:keep])
    return [i in fastest for i in range(len(samples))]


def segment_offset(samples: list[ClockSample]) -> float:
    flags = accepted(samples)
    return median(sample.offset_ms for sample, keep in zip(samples, flags) if keep)
