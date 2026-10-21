"""Clock samples: round trip, offset, and which samples set a page load's offset."""

from backend.app.telemetry.clock import ClockSample, accepted, segment_offset


def _sample(rtt, offset):
    # A ping with the given round trip and clock difference, and no server delay.
    return ClockSample(t0=0.0, t1=offset + rtt / 2, t2=offset + rtt / 2, t3=rtt)


def test_offset_and_round_trip_of_an_even_ping():
    # Server clock 1,000,000 ms ahead, 10 ms each way, 2 ms of server work.
    sample = ClockSample(t0=100.0, t1=1_000_110.0, t2=1_000_112.0, t3=122.0)
    assert sample.rtt_ms == 20.0
    assert sample.offset_ms == 1_000_000.0


def test_a_slightly_negative_round_trip_counts_as_zero():
    assert ClockSample(t0=0.0, t1=1000.0, t2=1000.3, t3=0.2).rtt_ms == 0.0


def test_the_faster_half_is_accepted_in_the_original_order():
    samples = [_sample(40, 0), _sample(10, 0), _sample(30, 0), _sample(20, 0)]
    assert accepted(samples) == [False, True, False, True]


def test_an_odd_count_keeps_the_middle_sample():
    assert accepted([_sample(30, 0), _sample(10, 0), _sample(20, 0)]) == [False, True, True]


def test_slow_samples_cannot_move_the_offset():
    fast = [_sample(1, 100), _sample(2, 102), _sample(3, 104), _sample(4, 106)]
    slow = [_sample(50, 5_000), _sample(60, -5_000), _sample(70, 9_000), _sample(80, 0)]
    assert segment_offset(fast + slow) == 103.0


def test_a_single_sample_sets_the_offset():
    assert accepted([_sample(5, 42)]) == [True]
    assert segment_offset([_sample(5, 42)]) == 42.0
