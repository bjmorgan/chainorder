"""Timing benchmark for ``circulation_invariants``: single-k vs full-FFT.

Measures the per-call cost of the single-k ``circulation_invariants`` (48 cubic
point operations, each a ``_apply_cubic_op`` followed by a ``_project_arm``
projection) against a full-FFT reference, and splits the single-k cost into the
operation-apply phase and the projection phase.

Run as a single command from the repository root::

    python benchmarks/bench_circulation_invariants.py

It prints a table (N, single-k median, FFT median, speedup, apply, project,
apply%) and writes it to ``circulation_invariants_timing.csv`` next to this
file. It is a standalone script, not a pytest test, so it never gates CI on
timing. Two assertions guard the methodology: a configuration-independence check
(per-call time agrees across occupancy contents to within timer noise) and a
correctness cross-check (single-k equals the full-FFT reference).
"""
from __future__ import annotations

import csv
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from chainorder.decompose import SublatticeOccupation
from chainorder.order_params import (
    CUBIC_OPS,
    _apply_cubic_op,
    _arm_ramp,
    _project_arm,
    circulation_invariants,
)

N_VALUES = [3, 6, 9, 12, 15, 18]
PERIOD = 3
REPEATS = 25
CONFIG_CHECK_N = 12
CONFIG_TOLERANCE = 0.20  # config-independence spread must stay within timer noise
IDENTITY = ((0, 1, 2), (1, 1, 1))  # the identity operation; projection cost is content-independent

_W = np.exp(2j * np.pi / 3)


def _circulation_invariants_fft(occupation: SublatticeOccupation, *, period: int) -> tuple[float, float]:
    """Full-FFT reference for the timing comparison."""
    sub = occupation.occupation
    n = sub.shape[1]
    idx = (n // period,) * 3
    chirality = 0.0
    coherence = 0.0
    for operation in CUBIC_OPS:
        moved = _apply_cubic_op(sub, operation.perm, operation.signs)
        f = np.fft.fftn(moved.astype(float), axes=(1, 2, 3)) / n**3
        a, b, c = f[0][idx], f[1][idx], f[2][idx]
        e_plus = a + _W * b + _W * _W * c
        e_minus = a + _W * _W * b + _W * c
        chirality += operation.det * (abs(e_plus) ** 2 - abs(e_minus) ** 2)
        coherence += abs(e_plus) ** 2 + abs(e_minus) ** 2
    return chirality / len(CUBIC_OPS), coherence / len(CUBIC_OPS)


IMPLEMENTATIONS: list[tuple[str, Callable[..., object]]] = [
    ("single-k", circulation_invariants),
    ("fft", _circulation_invariants_fft),
]

CSV_PATH = Path(__file__).with_name("circulation_invariants_timing.csv")


def _gs_occupation(n: int) -> SublatticeOccupation:
    """A representative cubic occupation: the single-q <111> ground-state helix."""
    triad = np.indices((n, n, n)).sum(axis=0)
    occupation = np.zeros((3, n, n, n), dtype=np.int64)
    for sublattice in range(3):
        occupation[sublattice] = triad % PERIOD == sublattice % PERIOD
    return SublatticeOccupation(occupation=occupation)


def _random_occupation(n: int) -> SublatticeOccupation:
    """A disordered cubic occupation."""
    rng = np.random.default_rng(0)
    return SublatticeOccupation(occupation=rng.integers(0, 2, size=(3, n, n, n)))


def _zero_occupation(n: int) -> SublatticeOccupation:
    """An empty cubic occupation."""
    return SublatticeOccupation(occupation=np.zeros((3, n, n, n), dtype=np.int64))


def _time(call: Callable[[], object], repeats: int) -> tuple[float, float]:
    """Median and minimum wall-clock time of ``call`` in milliseconds.

    One warm-up call is made and discarded before the timed repeats.
    """
    call()  # warm up, discarded
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1e3, min(samples) * 1e3


def _breakdown(occupation: np.ndarray, n: int, repeats: int) -> tuple[float, float]:
    """Median ms for the 48-operation apply phase and the 48-call projection phase."""
    def apply_phase() -> None:
        for operation in CUBIC_OPS:
            _apply_cubic_op(occupation, operation.perm, operation.signs)

    ramp = _arm_ramp(n, PERIOD)
    moved = _apply_cubic_op(occupation, *IDENTITY)

    def project_phase() -> None:
        for _ in CUBIC_OPS:
            _project_arm(moved, ramp, n)

    apply_ms, _ = _time(apply_phase, repeats)
    project_ms, _ = _time(project_phase, repeats)
    return apply_ms, project_ms


def _config_independence(n: int, repeats: int) -> tuple[dict[str, float], float]:
    """Per-call median (ms) on three occupancy contents and their relative spread."""
    contents = {
        "gs": _gs_occupation(n),
        "random": _random_occupation(n),
        "zeros": _zero_occupation(n),
    }
    medians = {
        name: _time(lambda bound=occ: circulation_invariants(bound, period=PERIOD), repeats)[0]
        for name, occ in contents.items()
    }
    values = list(medians.values())
    spread = (max(values) - min(values)) / statistics.median(values)
    return medians, spread


def main() -> None:
    # Configuration independence: per-call cost is fixed per N because neither the
    # array applies nor the FFTs branch on content, so a synthetic fixture times
    # like production. Validate at one N before trusting the sweep.
    medians, spread = _config_independence(CONFIG_CHECK_N, REPEATS)
    print(f"Configuration independence at N={CONFIG_CHECK_N} (period={PERIOD}):")
    for name, ms in medians.items():
        print(f"  {name:>6}: {ms:8.3f} ms")
    print(f"  spread: {spread * 100:.1f}% (tolerance {CONFIG_TOLERANCE * 100:.0f}%)")
    assert spread < CONFIG_TOLERANCE, (
        f"per-call time varies by {spread * 100:.1f}% across occupancy contents "
        f"at N={CONFIG_CHECK_N}; expected < {CONFIG_TOLERANCE * 100:.0f}% for a "
        f"content-independent cost. Re-run on a quiet machine."
    )

    # Correctness cross-check: single-k equals the FFT reference (one input).
    check = _gs_occupation(CONFIG_CHECK_N)
    single_k = circulation_invariants(check, period=PERIOD)
    reference = _circulation_invariants_fft(check, period=PERIOD)
    assert (
        abs(single_k.chirality - reference[0]) < 1e-10
        and abs(single_k.coherence - reference[1]) < 1e-10
    ), "single-k disagrees with the FFT reference; the optimisation is wrong."

    # Per-call cost of each implementation, their speedup, and the single-k split.
    rows = []
    for n in N_VALUES:
        occ = _gs_occupation(n)
        timings = {
            name: _time(lambda f=fn, bound=occ: f(bound, period=PERIOD), REPEATS)
            for name, fn in IMPLEMENTATIONS
        }
        singlek_ms = timings["single-k"][0]
        fft_ms = timings["fft"][0]
        speedup = fft_ms / singlek_ms
        apply_ms, project_ms = _breakdown(occ.occupation, n, REPEATS)
        apply_pct = 100.0 * apply_ms / (apply_ms + project_ms)
        rows.append((n, singlek_ms, fft_ms, speedup, apply_ms, project_ms, apply_pct))

    print(f"\nSingle-k vs FFT (period={PERIOD}, repeats={REPEATS}):")
    print(
        f"{'N':>3}  {'singlek_ms':>10}  {'fft_ms':>9}  {'speedup':>7}  "
        f"{'apply_ms':>9}  {'project_ms':>10}  {'apply%':>7}"
    )
    for n, singlek_ms, fft_ms, speedup, apply_ms, project_ms, apply_pct in rows:
        print(
            f"{n:>3}  {singlek_ms:>10.3f}  {fft_ms:>9.3f}  {speedup:>6.2f}x  "
            f"{apply_ms:>9.3f}  {project_ms:>10.3f}  {apply_pct:>6.1f}%"
        )

    header = ("N", "singlek_ms", "fft_ms", "speedup", "apply_ms", "project_ms", "apply_pct")
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(
            (n, f"{sk:.6f}", f"{ff:.6f}", f"{sp:.4f}", f"{ap:.6f}", f"{pr:.6f}", f"{pct:.3f}")
            for n, sk, ff, sp, ap, pr, pct in rows
        )
    print(f"\nWrote {CSV_PATH.name} ({len(rows)} rows).")


if __name__ == "__main__":
    main()
