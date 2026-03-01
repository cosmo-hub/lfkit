"""Unit tests for `lfkit.corrections.responses` module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# Adjust import path to your package layout
from lfkit.corrections.responses import write_kcorrect_response


def test_write_kcorrect_response_rejects_non_1d_arrays(tmp_path: Path) -> None:
    """Tests that write_kcorrect_response raises ValueError for non-1D inputs."""
    wave = np.zeros((2, 5))
    thr = np.zeros(10)

    with pytest.raises(ValueError):
        write_kcorrect_response(
            name="kids_r",
            wave_angst=wave,
            throughput=thr,
            out_dir=tmp_path,
        )


def test_write_kcorrect_response_rejects_length_mismatch(tmp_path: Path) -> None:
    """Tests that write_kcorrect_response raises ValueError when array lengths differ."""
    wave = np.linspace(4000, 8000, 10)
    thr = np.linspace(0, 1, 11)

    with pytest.raises(ValueError):
        write_kcorrect_response(
            name="kids_r",
            wave_angst=wave,
            throughput=thr,
            out_dir=tmp_path,
        )


def test_write_kcorrect_response_rejects_too_few_points(tmp_path: Path) -> None:
    """Tests that write_kcorrect_response raises ValueError for curves with <10 samples."""
    wave = np.linspace(4000, 8000, 9)
    thr = np.linspace(0, 1, 9)

    with pytest.raises(ValueError):
        write_kcorrect_response(
            name="kids_r",
            wave_angst=wave,
            throughput=thr,
            out_dir=tmp_path,
        )


def test_write_kcorrect_response_writes_dat_file(tmp_path: Path) -> None:
    """Tests that write_kcorrect_response writes a .dat file to the output directory."""
    wave = np.linspace(4000, 8000, 10)
    thr = np.linspace(0, 1, 10)

    name = write_kcorrect_response(
        name="kids_r",
        wave_angst=wave,
        throughput=thr,
        out_dir=tmp_path,
        normalize=True,
    )

    assert name == "kids_r"
    out_path = tmp_path / "kids_r.dat"
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_write_kcorrect_response_sorts_wavelengths(tmp_path: Path) -> None:
    """Tests that write_kcorrect_response sorts the output by increasing wavelength."""
    wave = np.array([6000, 5000, 7000, 4000, 8000, 4500, 6500, 5500, 7500, 8500], float)
    thr = np.linspace(0, 1, wave.size)

    write_kcorrect_response(
        name="kids_r",
        wave_angst=wave,
        throughput=thr,
        out_dir=tmp_path,
        normalize=False,
    )

    arr = np.loadtxt(tmp_path / "kids_r.dat")
    w_out = arr[:, 0]
    assert np.all(w_out[1:] >= w_out[:-1])


def test_write_kcorrect_response_clips_negative_throughput(tmp_path: Path) -> None:
    """Tests that write_kcorrect_response clips negative throughput values to zero."""
    wave = np.linspace(4000, 8000, 10)
    thr = np.array([0.1, -0.2, 0.3, -1.0, 0.5, 0.6, -0.1, 0.8, 0.9, 1.0], float)

    write_kcorrect_response(
        name="kids_r",
        wave_angst=wave,
        throughput=thr,
        out_dir=tmp_path,
        normalize=False,
    )

    arr = np.loadtxt(tmp_path / "kids_r.dat")
    thr_out = arr[:, 1]
    assert np.all(thr_out >= 0.0)


def test_write_kcorrect_response_normalizes_to_unity_max(tmp_path: Path) -> None:
    """Tests that write_kcorrect_response normalizes throughput so max equals one when enabled."""
    wave = np.linspace(4000, 8000, 10)
    thr = np.linspace(0, 5, 10)

    write_kcorrect_response(
        name="kids_r",
        wave_angst=wave,
        throughput=thr,
        out_dir=tmp_path,
        normalize=True,
    )

    arr = np.loadtxt(tmp_path / "kids_r.dat")
    thr_out = arr[:, 1]
    assert np.isclose(float(np.max(thr_out)), 1.0)


def test_write_kcorrect_response_does_not_normalize_when_disabled(tmp_path: Path) -> None:
    """Tests that write_kcorrect_response leaves throughput scale unchanged when normalize is False."""
    wave = np.linspace(4000, 8000, 10)
    thr = np.linspace(0, 5, 10)

    write_kcorrect_response(
        name="kids_r",
        wave_angst=wave,
        throughput=thr,
        out_dir=tmp_path,
        normalize=False,
    )

    arr = np.loadtxt(tmp_path / "kids_r.dat")
    thr_out = arr[:, 1]
    assert np.isclose(float(np.max(thr_out)), float(np.max(thr)))


def test_write_kcorrect_response_creates_nested_out_dir(tmp_path: Path) -> None:
    """Tests that write_kcorrect_response creates parent directories for out_dir."""
    out_dir = tmp_path / "a" / "b" / "c"
    wave = np.linspace(4000, 8000, 10)
    thr = np.linspace(0, 1, 10)

    write_kcorrect_response(
        name="kids_r",
        wave_angst=wave,
        throughput=thr,
        out_dir=out_dir,
        normalize=True,
    )

    assert (out_dir / "kids_r.dat").exists()


def test_write_kcorrect_response_writes_two_columns(tmp_path: Path) -> None:
    """Tests that write_kcorrect_response writes exactly two numeric columns."""
    wave = np.linspace(4000, 8000, 10)
    thr = np.linspace(0, 1, 10)

    write_kcorrect_response(
        name="kids_r",
        wave_angst=wave,
        throughput=thr,
        out_dir=tmp_path,
        normalize=True,
    )

    arr = np.loadtxt(tmp_path / "kids_r.dat")
    assert arr.ndim == 2
    assert arr.shape[1] == 2
