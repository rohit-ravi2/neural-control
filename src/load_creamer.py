"""Loader for the Creamer/Leifer 2025 processed whole-brain dataset (OSF qxhjd).

Each recording directory contains:
  neural_data.txt        T x N comma-separated calcium traces (NaN where untracked)
  nan_mask.txt           T x N mask, same shape
  cell_ids.txt           N lines, neuron name or "" if unidentified
  stim_cell_indicies.txt K lines, column index of the stimulated neuron
                         (negative = target not resolved to a tracked column)
  stim_frame_indicies.txt K lines, frame index of each stimulation event
"""
from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "work/creamer_lds/2025_creamer_lds"


@dataclasses.dataclass
class Recording:
    name: str
    traces: np.ndarray       # (T, N) float, NaN preserved
    nan_mask: np.ndarray     # (T, N) bool, True where the loaded value is NaN
    cell_ids: list[str]      # length N, "" for unidentified
    stim_cells: np.ndarray   # (K,) int column indices, negative = unresolved
    stim_frames: np.ndarray  # (K,) int frame indices

    @property
    def n_frames(self) -> int:
        return self.traces.shape[0]

    @property
    def n_neurons(self) -> int:
        return self.traces.shape[1]

    @property
    def named_mask(self) -> np.ndarray:
        """(N,) bool — columns carrying a neuron identity."""
        return np.array([bool(c) for c in self.cell_ids])

    @property
    def resolved_stims(self) -> np.ndarray:
        """(K,) bool — stimulation events whose target column is known."""
        return self.stim_cells >= 0

    def stim_targets(self) -> list[str]:
        """Neuron name per stimulation event; "" unidentified, None unresolved."""
        out = []
        for c in self.stim_cells:
            out.append(self.cell_ids[c] if c >= 0 else None)
        return out


def load_recording(path: pathlib.Path) -> Recording:
    traces = np.loadtxt(path / "neural_data.txt", delimiter=",")
    nan_mask = np.loadtxt(path / "nan_mask.txt", delimiter=",").astype(bool)
    # cell_ids.txt quotes unidentified entries as "" — strip quotes, keep position.
    cell_ids = [
        line.strip().strip('"')
        for line in (path / "cell_ids.txt").read_text().splitlines()
    ]
    stim_cells = np.loadtxt(path / "stim_cell_indicies.txt", dtype=int, ndmin=1)
    stim_frames = np.loadtxt(path / "stim_frame_indicies.txt", dtype=int, ndmin=1)

    if traces.shape != nan_mask.shape:
        raise ValueError(f"{path.name}: traces {traces.shape} != mask {nan_mask.shape}")
    if len(cell_ids) != traces.shape[1]:
        raise ValueError(
            f"{path.name}: {len(cell_ids)} cell_ids != {traces.shape[1]} columns"
        )
    if stim_cells.shape != stim_frames.shape:
        raise ValueError(
            f"{path.name}: {stim_cells.shape} stim cells != {stim_frames.shape} frames"
        )

    return Recording(
        name=path.name,
        traces=traces,
        nan_mask=nan_mask,
        cell_ids=cell_ids,
        stim_cells=stim_cells,
        stim_frames=stim_frames,
    )


def recording_paths(root: pathlib.Path = DATA_ROOT) -> list[pathlib.Path]:
    return sorted(p for p in root.iterdir() if p.is_dir())


def load_all(root: pathlib.Path = DATA_ROOT) -> list[Recording]:
    return [load_recording(p) for p in recording_paths(root)]
