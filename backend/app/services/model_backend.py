"""Abstract model backend interface.

Concrete implementations load the actual ML framework and translate a
preprocessed tensor into class probabilities and an attention heatmap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ModelBackend(ABC):
    name: str
    version: str
    architecture: str
    input_size: int
    classes: list[str]

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory (idempotent)."""

    @abstractmethod
    def predict(self, preprocessed: np.ndarray) -> np.ndarray:
        """Return probability vector over ``classes`` for a (1,H,W,3) tensor."""

    @abstractmethod
    def grad_cam(self, preprocessed: np.ndarray, class_index: int) -> np.ndarray:
        """Return a 2D [0,1] heatmap aligned with the model input size."""

    def must_reload(self) -> bool:
        return False

    def close(self) -> None:
        pass