"""Average-reference numerical helper."""

from __future__ import annotations

from typing import Any

import numpy as np


def averef(
    data: Any,
    weights: Any | None = None,
    sphere: Any | None = None,
    *,
    return_parameters: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Average-reference channel-major data and optionally transform ICA weights.

    Set ``return_parameters=True`` to also receive transformed weights, sphere,
    and the removed channel mean. This explicit switch replaces MATLAB's
    output-count-dependent return behavior.
    """
    values = np.asarray(data)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("data must contain at least two channels")
    mean_data = np.mean(values, axis=0)
    referenced = values - mean_data

    output_weights: np.ndarray | None = None
    output_sphere: np.ndarray | None = None
    if weights is not None:
        weight_array = np.asarray(weights)
        unmixing = weight_array if sphere is None else weight_array @ np.asarray(sphere)
        inverse = np.linalg.pinv(unmixing)
        average_matrix = np.eye(inverse.shape[0]) - np.ones((inverse.shape[0], inverse.shape[0])) / inverse.shape[0]
        output_weights = np.linalg.pinv(average_matrix @ inverse)
        output_sphere = None if sphere is None else np.eye(values.shape[0])

    if return_parameters:
        return referenced, output_weights, output_sphere, mean_data
    return referenced


__all__ = ["averef"]
