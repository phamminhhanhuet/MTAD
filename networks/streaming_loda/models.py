from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class _HistogramModel:
    projection: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    edges: np.ndarray
    probabilities: np.ndarray
    generation: int


class StreamingLODA:
    """Streaming variant of LODA for batch-wise online anomaly detection.

    Parameters
    ----------
    n_features : int
        Number of input dimensions.
    n_histograms : int, default=100
        Maximum number of active histograms (``k_max`` in the paper).
    histograms_per_batch : int, default=1
        Number of new histograms learned per incoming batch (``k_b``).
    memory : int, default=5
        Number of most recent batches used to train every new histogram.
        This matches the semantics of the authors' released implementation.
    score_decay : float, default=1.0
        Optional age weighting. ``1.0`` gives the unweighted score in Eq. (2).
        A value such as ``0.95`` gives newer histogram groups more weight as
        discussed around Eq. (3).
    padding : float, default=0.1
        Fraction of the projected training range reserved as padding on both
        sides before the two overflow bins.
    epsilon : float, default=1e-10
        Pseudocount used to keep log probabilities finite.
    standardize_window : bool, default=True
        Standardize the stacked memory batches before each new projection.
        The public StreamingLODA implementation stores this mean/std per
        histogram; keeping it here also makes old histograms represent the
        distribution that existed when they were created.
    random_state : int or None, default=None
        Seed for reproducible random sparse projections.
    """

    def __init__(
        self,
        n_features,
        n_histograms=100,
        histograms_per_batch=1,
        memory=5,
        score_decay=1.0,
        padding=0.1,
        epsilon=1e-10,
        standardize_window=True,
        random_state=None,
    ):
        self.n_features = int(n_features)
        self.n_histograms = int(n_histograms)
        self.histograms_per_batch = int(histograms_per_batch)
        self.memory = int(memory)
        self.score_decay = float(score_decay)
        self.padding = float(padding)
        self.epsilon = float(epsilon)
        self.standardize_window = bool(standardize_window)
        self.random_state = random_state

        if self.n_features <= 0:
            raise ValueError("n_features phải > 0")
        if self.n_histograms <= 0:
            raise ValueError("n_histograms phải > 0")
        if not 1 <= self.histograms_per_batch <= self.n_histograms:
            raise ValueError("histograms_per_batch phải nằm trong [1, n_histograms]")
        if self.memory <= 0:
            raise ValueError("memory phải > 0")
        if not 0.0 < self.score_decay <= 1.0:
            raise ValueError("score_decay phải nằm trong (0, 1]")
        if self.padding < 0.0:
            raise ValueError("padding phải >= 0")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon phải > 0")

        self._rng = np.random.RandomState(random_state)
        self._recent_batches = deque(maxlen=self.memory)
        self._histograms = deque(maxlen=self.n_histograms)
        self._generation = 0

    @property
    def n_active_histograms(self):
        return len(self._histograms)

    @property
    def is_fitted(self):
        return bool(self._histograms)

    def reset(self):
        """Reset learned state while preserving hyperparameters and RNG seed."""
        self._rng = np.random.RandomState(self.random_state)
        self._recent_batches.clear()
        self._histograms.clear()
        self._generation = 0
        return self

    def _validate_X(self, X, allow_empty=False):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X phải có shape [N, D], nhận {X.shape}")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Sai số chiều feature: model={self.n_features}, X={X.shape[1]}"
            )
        if not allow_empty and X.shape[0] == 0:
            raise ValueError("X không được rỗng")
        if not np.isfinite(X).all():
            raise ValueError("X chứa NaN/Inf")
        return X

    def _new_projection(self):
        # Original LODA keeps approximately sqrt(d) non-zero components.
        d = self.n_features
        n_nonzero = max(1, int(math.sqrt(d)))
        projection = self._rng.normal(0.0, 1.0, size=d)
        if n_nonzero < d:
            zero_idx = self._rng.permutation(d)[: d - n_nonzero]
            projection[zero_idx] = 0.0
        return projection

    @staticmethod
    def _rice_bins(n_samples):
        # Paper: M = 2 * ceil(cuberoot(n)). Two overflow bins are added later.
        return max(2, 2 * int(math.ceil(float(n_samples) ** (1.0 / 3.0))))

    def _fit_histogram(self, stacked_batch, generation):
        if self.standardize_window:
            mean = np.mean(stacked_batch, axis=0)
            scale = np.std(stacked_batch, axis=0)
            scale = np.where(scale > 1e-12, scale, 1.0)
            transformed = (stacked_batch - mean) / scale
        else:
            mean = np.zeros(self.n_features, dtype=np.float64)
            scale = np.ones(self.n_features, dtype=np.float64)
            transformed = stacked_batch

        projection = self._new_projection()
        projected = transformed @ projection

        n_bins = self._rice_bins(len(projected))
        observed_min = float(np.min(projected))
        observed_max = float(np.max(projected))
        observed_range = observed_max - observed_min

        # The paper asks for ample padding around the learned range so normal
        # boundary variation does not immediately spill into overflow bins.
        if observed_range <= 1e-12:
            base_span = max(abs(observed_min), 1.0)
            pad = max(self.padding * base_span, 1e-6)
        else:
            pad = self.padding * observed_range

        lower = observed_min - pad
        upper = observed_max + pad
        if not upper > lower:
            upper = lower + 1e-6

        # ``edges`` define only the M normal bins. searchsorted naturally maps
        # values below/above them to index 0 / M+1, the two overflow bins.
        edges = np.linspace(lower, upper, n_bins + 1, dtype=np.float64)
        bin_idx = np.searchsorted(edges, projected, side="right")
        bin_idx = np.clip(bin_idx, 0, n_bins + 1)

        counts = np.bincount(bin_idx, minlength=n_bins + 2).astype(np.float64)
        # Overflow counts should be zero for training samples due to padding.
        # Pseudocounts keep their log-probability finite but very small.
        counts += self.epsilon
        probabilities = counts / np.sum(counts)

        return _HistogramModel(
            projection=projection,
            mean=mean,
            scale=scale,
            edges=edges,
            probabilities=probabilities,
            generation=generation,
        )

    def partial_fit(self, X):
        """Update Streaming LODA with one incoming batch."""
        X = self._validate_X(X)
        self._recent_batches.append(X.copy())
        stacked = np.vstack(self._recent_batches)

        generation = self._generation
        for _ in range(self.histograms_per_batch):
            self._histograms.append(self._fit_histogram(stacked, generation))
        self._generation += 1
        return self

    def decision_function(self, X):
        """Return anomaly scores without modifying model state."""
        X = self._validate_X(X, allow_empty=True)
        if not self.is_fitted:
            raise RuntimeError("StreamingLODA chưa được fit")
        if X.shape[0] == 0:
            return np.empty(0, dtype=np.float64)

        newest_generation = self._generation - 1
        total = np.zeros(X.shape[0], dtype=np.float64)

        # Eq. (2): average negative log histogram probability. Optional age
        # weights implement the newer-histogram preference discussed in Eq. (3).
        for hist in self._histograms:
            transformed = (X - hist.mean) / hist.scale
            projected = transformed @ hist.projection
            n_bins = len(hist.probabilities) - 2
            bin_idx = np.searchsorted(hist.edges, projected, side="right")
            bin_idx = np.clip(bin_idx, 0, n_bins + 1)
            age = newest_generation - hist.generation
            weight = self.score_decay ** max(age, 0)
            total -= weight * np.log(hist.probabilities[bin_idx])

        # During warm-up use the active count; once full this is exactly k_max,
        # matching the denominator in the paper's Eq. (2)/(3).
        return total / float(len(self._histograms))

    def fit_stream(self, X, batch_size):
        """Reset model, learn a stream, and return one score per train sample.

        The first batch must initialize the model, so it is scored immediately
        after its first update. Every later batch is evaluated before update,
        matching the prequential behavior used by the authors.
        """
        X = self._validate_X(X)
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size phải > 0")

        self.reset()
        output = np.empty(X.shape[0], dtype=np.float64)
        first = True
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch = X[start:end]
            if first:
                self.partial_fit(batch)
                output[start:end] = self.decision_function(batch)
                first = False
            else:
                output[start:end] = self.decision_function(batch)
                self.partial_fit(batch)
        return output

    def score_stream(self, X, batch_size, update=True):
        """Score a stream batch-by-batch, optionally updating after each batch.

        Crucially, every batch is scored *before* it is used for ``partial_fit``;
        therefore test samples never influence their own anomaly scores.
        """
        X = self._validate_X(X, allow_empty=True)
        if not self.is_fitted:
            raise RuntimeError("StreamingLODA chưa được fit")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size phải > 0")

        output = np.empty(X.shape[0], dtype=np.float64)
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch = X[start:end]
            output[start:end] = self.decision_function(batch)
            if update:
                self.partial_fit(batch)
        return output
