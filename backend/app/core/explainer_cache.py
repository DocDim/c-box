import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import pandas as pd

from app.config import settings
from app.core.shap_engine import ShapExplainerEngine


class ExplainerCache:
    """
    LRU cache of (estimator, ShapExplainerEngine) per model. Reloading a model and
    rebuilding its explainer on every request dominated latency (NFR-3). The key
    includes the file's mtime and the background dataset id, so replacing either
    invalidates the entry automatically.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._entries: "OrderedDict[Tuple, ShapExplainerEngine]" = OrderedDict()
        self._estimators: "OrderedDict[Tuple, Any]" = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(model_id: int, file_path: str, background_id: Optional[int]) -> Tuple:
        try:
            mtime = Path(file_path).stat().st_mtime_ns
        except OSError:
            mtime = 0
        return (model_id, mtime, background_id)

    def get_estimator(self, model_id: int, file_path: str, load: Callable[[], Any]) -> Any:
        """Cached deserialized estimator, shared by /predict and explainer construction."""
        key = self._key(model_id, file_path, None)[:2]
        with self._lock:
            estimator = self._estimators.get(key)
            if estimator is not None:
                self._estimators.move_to_end(key)
                return estimator
        estimator = load()
        with self._lock:
            self._estimators[key] = estimator
            while len(self._estimators) > self.max_size:
                self._estimators.popitem(last=False)
        return estimator

    def get_or_create(
        self,
        model_id: int,
        file_path: str,
        background_id: Optional[int],
        build: Callable[[], Tuple[Any, str, Optional[pd.DataFrame]]],
    ) -> ShapExplainerEngine:
        """build() returns (estimator, task_type, background_frame); called on cache miss."""
        key = self._key(model_id, file_path, background_id)
        with self._lock:
            engine = self._entries.get(key)
            if engine is not None:
                self._entries.move_to_end(key)
                return engine

        estimator, task_type, background = build()
        engine = ShapExplainerEngine(
            estimator,
            task_type=task_type,
            background_data=background,
            background_max_rows=settings.SHAP_BACKGROUND_MAX_ROWS,
            kmeans_k=settings.SHAP_KMEANS_K,
        )
        with self._lock:
            self._entries[key] = engine
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)
        return engine

    def invalidate(self, model_id: int) -> None:
        with self._lock:
            for key in [k for k in self._entries if k[0] == model_id]:
                del self._entries[key]
            for key in [k for k in self._estimators if k[0] == model_id]:
                del self._estimators[key]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._estimators.clear()


explainer_cache = ExplainerCache(settings.EXPLAINER_CACHE_SIZE)
