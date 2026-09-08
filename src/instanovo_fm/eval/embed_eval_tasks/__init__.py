"""
Evaluation task plug-in interface with auto-discovery.

This module provides a base class and metaclass for automatically registering
evaluation tasks. Any class that inherits from BaseTask will be automatically
registered in the TASK_REGISTRY.
"""

from typing import Any, Dict, List, Tuple, Type
import numpy as np
try:
    import faiss
except ImportError:
    faiss = None

# Global registry for evaluation tasks
TASK_REGISTRY: Dict[str, Type['BaseTask']] = {}


class SpectrumEvalTask(type):
    """
    Metaclass for automatically registering evaluation tasks.

    Any class that uses this metaclass will be automatically added to
    the TASK_REGISTRY when the module is imported.
    """

    def __init__(cls, name: str, bases: tuple, attrs: dict):
        """Register the class in the task registry if it's not the base class."""
        if name != 'BaseTask':
            # Register with lowercase class name for easy lookup
            TASK_REGISTRY[cls.__name__.lower()] = cls
            # Also register with the task's .name attribute (if it's a valid identifier)
            if hasattr(cls, 'name') and isinstance(cls.name, str):
                task_name_key = cls.name.lower().replace(' ', '_').replace('-', '_')
                TASK_REGISTRY[task_name_key] = cls
        super().__init__(name, bases, attrs)


class BaseTask(metaclass=SpectrumEvalTask):
    """
    Base class for all evaluation tasks.

    All evaluation tasks should inherit from this class. They will be
    automatically registered in the TASK_REGISTRY when their module is imported.

    Attributes:
        name: Human-readable name for the task
        description: Description of what the task does
        requires_metadata: Whether the task requires metadata
        requires_faiss: Whether the task requires FAISS index
        requires_model: Whether the task requires direct model access (for attention/gradient analysis)
    """

    name: str = "Base Task"
    description: str = "Base evaluation task"
    requires_metadata: bool = True
    requires_faiss: bool = False
    requires_model: bool = False
    requires_multi_split: bool = False

    def __init__(self, **kwargs):
        """Initialize the task with optional configuration."""
        self.config = kwargs

    def run(self, emb: np.ndarray, meta: Dict[str, np.ndarray], faiss_index: Any,
            model: Any = None, dataloader: Any = None, config: Any = None, device: Any = None) -> Dict[str, Any]:
        """
        Run the evaluation task.

        Args:
            emb: Embeddings array of shape (N, D)
            meta: Metadata dictionary with keys mapping to arrays of length N
            faiss_index: FAISS index for similarity search (if available)
            model: Model instance (only provided if requires_model=True)
            dataloader: DataLoader instance (only provided if requires_model=True)
            config: Configuration dict (only provided if requires_model=True)
            device: Device to run on (only provided if requires_model=True)

        Returns:
            Dictionary containing evaluation results
        """
        raise NotImplementedError("Subclasses must implement run()")

    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract metrics suitable for logging to TensorBoard/Neptune from task results.

        This method should be overridden by subclasses to define which metrics
        from their results should be logged. The returned dictionary should have
        flat string keys (without prefixes) and scalar float values.

        Args:
            task_results: Results dictionary returned by run()

        Returns:
            Dictionary mapping metric names to scalar values for logging.
            Keys should be simple metric names (e.g., "recall@5", "accuracy").
            The evaluator/trainer will add task-specific prefixes.

        Example:
            {
                "recall@1": 0.85,
                "recall@5": 0.92,
                "map": 0.78,
                "accuracy": 0.91
            }
        """
        # Default implementation: return empty dict (no metrics to log)
        # Subclasses should override this to extract their specific metrics
        return {}

    def validate_inputs(self, emb: np.ndarray, meta: Dict[str, np.ndarray], faiss_index: Any) -> None:
        """
        Validate inputs before running the task.

        Args:
            emb: Embeddings array
            meta: Metadata dictionary
            faiss_index: FAISS index

        Raises:
            ValueError: If inputs are invalid
        """
        if emb is None or len(emb.shape) != 2:
            raise ValueError("emb must be a 2D numpy array")

        if self.requires_metadata and (meta is None or len(meta) == 0):
            raise ValueError(f"Task {self.name} requires metadata but none provided")

        if self.requires_faiss and faiss_index is None:
            raise ValueError(f"Task {self.name} requires FAISS index but none provided")

        # Note: Model access validation is handled by the evaluator, not here
        # since model/dataloader/config/device are passed separately

        # Check that metadata arrays have the same length as embeddings
        # Skip non-array values (dicts, scalars, strings) which are aggregate metadata
        if meta:
            expected_length = len(emb)
            for key, value in meta.items():
                if isinstance(value, (dict, str, int, float, bool)):
                    continue
                if hasattr(value, '__len__') and len(value) != expected_length:
                    raise ValueError(f"Metadata '{key}' has length {len(value)} but expected {expected_length}")

    @staticmethod
    def apply_conditional_filter(
        emb: np.ndarray,
        meta: Dict[str, np.ndarray],
        subset_config: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
        """Filter embeddings and metadata by instrument conditions.

        Applies cumulative AND filters on frag_type, search_detector, and
        search_instrument using ``np.isin``.  Same pattern as
        PeakTypeClassificationTask's fragment-UMAP filtering.

        Args:
            emb: Embeddings array of shape (N, D).
            meta: Metadata dict with arrays of length N.
            subset_config: Dict with optional keys ``frag_types``,
                ``detectors``, ``instruments`` (each a list of allowed values
                or None to skip).

        Returns:
            (filtered_emb, filtered_meta, filter_description)
        """
        mask = np.ones(len(emb), dtype=bool)
        filter_parts: List[str] = []

        frag_types = subset_config.get("frag_types")
        if frag_types and "frag_type" in meta:
            mask &= np.isin(meta["frag_type"], frag_types)
            filter_parts.append(f"frag_type={frag_types}")

        detectors = subset_config.get("detectors")
        if detectors and "search_detector" in meta:
            mask &= np.isin(meta["search_detector"], detectors)
            filter_parts.append(f"detector={detectors}")

        instruments = subset_config.get("instruments")
        if instruments and "search_instrument" in meta:
            mask &= np.isin(meta["search_instrument"], instruments)
            filter_parts.append(f"instrument={instruments}")

        emb_filtered = emb[mask]

        meta_filtered: Dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, np.ndarray) and v.shape[0] == len(emb):
                meta_filtered[k] = v[mask]
            elif isinstance(v, list) and len(v) == len(emb):
                meta_filtered[k] = [v[i] for i in range(len(v)) if mask[i]]
            else:
                meta_filtered[k] = v

        desc = ", ".join(filter_parts) if filter_parts else "no filter"
        return emb_filtered, meta_filtered, desc


def get_task(task_name: str) -> Type[BaseTask]:
    """
    Get a task class by name.

    Args:
        task_name: Name of the task (case-insensitive)

    Returns:
        Task class

    Raises:
        KeyError: If task not found
    """
    task_name_lower = task_name.lower()
    if task_name_lower not in TASK_REGISTRY:
        available_tasks = list(TASK_REGISTRY.keys())
        raise KeyError(f"Task '{task_name}' not found. Available tasks: {available_tasks}")
    return TASK_REGISTRY[task_name_lower]


def list_tasks() -> Dict[str, Type[BaseTask]]:
    """
    List all available tasks.

    Returns:
        Dictionary mapping task names to task classes
    """
    return TASK_REGISTRY.copy()


def run_task(task_name: str, emb: np.ndarray, meta: Dict[str, np.ndarray],
             faiss_index: Any = None, **task_kwargs) -> Dict[str, Any]:
    """
    Run a specific task by name.

    Args:
        task_name: Name of the task to run
        emb: Embeddings array
        meta: Metadata dictionary
        faiss_index: FAISS index (optional)
        **task_kwargs: Additional arguments to pass to task constructor

    Returns:
        Task results
    """
    task_class = get_task(task_name)
    task = task_class(**task_kwargs)
    return task.run(emb, meta, faiss_index)


# Import all task modules to register them
# This will automatically discover and register any tasks defined in .py files
# in this directory (except __init__.py)
import importlib
from pathlib import Path

def _discover_tasks():
    """Discover and import all task modules in this directory."""
    current_dir = Path(__file__).parent
    for file_path in current_dir.glob("*.py"):
        if file_path.name != "__init__.py":
            module_name = file_path.stem
            try:
                importlib.import_module(f"instanovo_fm.eval.embed_eval_tasks.{module_name}")
            except ImportError as e:
                # Log but don't fail - some tasks might have optional dependencies
                print(f"Warning: Could not import task module {module_name}: {e}")

# Discover tasks when this module is imported
_discover_tasks()

# Explicit imports for direct access
try:
    from .embedding_statistics import EmbeddingStatisticsTask
    from .duplicate_retrieval import DuplicateRetrievalTask
    from .linear_probe import LinearProbeTask
    from .cosine_hyperscore_correlation import CosineHyperscoreCorrelationTask
    from .umap_visualisation import UMAPVisualisationTask
    from .confidence_signal_analysis import ConfidenceSignalAnalysisTask
    from .head_analysis import HeadAnalysisTask
    from .ig_attribution import IGAttributionTask
    from .peak_type_classification import PeakTypeClassificationTask
    from .esm2_cross_modal_alignment import ESM2CrossModalAlignmentTask
    from .evoc_clustering import EVoCClusteringTask
    from .glass_box_attribution import GlassBoxAttributionTask
except ImportError as e:
    # Some tasks might have optional dependencies
    print(f"Warning: Could not import some task classes: {e}")

# Export all task classes
__all__ = [
    'BaseTask',
    'TASK_REGISTRY',
    'get_task',
    'list_tasks',
    'run_task',
    'EmbeddingStatisticsTask',
    'DuplicateRetrievalTask',
    'LinearProbeTask',
    'CosineHyperscoreCorrelationTask',
    'UMAPVisualisationTask',
    'ConfidenceSignalAnalysisTask',
    'HeadAnalysisTask',
    'IGAttributionTask',
    'PeakTypeClassificationTask',
    'ESM2CrossModalAlignmentTask',
    'EVoCClusteringTask',
    'GlassBoxAttributionTask',
]
