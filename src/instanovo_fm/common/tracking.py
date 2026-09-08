"""Experiment tracking module for InstaNovo.

This module provides MLFlow-based experiment tracking for training runs,
including metrics, parameters, artifacts, datasets, and models.

Params vs tags
--------------
MLflow exposes two run-level key/value stores that are easy to confuse. They
have different intents:

- **Params** (``log_param`` / ``log_params`` / ``log_hparams``): the inputs
  that produced the run — hyperparameters, dataset paths, seeds, model
  config. Logged once per key, immutable afterwards. Use these when the
  value is something you'd want a future reader to vary if reproducing the
  experiment.
- **Tags** (``mlflow.set_tags`` via the ``tags`` constructor argument): free
  post-hoc metadata for filtering and grouping in the MLflow UI — variant
  labels, owners, ticket IDs, "this is the run we shipped", etc. Mutable
  and ad-hoc. Use these when the value is descriptive but doesn't define
  the experiment.

See https://stackoverflow.com/a/72499899 for the canonical short summary.

Example usage:
    from instanovo_fm.common.tracking import create_tracker

    # Create tracker from config
    tracker = create_tracker(config, run_name="my_experiment")

    # Log metrics
    tracker.log_scalar("train/loss", 0.5, step=100)

    # Log hyperparameters
    tracker.log_hparams({"learning_rate": 0.001, "batch_size": 32})

    # Close when done
    tracker.close()
"""

from __future__ import annotations

import math
import os
import tempfile
import traceback
import warnings
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
import torch.nn as nn
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog
from instanovo_fm.utils.git_utils import get_current_git_branch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = ColorLog(console, __name__).logger


def _apply_author_mlflow_token() -> None:
    """Apply the per-author MLflow personal access token from the environment.

    AIchor injects a per-user MLflow personal access token (PAT) for each
    colleague as a ``{NAME}__MLFLOW_TOKEN`` env var derived from the commit
    author's email (e.g. ``ALICE__MLFLOW_TOKEN``, ``BOB__MLFLOW_TOKEN``). The
    prod MLflow server uses HTTP basic auth, with the user's full email as
    the username and the PAT as the password — so we install the PAT as
    ``MLFLOW_TRACKING_PASSWORD`` and ``VCS_AUTHOR_EMAIL`` as
    ``MLFLOW_TRACKING_USERNAME``, overwriting any shared service-account
    credentials the platform may also have injected.

    No-op when ``VCS_AUTHOR_EMAIL`` is unset (i.e. running locally) or when
    the corresponding ``{NAME}__MLFLOW_TOKEN`` env var isn't present.
    """
    author_email = os.environ.get("VCS_AUTHOR_EMAIL")
    if not author_email:
        return
    local_part = author_email.split("@", 1)[0]
    name = local_part.replace("-", "_").replace(".", "_").upper()
    var = f"{name}__MLFLOW_TOKEN"
    token = os.environ.get(var)
    if not token:
        logger.warning(f"MLflow token env var {var} not set; remote auth will likely fall back to any shared credentials")
        return
    os.environ["MLFLOW_TRACKING_USERNAME"] = author_email
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    os.environ.pop("MLFLOW_TRACKING_TOKEN", None)
    logger.info(f"Using MLflow basic-auth credentials from {var} for {author_email}")


def _apply_mlflow_env_defaults() -> None:
    """Apply our MLflow runtime defaults: async logging + extra retry headroom.

    Async logging moves log_metric / log_param / set_tag calls onto a
    background queue so the training loop doesn't block waiting for the
    tracking server (validated end-to-end in the async-vs-sync comparison
    on 149-mlflow-async — roughly 10% faster steady-state and qualitatively
    insulates training from server retries).

    Larger retry/backoff/timeout values let each call ride out the kind of
    short outages we've been seeing in the prod workspaces deployment
    instead of crashing the run at the first probe.

    Each variable is only set when missing so anything supplied by the
    platform/user still wins.

    - ``MLFLOW_ENABLE_ASYNC_LOGGING=True``      — buffer logs on a background thread.
    - ``MLFLOW_HTTP_REQUEST_MAX_RETRIES=10``    — retry budget per call.
    - ``MLFLOW_HTTP_REQUEST_TIMEOUT=300``       — per-attempt timeout in seconds.
    - ``MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR=2``  — exponential backoff base.

    Together these give a single MLflow call ~17 minutes of cumulative
    retry/backoff before raising — long enough for typical server hiccups
    we've observed in the prod workspaces deployment.
    """
    defaults = {
        "MLFLOW_ENABLE_ASYNC_LOGGING": "True",
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "10",
        "MLFLOW_HTTP_REQUEST_TIMEOUT": "300",
        "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": "2",
    }
    applied = []
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            applied.append(key)
    if applied:
        logger.info(f"Applied MLflow env defaults for: {', '.join(applied)}")


def _silence_mlflow_int_schema_warning() -> None:
    """Suppress MLflow's hint about integer columns in inferred schemas.

    When we hand an InstaNovo HuggingFace Dataset (with integer columns like
    ``peptide_length`` / ``precursor_charge``) to ``mlflow.data.from_pandas``
    or ``from_huggingface``, MLflow emits a ``UserWarning`` warning:

        Hint: Inferred schema contains integer column(s). Integer columns
        in Python cannot represent missing values...

    The hint is only relevant when *inference-time* data could have missing
    integer values, which doesn't apply to our validation/evaluation
    datasets where those columns are always populated. The filter is
    pinned to the exact message prefix so other MLflow warnings still
    surface.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"Hint: Inferred schema contains integer column\(s\).*",
        category=UserWarning,
    )


def _to_peptide_string(value: Any) -> str:
    """Coerce a per-row peptide value to the string form Metrics expects.

    Trainer/predictor pass peptides as either plain strings (``"NHGMHFR"`` or
    ``"N,H,M[UNIMOD:35],G,..."``) or as lists of tokens (``["N","H","G",...]``).
    ``Metrics.compute_precision_recall`` splits on commas when present and on
    characters otherwise, so emit comma-joined form for list/tuple inputs
    instead of letting ``str([...])`` produce the literal ``"['N','H',...]"``
    that triggers ``ConfigKeyError`` inside the residues lookup.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        for token in value:
            if token is None:
                raise TypeError(
                    "Encountered None inside a peptide token list; expected each token to be a "
                    "non-None string. Check the upstream decoder/predictions_to_df output."
                )
        return ",".join(str(token) for token in value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _mask_uri_userinfo(uri: str) -> str:
    """Return ``uri`` with any embedded ``user:password@`` replaced by ``***@``.

    Only http/https URIs are masked; other schemes (file://, ./mlruns, ...)
    are returned unchanged.
    """
    if not uri or not uri.startswith(("http://", "https://")):
        return uri
    try:
        parts = urlsplit(uri)
    except ValueError:
        return uri
    if not parts.username and not parts.password:
        return uri
    host = parts.hostname or ""
    if parts.port:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, f"***@{host}", parts.path, parts.query, parts.fragment))


class MLFlowTracker:
    """MLFlow experiment tracker.

    Provides comprehensive experiment tracking including metrics, parameters,
    artifacts, datasets, models, and system metrics.
    """

    def __init__(
        self,
        run_name: str,
        experiment_name: str | None = None,
        tracking_uri: str | None = None,
        workspace: str | None = None,
        tags: dict[str, str] | None = None,
        log_system_metrics: bool = True,
        allow_local_fallback: bool = True,
        parent_run_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize MLFlow tracker.

        Args:
            run_name: Name for the MLFlow run
            experiment_name: Optional experiment name (defaults to run_name)
            tracking_uri: Optional MLFlow tracking URI
            workspace: Optional MLFlow workspace (lowercase). When set, the
                experiment is created inside this workspace via
                ``mlflow.set_workspace``.
            tags: Optional mapping of tag name to value applied to the run.
                Tags are descriptive metadata used for filtering/grouping in
                the MLflow UI; use ``log_param``/``log_hparams`` for the
                hyperparameters that *produce* the run. See the module
                docstring for the params vs tags distinction.
            log_system_metrics: Whether to log system metrics (GPU, CPU, memory)
            allow_local_fallback: Whether to fall back to local MLFlow store
                when remote authentication fails
            parent_run_id: Optional MLFlow run ID of a parent run. When set,
                the new run is created in the parent's experiment (overriding
                ``experiment_name``) and tagged with ``mlflow.parentRunId`` so
                it renders nested under the parent in the UI.
            **kwargs: Additional arguments (ignored)
        """
        try:
            import mlflow

            self._mlflow = mlflow
        except ImportError as e:
            raise ImportError("MLFlow is not installed. Install it with: pip install mlflow") from e

        # On AIchor, resolve the commit author's MLflow token and copy it into
        # MLFLOW_TRACKING_TOKEN before any MLflow call so the SDK picks up
        # bearer-token auth instead of failing back to local.
        _apply_author_mlflow_token()
        # Turn on async logging + raise the retry/backoff budget so the
        # training loop doesn't block on a flaky tracking server and a
        # transient hiccup at the first MLflow probe doesn't kill the run.
        _apply_mlflow_env_defaults()
        # Squelch the (informational) integer-schema hint MLflow emits on
        # every from_pandas / from_huggingface call.
        _silence_mlflow_int_schema_warning()

        self._ephemeral_tracking_dir: tempfile.TemporaryDirectory[str] | None = None

        # Ensure we can safely create a new run in long-lived processes/tests.
        # Do this before potentially changing tracking URI, to avoid ending runs
        # against the wrong backend.
        active_run = mlflow.active_run()
        if active_run is not None:
            try:
                logger.warning(f"Ending existing active MLFlow run before starting a new one: {active_run.info.run_id}")
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to end existing active MLFlow run: {e}")

        # Configure tracking URI
        effective_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if isinstance(effective_uri, str) and effective_uri.strip().lower() == "ephemeral":
            self._ephemeral_tracking_dir = tempfile.TemporaryDirectory(prefix="instanovo-mlflow-")
            effective_uri = f"file://{self._ephemeral_tracking_dir.name}"
            logger.info(f"MLFlow tracking URI set to ephemeral local store: {effective_uri}")

        if effective_uri:
            mlflow.set_tracking_uri(effective_uri)
            logger.info(f"MLFlow tracking URI set to {_mask_uri_userinfo(effective_uri)}")

            # Check authentication for remote URIs
            if effective_uri.startswith(("http://", "https://")):
                self._check_auth_config(effective_uri)
        else:
            # Default to local file store
            mlflow.set_tracking_uri("./mlruns")
            logger.info("MLFlow tracking URI set to local file store: ./mlruns")

        # Select workspace (new MLflow self-hosting model) and create the
        # experiment. Both happen inside the same try so a workspace permission
        # error (HTTP 403) triggers the same local fallback as an auth error.
        exp_name = experiment_name or run_name
        parent_experiment_id: str | None = None
        try:
            if workspace:
                mlflow.set_workspace(workspace)
                logger.info(f"MLFlow workspace set to {workspace}")
            if parent_run_id:
                try:
                    parent_info = mlflow.get_run(parent_run_id).info
                    parent_experiment_id = parent_info.experiment_id
                    if experiment_name and parent_experiment_id:
                        logger.info(
                            f"parent_run_id={parent_run_id} provided; overriding experiment_name={exp_name!r} "
                            f"with parent's experiment_id={parent_experiment_id} so the child renders nested in the UI"
                        )
                except Exception as e:
                    resolved_uri = _mask_uri_userinfo(mlflow.get_tracking_uri())
                    msg = (
                        f"Failed to look up parent_run_id={parent_run_id} on tracking URI {resolved_uri} "
                        f"(workspace={workspace!r}): {type(e).__name__}: {e}. The parent run must exist on "
                        f"the *same* tracking server AND workspace as the child — verify the run ID in the "
                        f"MLflow UI and confirm mlflow_workspace matches the workspace shown in the URL."
                    )
                    raise RuntimeError(msg) from e
            if parent_experiment_id is not None:
                mlflow.set_experiment(experiment_id=parent_experiment_id)
            else:
                mlflow.set_experiment(exp_name)
        except Exception as e:
            logger.error(f"MLFlow set_workspace/set_experiment failed: {type(e).__name__}: {e}")
            self._handle_auth_error(e, effective_uri)
            if allow_local_fallback and self._is_auth_error(e) and effective_uri and effective_uri.startswith(("http://", "https://")):
                logger.warning("Falling back to local MLFlow file store due to remote authentication failure")
                effective_uri = self._apply_local_fallback(workspace, exp_name)
            else:
                raise

        # Enable system metrics logging if requested
        if log_system_metrics:
            try:
                mlflow.enable_system_metrics_logging()
                logger.info("MLFlow system metrics logging enabled")
            except Exception as e:
                logger.warning(f"Failed to enable system metrics logging: {e}")

        # Start run
        try:
            self.run = mlflow.start_run(run_name=run_name)
        except Exception as e:
            logger.error(f"MLFlow start_run failed: {type(e).__name__}: {e}")
            self._handle_auth_error(e, effective_uri)
            if allow_local_fallback and self._is_auth_error(e) and effective_uri and effective_uri.startswith(("http://", "https://")):
                logger.warning("Falling back to local MLFlow file store due to run-start authentication failure")
                effective_uri = self._apply_local_fallback(workspace, exp_name)
                self.run = mlflow.start_run(run_name=run_name)
            else:
                raise
        self.run_id = self.run.info.run_id
        self.experiment_id = self.run.info.experiment_id

        if parent_run_id:
            mlflow.set_tag("mlflow.parentRunId", parent_run_id)
            logger.info(f"MLFlow run {self.run_id} attached as child of parent run {parent_run_id}")

        if tags:
            mlflow.set_tags(tags)

        # Log run info with clickable URL pointing into the right workspace
        # so the link drops the reader directly on the run page.
        tracking_uri = mlflow.get_tracking_uri()
        if tracking_uri.startswith(("http://", "https://")):
            base = _mask_uri_userinfo(tracking_uri).rstrip("/")
            workspace_qs = f"?workspace={workspace}" if workspace else ""
            run_url = f"{base}/#/experiments/{self.experiment_id}/runs/{self.run_id}{workspace_qs}"
            experiment_url = f"{base}/#/experiments/{self.experiment_id}{workspace_qs}"
            logger.info(f"MLFlow run started: {run_name} (ID: {self.run_id})")
            logger.info(f"MLFlow run URL:        {run_url}")
            logger.info(f"MLFlow experiment URL: {experiment_url}")
        else:
            logger.info(f"MLFlow run started: {run_name} (ID: {self.run_id})")

    def log_scalar(self, tag: str, value: float, step: int | None = None) -> None:
        """Log a scalar metric to MLFlow."""
        self._check_nan(tag, value, step)
        self._mlflow.log_metric(key=tag, value=value, step=step)

    def log_text(self, artifact_file: str, text: str) -> None:
        """Log text to MLFlow as an artifact file."""
        self._mlflow.log_text(text=text, artifact_file=artifact_file)

    def log_param(self, key: str, value: str) -> None:
        """Log a run-level parameter. Shown in Parameters in MLflow UI."""
        self._mlflow.log_param(key=key, value=value)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log multiple run-level parameters."""
        self._mlflow.log_params(params)

    def log_hparams(self, params: dict[str, Any], metrics: dict[str, Any] | None = None) -> None:
        """Log hyperparameters to MLFlow."""
        # Flatten nested dicts and convert to strings for non-primitive types
        flat_params = self._flatten_params(params)
        self._mlflow.log_params(flat_params)

        # Note: metrics here are summary metrics associated with hyperparameters,
        # not time-series metrics. They're logged without step as final values.
        if metrics:
            for key, value in metrics.items():
                self._mlflow.log_metric(key=key, value=value)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a file or directory as an artifact."""
        if os.path.isdir(local_path):
            self._mlflow.log_artifacts(local_path, artifact_path=artifact_path)
        else:
            self._mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_dataset(
        self,
        context: str,
        name: str,
        source: str,
        num_samples: int,
    ) -> None:
        """Log dataset metadata to MLFlow.

        Logs the dataset source path and sample count as metrics.
        For in-memory HuggingFace datasets, prefer ``log_hf_dataset``
        which also records schema and digest via ``mlflow.data``.

        Args:
            context: Dataset context (e.g., "training", "validation")
            name: Human-readable dataset name
            source: Source path or URL for lineage tracking
            num_samples: Total number of samples in the dataset
        """
        try:
            self._mlflow.log_metric(f"dataset.{context}.total_samples", num_samples, step=0)
            self._mlflow.log_param(f"dataset.{context}.name", name)
            self._mlflow.log_param(f"dataset.{context}.source", source[:250])
        except Exception as e:
            logger.warning(f"Failed to log dataset metadata to MLFlow: {e}")

    def log_hf_dataset(
        self,
        data: Any,
        context: str,
        name: str,
        source: str,
        num_samples: int,
        targets: str | None = None,
    ) -> None:
        """Log an in-memory HuggingFace Dataset to MLFlow.

        Uses ``mlflow.data.from_huggingface`` for proper schema, digest, and
        source tracking. Falls back to ``log_dataset`` on failure.

        Args:
            data: HuggingFace Dataset (must not be an IterableDataset)
            context: Dataset context (e.g., "training", "validation")
            name: Human-readable dataset name
            source: Source path or URL for lineage tracking
            num_samples: Total number of samples
            targets: Optional target column name
        """
        try:
            import mlflow.data

            mlflow_dataset = mlflow.data.from_huggingface(
                data,
                path=source,
                name=name,
                targets=targets,
            )
            self._mlflow.log_input(mlflow_dataset, context=context)
            self._mlflow.log_metric(f"dataset.{context}.total_samples", num_samples, step=0)
            logger.debug(f"Logged HF dataset '{name}' to MLFlow (digest: {mlflow_dataset.digest})")
        except Exception as e:
            logger.debug(f"from_huggingface failed, falling back to metadata-only: {e}")
            self.log_dataset(context=context, name=name, source=source, num_samples=num_samples)

    def log_evaluation_dataset(
        self,
        data: pd.DataFrame,
        targets: str,
        predictions: str,
        dataset_name: str,
        context: str = "evaluation",
    ) -> None:
        """Log an evaluation dataset with predictions for MLFlow evaluate integration.

        Creates an EvaluationDataset that can be used with mlflow.models.evaluate().

        See: https://mlflow.org/docs/latest/ml/dataset/#integration-with-mlflow-evaluate

        Args:
            data: DataFrame containing features, targets, and predictions
            targets: Name of the target column
            predictions: Name of the predictions column
            dataset_name: Name for the dataset
            context: Context for the dataset
        """
        try:
            import mlflow.data

            # Create evaluation dataset with predictions specified
            eval_dataset = mlflow.data.from_pandas(
                data,
                source=self._local_dataset_source(f"evaluation_{dataset_name}"),
                name=dataset_name,
                targets=targets,
                predictions=predictions,
            )

            # Log the evaluation dataset — schema, digest, and source live on
            # the logged Dataset itself; no need to also stamp them as tags.
            self._mlflow.log_input(eval_dataset, context=context)

            logger.debug(f"Logged evaluation dataset '{dataset_name}' to MLFlow (digest: {eval_dataset.digest})")

        except Exception as e:
            logger.warning(f"Failed to log evaluation dataset to MLFlow: {e}")

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
        input_example: Any | None = None,
    ) -> None:
        """Log a PyTorch model to MLFlow Model Registry.

        To avoid pickling issues with Accelerate's automatic mixed precision (AMP),
        we save/reload the model through a buffer to strip AMP-related state.

        Args:
            model: PyTorch model to log
            artifact_path: Path within artifacts to store the model
            registered_model_name: Optional name to register in model registry
            input_example: Optional example input for signature inference (used
                to create an explicit tensor-based signature)
        """
        import copy
        import io
        import tempfile

        try:
            import mlflow.pytorch
            import torch
            from mlflow.models import ModelSignature
            from mlflow.types.schema import Schema, TensorSpec

            # Create a deep copy of the model to avoid affecting the training model
            # Then move the copy to CPU and eval mode
            model_copy = copy.deepcopy(model)
            model_copy = model_copy.cpu().eval()

            # Save entire model to buffer and reload to strip AMP hooks
            buffer = io.BytesIO()
            torch.save(model_copy, buffer, pickle_protocol=4)
            buffer.seek(0)

            # Reload model - this creates a clean copy without AMP context
            clean_model = torch.load(buffer, map_location="cpu", weights_only=False)
            clean_model.eval()

            # Create explicit signature from input example shape and model config
            signature = None
            if input_example is not None:
                try:
                    import numpy as np

                    from instanovo.constants import MAX_SEQUENCE_LENGTH

                    # input_example is expected to be a numpy array (spectra)
                    if isinstance(input_example, np.ndarray):
                        # Create tensor spec from shape: (batch, peaks, features)
                        input_schema = Schema(
                            [
                                TensorSpec(
                                    np.dtype("float32"),
                                    (-1,) + input_example.shape[1:],
                                    "spectra",
                                )
                            ]
                        )

                        # Try to get vocab_size and max_length from model
                        vocab_size = -1
                        max_length = MAX_SEQUENCE_LENGTH  # Default from constants

                        if hasattr(clean_model, "vocab_size"):
                            vocab_size = clean_model.vocab_size
                        elif hasattr(clean_model, "head") and hasattr(clean_model.head, "out_features"):
                            vocab_size = clean_model.head.out_features
                        elif hasattr(clean_model, "config"):
                            vocab_size = getattr(clean_model.config, "vocab_size", -1)

                        # Output is token logits: (batch, max_length, vocab_size)
                        output_schema = Schema(
                            [
                                TensorSpec(
                                    np.dtype("float32"),
                                    (-1, max_length, vocab_size),
                                    "logits",
                                )
                            ]
                        )
                        signature = ModelSignature(inputs=input_schema, outputs=output_schema)
                        logger.debug(f"Created model signature: input={input_example.shape}, output=(-1, {max_length}, {vocab_size})")
                except Exception as e:
                    logger.debug(f"Could not create model signature: {e}")

            # Log the clean model with explicit signature
            mlflow.pytorch.log_model(
                clean_model,
                name=artifact_path,
                registered_model_name=registered_model_name,
                signature=signature,
            )

            if registered_model_name:
                logger.info(f"Model registered to MLFlow Model Registry: {registered_model_name}")
            else:
                logger.info(f"Model logged to MLFlow: {artifact_path}")

        except Exception as e:
            # Fall back to logging state_dict as artifact
            logger.debug(f"Could not log full model: {e}")
            logger.info("Falling back to logging state_dict as artifact")

            try:
                import torch

                with tempfile.TemporaryDirectory() as tmpdir:
                    temp_model_path = os.path.join(tmpdir, "model_state_dict.pt")
                    torch.save(model.state_dict(), temp_model_path)
                    self._mlflow.log_artifact(temp_model_path, artifact_path=artifact_path)

                logger.info(f"Model state_dict logged to MLFlow: {artifact_path}/model_state_dict.pt")

                if registered_model_name:
                    logger.warning(f"Model registry '{registered_model_name}' requires a full model. State_dict logged as artifact instead.")
            except Exception as fallback_error:
                logger.warning(f"Failed to log model to MLFlow: {fallback_error}")

    def log_evaluation(
        self,
        predictions: list[str],
        targets: list[str],
        dataset_name: str,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        """Log evaluation results to MLFlow."""
        # Log metrics with dataset name prefix
        for metric_name, metric_value in metrics.items():
            self.log_scalar(f"eval/{dataset_name}/{metric_name}", metric_value, step)

        # Log evaluation data as dataset
        try:
            import mlflow.data

            eval_df = pd.DataFrame(
                {
                    "prediction": predictions[:1000],  # Limit size for large datasets
                    "target": targets[:1000],
                }
            )
            dataset = mlflow.data.from_pandas(
                eval_df,
                source=self._local_dataset_source(f"evaluation_{dataset_name}"),
                name=f"{dataset_name}_evaluation",
            )
            self._mlflow.log_input(dataset, context=f"evaluation_{dataset_name}")
        except Exception as e:
            logger.debug(f"Failed to log evaluation dataset: {e}")

    def _local_dataset_source(self, source_name: str) -> Any:
        """Wrap ``source_name`` in a ``LocalArtifactDatasetSource``.

        Passing a bare string to ``mlflow.data.from_pandas(source=...)``
        triggers MLflow's source-registry resolver which prints
        ``UserWarning: The specified dataset source can be interpreted in
        multiple ways: LocalArtifactDatasetSource, LocalArtifactDatasetSource.``
        on every call. Building the source object ourselves picks the
        intended type up front and silences the warning.
        """
        from mlflow.data.sources import LocalArtifactDatasetSource

        return LocalArtifactDatasetSource(source_name)

    def _compute_fallback_peptide_metrics(
        self,
        targets: list[str],
        predictions: list[str],
        confidence: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute approximate peptide metrics when no Metrics object is provided."""

        def split_tokens(seq: str) -> list[str]:
            if "," in seq:
                return [token.strip() for token in seq.split(",") if token.strip()]
            return list(seq)

        n_targ_aa = 0
        n_pred_aa = 0
        n_match_aa = 0
        n_match_pep = 0
        n_pred_pep = 0
        is_match: list[int] = []

        for target, pred in zip(targets, predictions, strict=True):
            target_tokens = split_tokens(target)
            pred_tokens = split_tokens(pred)
            n_targ_aa += len(target_tokens)

            if len(pred_tokens) > 0:
                n_pred_aa += len(pred_tokens)
                n_pred_pep += 1

            aa_matches = sum(1 for a, b in zip(target_tokens, pred_tokens, strict=True) if a == b)
            n_match_aa += aa_matches
            pep_match = int(
                len(pred_tokens) == len(target_tokens) and aa_matches == len(target_tokens),
            )
            n_match_pep += pep_match
            is_match.append(pep_match)

        pep_recall = n_match_pep / len(targets) if len(targets) > 0 else 0.0
        pep_prec = n_match_pep / n_pred_pep if n_pred_pep > 0 else 1.0
        aa_recall = n_match_aa / n_targ_aa if n_targ_aa > 0 else 0.0
        aa_prec = n_match_aa / n_pred_aa if n_pred_aa > 0 else 1.0

        aa_er = float(np.mean([int(t != p) for t, p in zip(targets, predictions, strict=True)])) if targets else 0.0
        auc = float("nan")
        if confidence is not None and len(confidence) == len(is_match) and len(is_match) > 1:
            order = np.argsort(confidence)[::-1]
            labels = np.array(is_match)[order]
            tp = np.cumsum(labels)
            precision = tp / (np.arange(len(labels)) + 1)
            recall = tp / len(labels)
            auc = float(np.trapz(precision, recall))

        return {
            "aa_er": float(aa_er),
            "aa_prec": float(aa_prec),
            "aa_recall": float(aa_recall),
            "pep_prec": float(pep_prec),
            "pep_recall": float(pep_recall),
            "auc": float(auc),
        }

    def log_mlflow_evaluation(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        context: str,
        step: int | None = None,
        targets_col: str = "target",
        predictions_col: str = "prediction",
        confidence_col: str | None = "confidence",
        delta_mass_col: str = "delta_mass_ppm",
        peptide_metrics: Any | None = None,
        log_artifacts: bool = True,
    ) -> dict[str, Any] | None:
        """Log an MLflow EvaluationDataset and a bundle of diagnostic artifacts.

        Specifically: registers ``data`` as an ``mlflow.data`` dataset for
        lineage, then calls ``mlflow.models.evaluate`` to produce diagnostic
        artifacts (confidence histogram, peptide PR curve, top-error CSV,
        delta-mass distribution).

        This function does NOT log scalar metrics. The trainer logs validation
        metrics directly under ``eval/...`` and the predictor under
        ``predict/...``; emitting a parallel ``mlflow_eval.<context>.peptide_*``
        series would duplicate those.
        """
        required_columns = {targets_col, predictions_col}
        missing = required_columns - set(data.columns)
        if missing:
            logger.warning(f"Skipping MLFlow evaluate for '{dataset_name}'; missing columns: {sorted(missing)}")
            return None

        eval_df = data.copy()

        eval_df[targets_col] = eval_df[targets_col].apply(_to_peptide_string)
        eval_df[predictions_col] = eval_df[predictions_col].apply(_to_peptide_string)

        self.log_evaluation_dataset(
            data=eval_df,
            targets=targets_col,
            predictions=predictions_col,
            dataset_name=dataset_name,
            context=context,
        )

        try:
            if not log_artifacts:
                logger.info(f"MLFlow evaluation dataset logged for '{dataset_name}' ({len(eval_df):,} rows); artifacts skipped")
                return {"artifacts": {}}

            # Build diagnostic artifacts (PR curve, confidence histogram,
            # delta-mass plot, top-error CSV) into a temp dir and log each
            # via mlflow.log_artifact. We used to do this through
            # mlflow.models.evaluate(custom_artifacts=...) but that API
            # requires extra_metrics whenever model_type=None, and we
            # already dropped extra_metrics in 213b7d8eb because every
            # metric it produced was a duplicate of eval/* / predict/*
            # logged elsewhere. Calling the artifact builder directly
            # bypasses the evaluate-API constraints and keeps the
            # artifacts on the run.
            artifacts: dict[str, str] = {}
            with tempfile.TemporaryDirectory(prefix="instanovo-mlflow-eval-") as artifacts_dir:
                self._build_eval_artifacts(
                    eval_df=eval_df,
                    artifacts_dir=artifacts_dir,
                    targets_col=targets_col,
                    predictions_col=predictions_col,
                    confidence_col=confidence_col,
                    delta_mass_col=delta_mass_col,
                    peptide_metrics=peptide_metrics,
                    out=artifacts,
                )
                for name, path in artifacts.items():
                    self._mlflow.log_artifact(path)
                    artifacts[name] = os.path.basename(path)

            logger.info(f"MLFlow evaluation logged for dataset '{dataset_name}' ({len(eval_df):,} rows)")
            return {"artifacts": artifacts}
        except Exception as e:
            logger.warning(f"MLFlow evaluate failed for '{dataset_name}': {e}")
            return None

    def _build_eval_artifacts(
        self,
        eval_df: pd.DataFrame,
        artifacts_dir: str,
        targets_col: str,
        predictions_col: str,
        confidence_col: str | None,
        delta_mass_col: str,
        peptide_metrics: Any | None,
        out: dict[str, str],
    ) -> None:
        """Build the diagnostic artifact files into ``artifacts_dir`` and record their paths in ``out``."""
        import matplotlib.pyplot as plt

        # Confidence distribution
        if confidence_col and confidence_col in eval_df.columns:
            confidence_series = pd.to_numeric(eval_df[confidence_col], errors="coerce").dropna()
            if len(confidence_series) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(confidence_series, bins=40)
                ax.set_title("Prediction Confidence Distribution")
                ax.set_xlabel("confidence")
                ax.set_ylabel("count")
                confidence_path = os.path.join(artifacts_dir, "confidence_histogram.png")
                fig.savefig(confidence_path, bbox_inches="tight")
                plt.close(fig)
                out["confidence_histogram"] = confidence_path

                if len(confidence_series.unique()) > 1:
                    thresholds = np.linspace(confidence_series.min(), confidence_series.max(), num=20)
                    recalls: list[float] = []
                    precisions: list[float] = []
                    preds = eval_df[predictions_col].fillna("").astype(str).tolist()
                    targs = eval_df[targets_col].fillna("").astype(str).tolist()
                    conf = confidence_series.to_numpy(dtype=float)
                    if len(conf) == len(preds):
                        for threshold in thresholds:
                            if peptide_metrics is not None:
                                _, _, pep_recall, pep_prec = peptide_metrics.compute_precision_recall(
                                    targs,
                                    preds,
                                    conf.tolist(),
                                    float(threshold),
                                )
                            else:
                                selected = conf >= threshold
                                selected_preds = [p if flag else "" for p, flag in zip(preds, selected, strict=True)]
                                fallback = self._compute_fallback_peptide_metrics(
                                    targets=targs,
                                    predictions=selected_preds,
                                    confidence=conf,
                                )
                                pep_recall = fallback["pep_recall"]
                                pep_prec = fallback["pep_prec"]

                            recalls.append(float(pep_recall))
                            precisions.append(float(pep_prec))

                        fig, ax = plt.subplots(figsize=(6, 6))
                        ax.plot(recalls, precisions, marker="o", linewidth=1)
                        ax.set_title("Peptide PR Curve (confidence thresholds)")
                        ax.set_xlabel("peptide_recall")
                        ax.set_ylabel("peptide_precision")
                        pr_path = os.path.join(artifacts_dir, "peptide_pr_curve.png")
                        fig.savefig(pr_path, bbox_inches="tight")
                        plt.close(fig)
                        out["peptide_pr_curve"] = pr_path

        # Delta mass diagnostics
        if delta_mass_col in eval_df.columns:
            delta_mass = pd.to_numeric(eval_df[delta_mass_col], errors="coerce").dropna()
            if len(delta_mass) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(delta_mass, bins=80)
                ax.set_title("Delta Mass Distribution (ppm)")
                ax.set_xlabel("delta_mass_ppm")
                ax.set_ylabel("count")
                delta_mass_path = os.path.join(artifacts_dir, "delta_mass_distribution.png")
                fig.savefig(delta_mass_path, bbox_inches="tight")
                plt.close(fig)
                out["delta_mass_distribution"] = delta_mass_path

        # Top error samples
        mismatch_df = eval_df[eval_df[predictions_col] != eval_df[targets_col]].head(100)
        if len(mismatch_df) > 0:
            mismatch_path = os.path.join(artifacts_dir, "top_error_samples.csv")
            mismatch_df.to_csv(mismatch_path, index=False)
            out["top_error_samples"] = mismatch_path

    def log_figure(
        self,
        tag: str,
        figure: Figure,
        step: int | None = None,
        close: bool = True,
    ) -> None:
        """Log a matplotlib figure to MLFlow."""
        import matplotlib.pyplot as plt

        try:
            # MLFlow can log figures directly
            self._mlflow.log_figure(
                figure,
                f"figures/{tag}_step_{step}.png" if step else f"figures/{tag}.png",
            )
        except Exception as e:
            logger.warning(f"Failed to log figure to MLFlow: {e}")

        if close:
            plt.close(figure)

    def close(self) -> None:
        """End the MLFlow run."""
        self._mlflow.end_run()
        if self._ephemeral_tracking_dir is not None:
            self._ephemeral_tracking_dir.cleanup()
            self._ephemeral_tracking_dir = None
        logger.info(f"MLFlow run ended: {self.run_id}")

    def _check_nan(self, tag: str, value: float, step: int | None) -> None:
        """Check for NaN values and raise an error if detected."""
        if math.isnan(value):
            error_msg = (
                f"NaN value detected when logging metric '{tag}' at step {step}. "
                f"This indicates a serious training problem (e.g., exploding gradients, division by zero). "
                f"Stopping training to prevent further issues.\n\n"
                f"Traceback showing where this NaN value originated:\n"
            )
            stack_trace = traceback.format_stack()
            relevant_frames = stack_trace[:-1][-6:]
            error_msg += "".join(relevant_frames)
            raise ValueError(error_msg)

    def _flatten_params(self, params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested parameter dictionaries."""
        flat: dict[str, Any] = {}
        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_params(value, full_key))
            elif isinstance(value, (list, tuple)):
                flat[full_key] = str(value)
            elif isinstance(value, (int, float, str, bool)) or value is None:
                flat[full_key] = value
            else:
                flat[full_key] = str(value)
        return flat

    def _check_auth_config(self, tracking_uri: str) -> None:
        """Check if authentication is configured for remote MLFlow server.

        Args:
            tracking_uri: The MLFlow tracking URI
        """
        has_username = bool(os.environ.get("MLFLOW_TRACKING_USERNAME"))
        has_password = bool(os.environ.get("MLFLOW_TRACKING_PASSWORD"))
        has_token = bool(os.environ.get("MLFLOW_TRACKING_TOKEN"))

        if not (has_token or (has_username and has_password)):
            logger.warning(
                f"Remote MLFlow server detected ({_mask_uri_userinfo(tracking_uri)}) but no authentication configured. "
                f"Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD, "
                f"or MLFLOW_TRACKING_TOKEN in your environment or .env file. "
                f"See: https://mlflow.org/docs/latest/auth/index.html"
            )

    def _handle_auth_error(self, error: Exception, tracking_uri: str | None) -> None:
        """Log helpful error message for authentication failures.

        Args:
            error: The exception that was raised
            tracking_uri: The MLFlow tracking URI
        """
        error_str = str(error)
        if "401" in error_str or "not authenticated" in error_str.lower():
            logger.error(
                f"MLFlow authentication failed for {_mask_uri_userinfo(tracking_uri) if tracking_uri else tracking_uri}. "
                f"Please set one of the following in your environment or .env file:\n"
                f"  - MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD (for basic auth)\n"
                f"  - MLFLOW_TRACKING_TOKEN (for token-based auth)\n"
                f"See: https://mlflow.org/docs/latest/auth/index.html"
            )

    def _is_auth_error(self, error: Exception) -> bool:
        """Check if an exception indicates MLFlow authentication or authorization failure.

        Treats workspace permission denials (HTTP 403) the same as authentication
        failures (HTTP 401) so that a misconfigured workspace also triggers the
        local fallback path when ``allow_local_fallback`` is enabled.
        """
        error_str = str(error).lower()
        markers = (
            "401",
            "403",
            "not authenticated",
            "authentication required",
            "permission denied",
        )
        return any(marker in error_str for marker in markers)

    def _apply_local_fallback(self, workspace: str | None, exp_name: str) -> str:
        """Switch to the local ./mlruns store and re-apply workspace/experiment.

        Returns the new effective tracking URI.
        """
        self._mlflow.set_tracking_uri("./mlruns")
        if workspace:
            self._mlflow.set_workspace(workspace)
        self._mlflow.set_experiment(exp_name)
        return "./mlruns"


def _coerce_tags(raw: Any) -> dict[str, str] | None:
    """Coerce a config-supplied ``tags`` value to ``dict[str, str]``.

    Accepts ``None``, a plain ``dict``, or an ``OmegaConf.DictConfig``
    mapping. Raises ``TypeError`` for legacy list-style configs so the user
    gets a clear migration message instead of silent tag mangling.
    """
    if raw is None:
        return None
    if isinstance(raw, DictConfig):
        from omegaconf import OmegaConf

        raw = OmegaConf.to_container(raw, resolve=True)
    if isinstance(raw, dict):
        return {str(k): str(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        raise TypeError(
            "tags must be a mapping (e.g. `tags: {variant: default}`), not a list. The legacy `tags: [foo, bar]` form is no longer supported."
        )
    raise TypeError(f"tags must be a mapping, got {type(raw).__name__}")


def create_tracker(
    config: DictConfig | dict[str, Any],
    run_name: str,
) -> MLFlowTracker:
    """Create an MLFlow experiment tracker.

    Args:
        config: Configuration dictionary or DictConfig
        run_name: Name for the experiment run

    Returns:
        An MLFlowTracker instance

    Config options:
        mlflow_tracking_uri: MLFlow server URI
            - Use ``ephemeral`` for a temporary local backend (useful in tests)
        mlflow_experiment_name: MLFlow experiment name
        mlflow_workspace: Optional MLFlow workspace (lowercase). The
            experiment will be created inside this workspace.
        mlflow_log_system_metrics: Whether to log system metrics
        mlflow_parent_run_id: Optional MLflow run ID. When set, the new run
            is created in the parent's experiment and tagged with
            ``mlflow.parentRunId`` so it renders nested under the parent.
        tags: Optional mapping of tag name -> value applied to the run.
            Configure as a YAML mapping, e.g. ``tags: {variant: default}``.
            See the module docstring for the params vs tags distinction.

    Example:
        tracker = create_tracker(config, run_name="my_run")
        tracker.log_scalar("loss", 0.5, step=100)
        tracker.close()
    """
    tracking_uri = config.get("mlflow_tracking_uri", None)
    experiment_name = config.get("mlflow_experiment_name", None)
    workspace = config.get("mlflow_workspace", None)
    log_system_metrics = config.get("mlflow_log_system_metrics", True)
    allow_local_fallback = config.get("mlflow_allow_local_fallback", True)
    parent_run_id = config.get("mlflow_parent_run_id", None)
    if isinstance(parent_run_id, str):
        parent_run_id = parent_run_id.strip() or None
    tags = _coerce_tags(config.get("tags", None))

    # Default experiment name to current git branch when not overridden
    if experiment_name is None or (isinstance(experiment_name, str) and experiment_name.strip() == ""):
        experiment_name = get_current_git_branch() or run_name

    if isinstance(workspace, str):
        workspace = workspace.strip().lower() or None

    return MLFlowTracker(
        run_name=run_name,
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        workspace=workspace,
        tags=tags,
        log_system_metrics=log_system_metrics,
        allow_local_fallback=allow_local_fallback,
        parent_run_id=parent_run_id,
    )
