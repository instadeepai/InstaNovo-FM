# Foundation Model Embedding Evaluation

Comprehensive evaluation framework for InstaNovo Foundation Model embeddings. This system provides automated evaluation of learned spectral representations through a variety of tasks including retrieval, clustering, linear probes, and visualization.

## Architecture

```
instanovo_fm/eval/
├── README.md                  # This file
├── __init__.py
├── __main__.py               # CLI entry point
├── embed_evaluation.py       # Main Hydra-based script
├── evaluator.py              # EmbeddingEvaluator class
├── embedding_io.py           # I/O utilities (HDF5, FAISS)
└── eval_tasks/               # Evaluation task plugins
    ├── __init__.py           # Task registry & BaseTask
    ├── embedding_analysis.py
    ├── duplicate_retrieval.py
    ├── charge_linear_probe.py
    ├── rt_linear_probe.py
    ├── ptm_linear_probe.py
    ├── kmeans_clustering.py
    └── ... (other tasks)
```

## Quick Start

### 1. Standalone Evaluation (After Training)

Evaluate a trained model checkpoint:

```bash
# Basic usage (uses config from foundational.yaml)
python -m instanovo_fm.eval.embed_evaluation \
    resume_checkpoint_path=checkpoints/instanovo-foundational-base/model_best.ckpt \
    evaluation.enabled=True

# Use dedicated evaluation config
python -m instanovo_fm.eval.embed_evaluation \
    --config-name evaluation \
    resume_checkpoint_path=checkpoints/instanovo-foundational-base/model_best.ckpt

# Evaluate on test split
python -m instanovo_fm.eval.embed_evaluation \
    resume_checkpoint_path=checkpoints/instanovo-foundational-base/model_best.ckpt \
    evaluation.enabled=True \
    evaluation.split=test

# Run specific tasks only
python -m instanovo_fm.eval.embed_evaluation \
    resume_checkpoint_path=checkpoints/instanovo-foundational-base/model_best.ckpt \
    evaluation.enabled=True \
    evaluation.tasks_to_run=[embeddinganalysistask,duplicateretrievaltask]

# Use 10% of data for quick testing
python -m instanovo_fm.eval.embed_evaluation \
    resume_checkpoint_path=checkpoints/instanovo-foundational-base/model_best.ckpt \
    evaluation.enabled=True \
    evaluation.data_subset=0.1
```

### 2. Integrated with Training

Enable automatic evaluation after training completes:

```yaml
# In foundational.yaml
run_post_training_evaluation: True
evaluation:
  enabled: True
  split: valid
  tasks_to_run:
    - embeddinganalysistask
    - duplicateretrievaltask
```

Then run training as normal:

```bash
python -m instanovo_fm.trainer.train
```

## Configuration

### Main Configuration (`foundational.yaml`)

```yaml
# Post-training evaluation flag
run_post_training_evaluation: False  # Set to True to run after training

evaluation:
  enabled: False  # Enable/disable evaluation
  split: valid    # "valid" or "test"
  batch_size: 256
  device: auto    # "auto", "cuda", or "cpu"
  output_dir: "./evaluation_results"
  data_subset: 1.0  # 1.0 = 100%, 0.1 = 10%
  force_regenerate_embeddings: False
  
  # Tasks to run (null = all tasks)
  tasks_to_run:
    - embeddinganalysistask
    - duplicateretrievaltask
    - chargelinearprobetask
    - rtlinearprobetask
  
  # Task-specific configurations
  task_configs:
    embeddinganalysistask:
      n_clusters: 50
      compute_pca: True
    duplicateretrievaltask:
      k_values: [1, 5, 10]
      max_samples: 1000
```

### Evaluation-Only Configuration (`evaluation.yaml`)

For standalone evaluation, use the dedicated config:

```bash
python -m instanovo_fm.eval.embed_evaluation \
    --config-name evaluation \
    resume_checkpoint_path=path/to/checkpoint.ckpt
```

## Usage Patterns

### Pattern 1: Quick Sanity Check

Run a fast subset of tasks on 10% of data:

```bash
python -m instanovo_fm.eval.embed_evaluation \
    resume_checkpoint_path=model_best.ckpt \
    evaluation.enabled=True \
    evaluation.data_subset=0.1 \
    evaluation.tasks_to_run=[duplicateretrievaltask,embeddinganalysistask]
```

**Expected time:** ~2-5 minutes  
**Purpose:** Verify embeddings are meaningful (not collapsed)

### Pattern 2: Comprehensive Evaluation

Run all tasks on full validation set:

```bash
python -m instanovo_fm.eval.embed_evaluation \
    --config-name evaluation \
    resume_checkpoint_path=model_best.ckpt
```

**Expected time:** ~30-60 minutes (depending on dataset size)  
**Purpose:** Full evaluation for paper/report

### Pattern 3: Specific Task with Custom Config

Run a single task with custom parameters:

```bash
python -m instanovo_fm.eval.embed_evaluation \
    resume_checkpoint_path=model_best.ckpt \
    evaluation.enabled=True \
    evaluation.tasks_to_run=[duplicateretrievaltask] \
    evaluation.task_configs.duplicateretrievaltask.k_values=[1,5,10,20,50] \
    evaluation.task_configs.duplicateretrievaltask.max_samples=5000
```

### Pattern 4: Test Set Evaluation (Final)

Evaluate on held-out test set for final results:

```bash
python -m instanovo_fm.eval.embed_evaluation \
    --config-name evaluation \
    resume_checkpoint_path=model_best.ckpt \
    evaluation.split=test
```

**⚠️ Warning:** Only run on test set once to avoid overfitting to test metrics!

## Available Evaluation Tasks

### 1. **EmbeddingAnalysisTask** - Comprehensive Quality Analysis

Analyzes embedding quality, dimensionality, and structure.

**Metrics:**
- Basic statistics (norms, means, stds)
- PCA analysis (explained variance, intrinsic dimensionality)
- Clustering quality (silhouette score)
- Similarity distribution

**Config:**
```yaml
embeddinganalysistask:
  n_clusters: 50
  n_components_pca: 50
  compute_clustering: True
  compute_pca: True
```

### 2. **DuplicateRetrievalTask** - Retrieval Sanity Check

Fast sanity check that verifies embeddings can retrieve spectra with identical peptides.

**Metrics:**
- Recall@k (k=1,5,10,...)
- Mean Average Precision (mAP)

**Config:**
```yaml
duplicateretrievaltask:
  k_values: [1, 5, 10, 20]
  max_samples: 1000
  peptide_key: peptide
```

**Interpretation:**
- Recall@1 > 0.5: Good embeddings
- Recall@1 < 0.1: Poor/collapsed embeddings

### 3. **ChargeLinearProbeTask** - Charge Detection

Tests if embeddings encode trivial experimental factors (charge state).

**Metrics:**
- Probe accuracy (logistic regression)
- Per-class accuracy

**Config:**
```yaml
chargelinearprobetask:
  max_samples: 2000
  test_size: 0.2
  charge_key: charge_id
```

**Interpretation:**
- High accuracy (>0.7): May indicate overfitting to experimental artifacts
- Low accuracy (<0.3): Good - not encoding trivial factors

### 4. **RTLinearProbeTask** - Retention Time Detection

Tests if embeddings encode retention time information.

**Metrics:**
- RT decile classification accuracy
- Per-decile accuracy

**Config:**
```yaml
rtlinearprobetask:
  max_samples: 2000
  n_rt_deciles: 10
  rt_key: rt_log
```

### 5. **KMeansClusteringTask** - Unsupervised Clustering

Evaluates clustering quality using peptide labels as ground truth.

**Metrics:**
- V-measure score
- Adjusted Rand Index
- Silhouette score

**Config:**
```yaml
kmeansclusteringtask:
  max_samples: 5000
  peptide_key: peptide
  batch_size: 1000
```

### 6. **UMAPVisualisationTask** - 2D Visualization

Projects embeddings to 2D using UMAP for visual inspection.

**Outputs:**
- PNG file with 2D projection
- Colored by charge/hydrophobicity

**Config:**
```yaml
umapvisualisationtask:
  max_samples: 10000
  color_by: charge
  n_neighbors: 15
  min_dist: 0.1
```

### 7. **Other Tasks**

- **PTMLinearProbeTask**: Tests PTM detection
- **CosineHyperscoreCorrelationTask**: Correlates similarity with database scores
- **RTRankingConsistencyTask**: Tests RT ordering preservation
- **ESM2SimilarityCorrelationTask**: Correlates with protein embeddings
- **HeadAnalysisTask**: Analyzes peak-to-peak attention patterns per head

## Output Structure

After evaluation, results are saved to `evaluation.output_dir`:

```
evaluation_results/
├── embeddings.h5                      # Cached embeddings (HDF5)
├── index.faiss                        # FAISS index for retrieval
├── evaluation_summary.json            # Summary metrics
├── evaluation_results_full.json       # Full results
└── task_outputs/                      # Task-specific outputs
    ├── umap.png
    ├── rt_ranking_consistency.json
    └── ...
```

### Summary File Format

```json
{
  "checkpoint": "path/to/model_best.ckpt",
  "output_dir": "./evaluation_results",
  "embeddings": {
    "num_embeddings": 10000,
    "embedding_dim": 512,
    "mean_norm": 0.9998
  },
  "tasks": {
    "duplicateretrievaltask": {
      "success": true,
      "execution_time": 2.5,
      "recall_metrics": {
        "recall@1": 0.85,
        "recall@5": 0.95,
        "recall@10": 0.98
      },
      "map": 0.92
    },
    ...
  }
}
```

## Programming Interface

### Using the EmbeddingEvaluator Class

```python
from omegaconf import OmegaConf
from instanovo_fm.eval.evaluator import EmbeddingEvaluator

# Load config
config = OmegaConf.load("instanovo/configs/foundational.yaml")
config.resume_checkpoint_path = "checkpoints/model_best.ckpt"
config.evaluation.enabled = True

# Create evaluator
evaluator = EmbeddingEvaluator(config)

# Run evaluation
results = evaluator.evaluate(split="valid", force_regenerate=False)

# Access results
print(f"Recall@1: {results['duplicateretrievaltask']['recall_metrics']['recall@1']}")
```

### Running Individual Tasks

```python
from instanovo_fm.eval import embedding_io
from instanovo_fm.eval.embed_eval_tasks import get_task

# Load embeddings
embeddings, metadata, faiss_index = embedding_io.load("./evaluation_results")

# Get task class
DuplicateRetrievalTask = get_task("duplicateretrievaltask")

# Run task with custom config
task = DuplicateRetrievalTask(k_values=[1, 5, 10], max_samples=1000)
results = task.run(embeddings, metadata, faiss_index)

print(f"Recall@1: {results['recall_metrics']['recall@1']}")
```

### Batch Evaluation Script

```python
"""Batch evaluate multiple checkpoints."""
from pathlib import Path
from omegaconf import OmegaConf
from instanovo_fm.eval.evaluator import EmbeddingEvaluator

checkpoints = [
    "checkpoints/run1/model_best.ckpt",
    "checkpoints/run2/model_best.ckpt",
    "checkpoints/run3/model_best.ckpt",
]

# Load base config
config = OmegaConf.load("instanovo/configs/evaluation.yaml")
config.evaluation.enabled = True

for ckpt_path in checkpoints:
    print(f"\n{'='*80}")
    print(f"Evaluating: {ckpt_path}")
    print('='*80)
    
    # Update checkpoint path
    config.resume_checkpoint_path = ckpt_path
    
    # Update output directory
    run_name = Path(ckpt_path).parent.name
    config.evaluation.output_dir = f"./evaluation_results/{run_name}"
    
    # Run evaluation
    evaluator = EmbeddingEvaluator(config)
    results = evaluator.evaluate(split="valid")
    
    print(f"✓ {ckpt_path} complete")
```

## Adding New Evaluation Tasks

### 1. Create Task Class

Create a new file in `eval_tasks/`:

```python
# embed_eval_tasks/my_custom_task.py
import numpy as np
from typing import Dict, Any
from instanovo_fm.eval.embed_eval_tasks import BaseTask


class MyCustomTask(BaseTask):
    """My custom evaluation task."""

    name = "My Custom Task"
    description = "Description of what this task does"
    requires_metadata = True  # Does it need metadata?
    requires_faiss = False  # Does it need FAISS index?

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Task-specific config
        self.threshold = kwargs.get('threshold', 0.5)

    def run(self, emb: np.ndarray, meta: Dict[str, np.ndarray],
            faiss_index: Any) -> Dict[str, Any]:
        """Run the task."""
        # Validate inputs
        self.validate_inputs(emb, meta, faiss_index)

        # Your computation here
        result = self._compute_metric(emb, meta)

        return {
            'task_name': self.name,
            'metric': result,
            'num_samples': len(emb),
        }

    def _compute_metric(self, emb, meta):
        # Your logic here
        return 0.0
```

### 2. Register Task

The task is automatically registered via metaclass! Just import it:

```python
# In embed_eval_tasks/__init__.py (already auto-discovered)
from .my_custom_task import MyCustomTask
```

### 3. Use Task

```yaml
# In config
evaluation:
  tasks_to_run:
    - mycustomtask
  task_configs:
    mycustomtask:
      threshold: 0.7
```

## Troubleshooting

### Issue: "No checkpoint specified"

**Solution:** Provide checkpoint path:
```bash
python -m instanovo_fm.eval.embed_evaluation \
    resume_checkpoint_path=checkpoints/model_best.ckpt \
    evaluation.enabled=True
```

### Issue: "h5py not found" or "faiss not found"

**Solution:** Install optional dependencies:
```bash
pip install h5py faiss-cpu
# Or for GPU:
pip install h5py faiss-gpu
```

### Issue: Task fails with missing metadata

**Solution:** Ensure data processor includes metadata:
```yaml
dataset:
  metadata_columns:
    - peptide
    - charge
    - retention_time
    - hyperscore
```

### Issue: Out of memory during embedding generation

**Solution:** Reduce batch size:
```bash
python -m instanovo_fm.eval.embed_evaluation \
    resume_checkpoint_path=model_best.ckpt \
    evaluation.enabled=True \
    evaluation.batch_size=64
```

### Issue: Cached embeddings are stale

**Solution:** Force regeneration:
```bash
python -m instanovo_fm.eval.embed_evaluation \
    resume_checkpoint_path=model_best.ckpt \
    evaluation.enabled=True \
    evaluation.force_regenerate_embeddings=True
```

## Best Practices

1. **Development:** Use `data_subset=0.1` and a small task subset for fast iteration
2. **Validation:** Run all tasks on full validation set periodically
3. **Test Set:** Only evaluate on test set once at the end
4. **Caching:** Let embeddings cache (don't force regenerate unless model changed)
5. **Task Selection:** Start with fast sanity checks (duplicate retrieval) before running expensive tasks
6. **Reproducibility:** Set `evaluation.task_configs.<task>.random_state` for deterministic results

## Performance Tips

- Use `data_subset` for faster testing
- Enable GPU with `evaluation.device=cuda`
- Increase `batch_size` if you have GPU memory
- Run tasks in parallel by manually calling individual tasks
- Cache embeddings across multiple task runs

## Citation

If you use this evaluation framework, please cite:

```bibtex
@software{instanovo_foundation_eval,
  title={InstaNovo Foundation Model Evaluation Framework},
  author={...},
  year={2025},
  url={https://github.com/...}
}
```



