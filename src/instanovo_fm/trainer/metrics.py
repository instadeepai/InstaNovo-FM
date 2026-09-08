"""Metrics computation for foundational pre-training.

This module contains streaming metrics computation for validation,
including MAE in Daltons, cosine similarity, and auxiliary task metrics.
"""

import random
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


class StreamingMetrics:
    """Streaming metrics computation for validation to prevent OOM on large datasets.

    Computes MAE in Daltons, percentage within thresholds,
    and auxiliary task metrics in a memory-efficient streaming fashion.
    """

    def __init__(self) -> None:
        """Initialize streaming metrics accumulators."""
        # Core metrics
        self.running_loss = 0.0
        self.running_mae = 0.0
        self.running_mae_squared = 0.0  # For std calculation

        # Counters
        self.running_counts: dict[str, Any] = {"losses": 0, "tokens": 0, "spectra": 0}

        # Percentage counters
        self.tokens_within_1da = 0
        self.tokens_within_01da = 0

        # PPM percentage counters
        self.tokens_within_10ppm = 0
        self.tokens_within_20ppm = 0

        # Per-spectrum metrics
        self.spectrum_mae_sum = 0.0
        self.spectrum_std_sum = 0.0

        # Limited accumulator for median calculation (reservoir sampling)
        self.running_median_accumulator: list[Any] = []
        self.running_median_ppm_accumulator: list[Any] = []  # For PPM median
        self.max_median_samples = 50000
        self._median_total_seen = 0  # Reservoir sampling index for Da
        self._median_ppm_total_seen = 0  # Reservoir sampling index for PPM

        # Auxiliary metrics (intensity only)
        self.aux_intensity_preds: list[Any] = []  # Regression predictions (continuous)
        self.aux_intensity_targets: list[Any] = []  # Regression targets (continuous)

        # Per-spectrum intensity ranking metrics (annotation-free, capped)
        self.intensity_spearman_scores: list[Any] = []  # Per-spectrum Spearman rank correlations
        self.intensity_topk_recalls: dict[int, Any] = {5: [], 10: []}  # Per-spectrum Top-K recall
        self.max_spearman_spectra = 10_000  # Cap to avoid per-spectrum loop over entire validation set
        self._spearman_spectra_seen = 0

        # Entropy metrics for epistemic uncertainty
        self.group_entropies: list[Any] = []  # Entropy of group predictions
        self.offset_entropies: list[Any] = []  # Entropy of offset predictions
        self.max_entropy_samples = 10000

        # Classifier confidence metrics
        self.conf_group_list: list[Any] = []  # Group confidences
        self.conf_offset_list: list[Any] = []  # Offset confidences
        self.conf_joint_list: list[Any] = []  # Joint confidences
        # Cap matches max_calibration_samples so ECE (group_preds/targets + conf_*)
        # uses the same sample budget — reduces ECE variance (CV) at no training cost.
        self.max_confidence_samples = 50000

        # Calibration metrics (for ECE computation)
        self.group_preds_list: list[Any] = []  # Group predictions
        self.offset_preds_list: list[Any] = []  # Offset predictions
        self.group_targets_list: list[Any] = []  # Group targets
        self.offset_targets_list: list[Any] = []  # Offset targets
        self.max_calibration_samples = 50000  # Cap for ECE accumulator lists

        # Bin accuracy counters (unconditional, streaming)
        self.bin_acc_group_correct = 0
        self.bin_acc_offset_correct = 0
        self.bin_acc_bin_correct = 0
        self.bin_acc_group_top5_correct = 0
        self.bin_acc_offset_top5_correct = 0
        self.bin_acc_offset_pm1_correct = 0  # ±1 bin tolerance
        self.bin_acc_bin_pm1_correct = 0  # group exact + offset ±1
        self.bin_acc_total = 0

        # Bin-only metrics (for delta Da comparison)
        self.bin_only_running_mae = 0.0
        self.bin_only_tokens_within_01da = 0
        self.bin_only_tokens_within_1da = 0
        self.bin_only_tokens_within_10ppm = 0
        self.bin_only_tokens_within_20ppm = 0
        self.bin_only_token_count = 0
        self.bin_only_median_ppm_accumulator: list = []
        self._bin_only_median_ppm_total_seen = 0

    def update_loss(self, loss: float) -> None:
        """Update running loss."""
        self.running_loss = (self.running_loss * self.running_counts["losses"] + loss) / (self.running_counts["losses"] + 1)
        self.running_counts["losses"] += 1

    def update_bin_accuracy(
        self,
        group_logits: torch.Tensor,
        offset_logits: torch.Tensor,
        group_targets: torch.Tensor,
        offset_targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """Update bin accuracy counters over all masked tokens.

        Args:
            group_logits: Group prediction logits (B, L, n_groups)
            offset_logits: Offset prediction logits (B, L, group_size)
            group_targets: Ground truth group indices (B, L)
            offset_targets: Ground truth offset indices (B, L)
            valid_mask: Mask for masked, non-padded positions (B, L)
        """
        # Top-1 predictions
        group_preds = group_logits.argmax(dim=-1)  # (B, L)
        offset_preds = offset_logits.argmax(dim=-1)  # (B, L)

        # Masked positions only
        gp = group_preds[valid_mask]
        gt = group_targets[valid_mask]
        op = offset_preds[valid_mask]
        ot = offset_targets[valid_mask]

        n = gp.numel()
        if n == 0:
            return

        group_correct = gp == gt
        offset_correct = op == ot
        bin_correct = group_correct & offset_correct

        # ±1 bin tolerance (offset within 1 bin of target, group must be exact)
        offset_pm1_correct = (op - ot).abs() <= 1
        bin_pm1_correct = group_correct & offset_pm1_correct

        # Top-5 accuracy
        k = min(5, group_logits.size(-1))
        group_top5 = group_logits[valid_mask].topk(k, dim=-1).indices  # (M, k)
        group_top5_hit = (group_top5 == gt.unsqueeze(-1)).any(dim=-1)

        k = min(5, offset_logits.size(-1))
        offset_top5 = offset_logits[valid_mask].topk(k, dim=-1).indices  # (M, k)
        offset_top5_hit = (offset_top5 == ot.unsqueeze(-1)).any(dim=-1)

        # Batch all sums into a single GPU→CPU transfer (7 values instead of 7 .item() calls)
        counts = torch.stack(
            [
                group_correct.sum(),
                offset_correct.sum(),
                bin_correct.sum(),
                offset_pm1_correct.sum(),
                bin_pm1_correct.sum(),
                group_top5_hit.sum(),
                offset_top5_hit.sum(),
            ]
        ).cpu()

        self.bin_acc_group_correct += int(counts[0])
        self.bin_acc_offset_correct += int(counts[1])
        self.bin_acc_bin_correct += int(counts[2])
        self.bin_acc_offset_pm1_correct += int(counts[3])
        self.bin_acc_bin_pm1_correct += int(counts[4])
        self.bin_acc_group_top5_correct += int(counts[5])
        self.bin_acc_offset_top5_correct += int(counts[6])

        self.bin_acc_total += n

    def update_entropy_metrics(self, group_logits: torch.Tensor, offset_logits: torch.Tensor, valid_mask: torch.Tensor) -> None:
        """Update entropy metrics for epistemic uncertainty estimation.

        Args:
            group_logits: Group prediction logits (B, L, n_groups)
            offset_logits: Offset prediction logits (B, L, group_size)
            valid_mask: Mask indicating valid positions (B, L)
        """
        if len(self.group_entropies) >= self.max_entropy_samples:
            return

        # Get masked positions only
        group_masked = group_logits[valid_mask]  # (M, n_groups)
        offset_masked = offset_logits[valid_mask]  # (M, group_size)

        # Compute entropy for group predictions
        group_probs = F.softmax(group_masked, dim=-1)
        group_entropy = -(group_probs * torch.log(group_probs.clamp(min=1e-12))).sum(dim=-1)

        # Compute entropy for offset predictions
        offset_probs = F.softmax(offset_masked, dim=-1)
        offset_entropy = -(offset_probs * torch.log(offset_probs.clamp(min=1e-12))).sum(dim=-1)

        # Filter non-finite values before storing (MAE decoder can produce
        # occasional extreme logits at decoded positions)
        finite = torch.isfinite(group_entropy) & torch.isfinite(offset_entropy)
        if finite.any():
            self.group_entropies.extend(group_entropy[finite].cpu().float().tolist())
            self.offset_entropies.extend(offset_entropy[finite].cpu().float().tolist())

    def update_confidence_metrics(
        self,
        aux_out: Dict[str, torch.Tensor],
        group_logits: torch.Tensor,
        offset_logits: torch.Tensor,
        group_targets: torch.Tensor,
        offset_targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        """Update classifier confidence metrics.

        Args:
            aux_out: Auxiliary outputs containing confidence metrics
            group_logits: Group logits (B, L, n_groups) for predictions
            offset_logits: Offset logits (B, L, group_size) for predictions
            group_targets: Group targets (B, L)
            offset_targets: Offset targets (B, L)
            valid_mask: Mask indicating valid positions (B, L)
        """
        if len(self.conf_group_list) >= self.max_confidence_samples:
            return

        # Cap calibration accumulator lists
        if self.group_preds_list:
            total_cal = sum(t.numel() for t in self.group_preds_list)
            if total_cal >= self.max_calibration_samples:
                return

        # Extract confidence metrics from aux_out
        if "conf_group" in aux_out:
            conf_group = aux_out["conf_group"][valid_mask]
            self.conf_group_list.extend(conf_group.cpu().float().tolist())

        if "conf_offset" in aux_out:
            conf_offset = aux_out["conf_offset"][valid_mask]
            self.conf_offset_list.extend(conf_offset.cpu().float().tolist())

        if "conf_joint" in aux_out:
            conf_joint = aux_out["conf_joint"][valid_mask]
            self.conf_joint_list.extend(conf_joint.cpu().float().tolist())

        # Collect predictions and targets for ECE computation
        group_preds = group_logits.argmax(dim=-1)[valid_mask]
        offset_preds = offset_logits.argmax(dim=-1)[valid_mask]

        self.group_preds_list.append(group_preds.cpu())
        self.offset_preds_list.append(offset_preds.cpu())
        self.group_targets_list.append(group_targets[valid_mask].cpu())
        self.offset_targets_list.append(offset_targets[valid_mask].cpu())

    def update_mz_metrics(self, pred_mz: torch.Tensor, target_mz: torch.Tensor, valid_mask: torch.Tensor) -> None:
        """Update m/z prediction metrics (vectorized batch implementation).

        Args:
            pred_mz: Predicted m/z values (B, L) in actual m/z units
            target_mz: Target m/z values (B, L) in actual m/z units
            valid_mask: Mask indicating valid positions (B, L)
        """
        batch_size = pred_mz.size(0)

        # Get all valid positions across the batch
        all_valid_pred = pred_mz[valid_mask]  # (N,)
        all_valid_tgt = target_mz[valid_mask]  # (N,)

        if all_valid_pred.numel() == 0:
            self.running_counts["spectra"] += batch_size
            return

        # Compute errors (Da) - all on GPU
        abs_errors = (all_valid_pred - all_valid_tgt).abs()

        # Compute PPM errors
        abs_errors_ppm = (abs_errors / (all_valid_tgt + 1e-8)) * 1e6

        # Batch threshold comparisons (all GPU ops)
        within_1da = abs_errors <= 1.0
        within_01da = abs_errors <= 0.1
        within_10ppm = abs_errors_ppm <= 10.0
        within_20ppm = abs_errors_ppm <= 20.0

        # Single GPU→CPU transfer for all scalar accumulations
        abs_errors_sum = abs_errors.sum().item()
        abs_errors_sq_sum = (abs_errors**2).sum().item()
        count_1da = within_1da.sum().item()
        count_01da = within_01da.sum().item()
        count_10ppm = within_10ppm.sum().item()
        count_20ppm = within_20ppm.sum().item()
        current_tokens = abs_errors.numel()

        # Update streaming MAE
        old_count = self.running_counts["tokens"]
        new_count = old_count + current_tokens
        self.running_mae = (self.running_mae * old_count + abs_errors_sum) / new_count
        self.running_mae_squared = (self.running_mae_squared * old_count + abs_errors_sq_sum) / new_count
        self.running_counts["tokens"] = new_count

        # Update percentage counters
        self.tokens_within_1da += count_1da
        self.tokens_within_01da += count_01da
        self.tokens_within_10ppm += count_10ppm
        self.tokens_within_20ppm += count_20ppm

        # Collect samples for median via reservoir sampling (unbiased across validation set)
        batch_sample_size = min(500, current_tokens)
        if batch_sample_size > 0:
            # Subsample from this batch uniformly (avoids always taking first N)
            perm = torch.randperm(current_tokens, device="cpu")[:batch_sample_size]
            batch_da = abs_errors.cpu()[perm].tolist()
            batch_ppm = abs_errors_ppm.cpu()[perm].tolist()

            self._reservoir_update(
                self.running_median_accumulator,
                self._median_total_seen,
                batch_da,
                self.max_median_samples,
            )
            self._median_total_seen += batch_sample_size

            self._reservoir_update(
                self.running_median_ppm_accumulator,
                self._median_ppm_total_seen,
                batch_ppm,
                self.max_median_samples,
            )
            self._median_ppm_total_seen += batch_sample_size

        # Per-spectrum metrics (vectorized with scatter_add)
        # Compute per-spectrum sums and counts
        spectrum_idx = torch.arange(batch_size, device=pred_mz.device).unsqueeze(1).expand_as(pred_mz)[valid_mask]

        # Use scatter_add for per-spectrum accumulation
        spectrum_sums = torch.zeros(batch_size, device=pred_mz.device)
        spectrum_sums.scatter_add_(0, spectrum_idx, abs_errors)

        spectrum_sq_sums = torch.zeros(batch_size, device=pred_mz.device)
        spectrum_sq_sums.scatter_add_(0, spectrum_idx, abs_errors**2)

        spectrum_counts = torch.zeros(batch_size, device=pred_mz.device)
        spectrum_counts.scatter_add_(0, spectrum_idx, torch.ones_like(abs_errors))

        # Compute per-spectrum means and stds (avoid div by zero)
        valid_spectra = spectrum_counts > 0
        spectrum_means = torch.zeros_like(spectrum_sums)
        spectrum_means[valid_spectra] = spectrum_sums[valid_spectra] / spectrum_counts[valid_spectra]

        spectrum_vars = torch.zeros_like(spectrum_sums)
        multi_token_spectra = spectrum_counts > 1
        if multi_token_spectra.any():
            spectrum_vars[multi_token_spectra] = (
                spectrum_sq_sums[multi_token_spectra] / spectrum_counts[multi_token_spectra] - spectrum_means[multi_token_spectra] ** 2
            ).clamp(min=0.0)
        spectrum_stds = spectrum_vars.sqrt()

        # Single GPU→CPU transfer for per-spectrum stats
        self.spectrum_mae_sum += spectrum_means.sum().item()
        self.spectrum_std_sum += spectrum_stds.sum().item()

        self.running_counts["spectra"] += batch_size

    def update_mz_metrics_bin_only(self, pred_mz_bin: torch.Tensor, target_mz: torch.Tensor, valid_mask: torch.Tensor) -> None:
        """Update bin-only m/z metrics (no delta correction) for comparison.

        Args:
            pred_mz_bin: Bin-center predictions (B, L) in Daltons
            target_mz: Target m/z values (B, L) in Daltons
            valid_mask: Mask indicating valid positions (B, L)
        """
        all_pred = pred_mz_bin[valid_mask]
        all_tgt = target_mz[valid_mask]

        if all_pred.numel() == 0:
            return

        abs_errors = (all_pred - all_tgt).abs()
        abs_errors_ppm = (abs_errors / (all_tgt + 1e-8)) * 1e6

        current_tokens = abs_errors.numel()
        abs_errors_sum = abs_errors.sum().item()

        # Streaming MAE
        old_count = self.bin_only_token_count
        new_count = old_count + current_tokens
        self.bin_only_running_mae = (self.bin_only_running_mae * old_count + abs_errors_sum) / new_count
        self.bin_only_token_count = new_count

        # Threshold counters
        self.bin_only_tokens_within_01da += (abs_errors <= 0.1).sum().item()
        self.bin_only_tokens_within_1da += (abs_errors <= 1.0).sum().item()
        self.bin_only_tokens_within_10ppm += (abs_errors_ppm <= 10.0).sum().item()
        self.bin_only_tokens_within_20ppm += (abs_errors_ppm <= 20.0).sum().item()

        # Reservoir sample for median PPM
        batch_sample_size = min(500, current_tokens)
        if batch_sample_size > 0:
            perm = torch.randperm(current_tokens, device="cpu")[:batch_sample_size]
            batch_ppm = abs_errors_ppm.cpu()[perm].tolist()
            self._reservoir_update(
                self.bin_only_median_ppm_accumulator,
                self._bin_only_median_ppm_total_seen,
                batch_ppm,
                self.max_median_samples,
            )
            self._bin_only_median_ppm_total_seen += batch_sample_size

    def update_auxiliary_metrics(
        self,
        aux_out: Dict[str, Any],
        batch: Dict[str, torch.Tensor],
        aux_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update auxiliary task metrics (intensity only).

        Args:
            aux_out: Auxiliary outputs from model
            batch: Input batch containing targets
        """
        lambda_intensity = aux_config.get("lambda_intensity", 0.0) if aux_config else 0.0

        # Intensity regression: evaluate strictly on masked, non-padding positions
        if lambda_intensity > 0.0 and "intensity" in aux_out and "intensity_labels" in batch:
            inten_preds: torch.Tensor = aux_out["intensity"]  # (B, L, 1)
            inten_labels: torch.Tensor = batch["intensity_labels"]  # (B, L)
            valid_mask = inten_labels != -100
            if "mlm_mask" in batch:
                valid_mask = valid_mask & batch["mlm_mask"].bool()
            if "spectra_mask" in batch:
                valid_mask = valid_mask & (~batch["spectra_mask"].bool())

            if valid_mask.any():
                self.aux_intensity_preds.append(inten_preds.squeeze(-1)[valid_mask])
                self.aux_intensity_targets.append(inten_labels[valid_mask])

                # Per-spectrum Spearman and Top-K (annotation-free ranking metrics)
                # Capped at max_spearman_spectra to avoid O(N) per-spectrum loop over entire validation
                if self._spearman_spectra_seen < self.max_spearman_spectra:
                    inten_preds_2d = inten_preds.squeeze(-1)  # (B, L)
                    batch_size = inten_preds_2d.size(0)
                    n_valid_per_spec = valid_mask.sum(dim=1)  # (B,) — single GPU op
                    n_valid_cpu = n_valid_per_spec.cpu()  # single transfer

                    for i in range(batch_size):
                        if self._spearman_spectra_seen >= self.max_spearman_spectra:
                            break

                        n_valid = int(n_valid_cpu[i].item())
                        if n_valid < 4:
                            continue

                        spec_mask = valid_mask[i]  # (L,)
                        spec_p = inten_preds_2d[i, spec_mask]  # (n_valid,)
                        spec_t = inten_labels[i, spec_mask]  # (n_valid,)

                        # Spearman rank correlation (torch-based, no scipy)
                        spearman = self._spearman_correlation(spec_p, spec_t)
                        if spearman is not None:
                            self.intensity_spearman_scores.append(spearman)

                        # Top-K recall
                        for k in [5, 10]:
                            if n_valid >= 2 * k:
                                recall = self._topk_recall(spec_p, spec_t, k)
                                self.intensity_topk_recalls[k].append(recall)

                        self._spearman_spectra_seen += 1

    def gather_across_ranks(self, accelerator: Any) -> None:
        """Gather scalar metric accumulators across distributed ranks.

        After distributed validation, each rank holds metrics from its subset
        of data.  This method packs SUM counters and weighted-average numerators
        into a single tensor, performs an all-gather + sum, and unpacks back
        into instance attributes so that ``compute_final_metrics()`` produces
        whole-dataset results.

        List-based accumulators (reservoir samples for median, intensity lists,
        entropy, confidence, calibration) are NOT gathered.  They remain
        rank-local because:
        - Reservoir samples are already statistical approximations.
        - Gathering variable-length Python lists across NCCL is complex/slow.
        - Medians from 1/N of data have <1% CV (Tier 1 in reproducibility study).

        When ``num_processes == 1`` this is a no-op.

        Args:
            accelerator: HuggingFace Accelerate ``Accelerator`` instance.
        """
        if accelerator.num_processes <= 1:
            return

        # ---- Pack scalar SUM counters (23 values) ----
        sums = torch.tensor(
            [
                # Core counters
                float(self.running_counts["losses"]),
                float(self.running_counts["tokens"]),
                float(self.running_counts["spectra"]),
                # Threshold counters (Da)
                float(self.tokens_within_1da),
                float(self.tokens_within_01da),
                # Threshold counters (PPM)
                float(self.tokens_within_10ppm),
                float(self.tokens_within_20ppm),
                # Per-spectrum sums
                self.spectrum_mae_sum,
                self.spectrum_std_sum,
                # Bin accuracy counters
                float(self.bin_acc_group_correct),
                float(self.bin_acc_offset_correct),
                float(self.bin_acc_bin_correct),
                float(self.bin_acc_group_top5_correct),
                float(self.bin_acc_offset_top5_correct),
                float(self.bin_acc_offset_pm1_correct),
                float(self.bin_acc_bin_pm1_correct),
                float(self.bin_acc_total),
                # Bin-only counters
                float(self.bin_only_tokens_within_01da),
                float(self.bin_only_tokens_within_1da),
                float(self.bin_only_tokens_within_10ppm),
                float(self.bin_only_tokens_within_20ppm),
                float(self.bin_only_token_count),
            ],
            dtype=torch.float64,
            device=accelerator.device,
        )

        # ---- Pack weighted-average numerators (4 values) ----
        # Convert running averages back to raw sums before gathering:
        #   numerator = running_avg * count
        weighted_numerators = torch.tensor(
            [
                self.running_loss * self.running_counts["losses"],
                self.running_mae * self.running_counts["tokens"],
                self.running_mae_squared * self.running_counts["tokens"],
                self.bin_only_running_mae * self.bin_only_token_count,
            ],
            dtype=torch.float64,
            device=accelerator.device,
        )

        # ---- Single gather call for all 27 scalars ----
        all_scalars = torch.cat([sums, weighted_numerators])
        n_values = all_scalars.shape[0]

        gathered = accelerator.gather_for_metrics(all_scalars)

        # Defensive type-check (return type varies across accelerate versions)
        if not isinstance(gathered, torch.Tensor):
            gathered = torch.tensor(gathered, dtype=torch.float64, device=accelerator.device)

        # Reshape: gather_for_metrics may return (num_proc * N,) or (num_proc, N)
        if gathered.dim() == 1 and gathered.numel() > n_values:
            gathered = gathered.view(-1, n_values)
        elif gathered.dim() == 1:
            # Single process fallback — nothing to sum
            gathered = gathered.unsqueeze(0)

        total = gathered.sum(dim=0)

        # ---- Unpack SUM counters ----
        i = 0
        self.running_counts["losses"] = int(total[i].item())
        i += 1
        self.running_counts["tokens"] = int(total[i].item())
        i += 1
        self.running_counts["spectra"] = int(total[i].item())
        i += 1
        self.tokens_within_1da = int(total[i].item())
        i += 1
        self.tokens_within_01da = int(total[i].item())
        i += 1
        self.tokens_within_10ppm = int(total[i].item())
        i += 1
        self.tokens_within_20ppm = int(total[i].item())
        i += 1
        self.spectrum_mae_sum = total[i].item()
        i += 1
        self.spectrum_std_sum = total[i].item()
        i += 1
        self.bin_acc_group_correct = int(total[i].item())
        i += 1
        self.bin_acc_offset_correct = int(total[i].item())
        i += 1
        self.bin_acc_bin_correct = int(total[i].item())
        i += 1
        self.bin_acc_group_top5_correct = int(total[i].item())
        i += 1
        self.bin_acc_offset_top5_correct = int(total[i].item())
        i += 1
        self.bin_acc_offset_pm1_correct = int(total[i].item())
        i += 1
        self.bin_acc_bin_pm1_correct = int(total[i].item())
        i += 1
        self.bin_acc_total = int(total[i].item())
        i += 1
        self.bin_only_tokens_within_01da = int(total[i].item())
        i += 1
        self.bin_only_tokens_within_1da = int(total[i].item())
        i += 1
        self.bin_only_tokens_within_10ppm = int(total[i].item())
        i += 1
        self.bin_only_tokens_within_20ppm = int(total[i].item())
        i += 1
        self.bin_only_token_count = int(total[i].item())
        i += 1

        # ---- Unpack weighted-average numerators → reconvert to averages ----
        loss_sum = total[i].item()
        i += 1
        mae_sum = total[i].item()
        i += 1
        mae_sq_sum = total[i].item()
        i += 1
        bin_only_mae_sum = total[i].item()
        i += 1

        if self.running_counts["losses"] > 0:
            self.running_loss = loss_sum / self.running_counts["losses"]
        if self.running_counts["tokens"] > 0:
            self.running_mae = mae_sum / self.running_counts["tokens"]
            self.running_mae_squared = mae_sq_sum / self.running_counts["tokens"]
        if self.bin_only_token_count > 0:
            self.bin_only_running_mae = bin_only_mae_sum / self.bin_only_token_count

    def compute_final_metrics(self) -> Dict[str, float]:
        """Compute final metrics from streaming data.

        Returns:
            Dictionary containing all computed metrics
        """
        if self.running_counts["tokens"] == 0:
            return {"loss": float("inf")}

        # Calculate median from accumulated samples
        float(np.median(self.running_median_accumulator)) if self.running_median_accumulator else self.running_mae
        median_ae_ppm = float(np.median(self.running_median_ppm_accumulator)) if self.running_median_ppm_accumulator else 0.0

        # Percentage metrics (Da)
        pct_within_1da = (self.tokens_within_1da / self.running_counts["tokens"]) * 100
        pct_within_01da = (self.tokens_within_01da / self.running_counts["tokens"]) * 100

        # Percentage metrics (PPM)
        pct_within_10ppm = (self.tokens_within_10ppm / self.running_counts["tokens"]) * 100
        pct_within_20ppm = (self.tokens_within_20ppm / self.running_counts["tokens"]) * 100

        # Core metrics
        metrics: dict[str, Any] = {
            "loss": self.running_loss,
            # PPM metrics (prioritized)
            "median_ae_ppm": median_ae_ppm,
            "pct_within_10ppm": pct_within_10ppm,
            "pct_within_20ppm": pct_within_20ppm,
            # Da metrics (kept for comparison)
            "mae_daltons": self.running_mae,
            "pct_within_1da": pct_within_1da,
            "pct_within_01da": pct_within_01da,
            # Other metrics
            "total_spectra": self.running_counts["spectra"],
            "total_masked_tokens": self.running_counts["tokens"],
        }

        # Bin accuracy metrics (unconditional)
        if self.bin_acc_total > 0:
            t = self.bin_acc_total
            metrics["group_accuracy"] = (self.bin_acc_group_correct / t) * 100
            metrics["offset_accuracy"] = (self.bin_acc_offset_correct / t) * 100
            metrics["bin_accuracy"] = (self.bin_acc_bin_correct / t) * 100
            metrics["group_top5_accuracy"] = (self.bin_acc_group_top5_correct / t) * 100
            metrics["offset_top5_accuracy"] = (self.bin_acc_offset_top5_correct / t) * 100
            metrics["offset_pm1_accuracy"] = (self.bin_acc_offset_pm1_correct / t) * 100
            metrics["bin_pm1_accuracy"] = (self.bin_acc_bin_pm1_correct / t) * 100

        # Compute auxiliary metrics if intensity data has been collected
        if self.aux_intensity_targets:
            aux_metrics = self._compute_auxiliary_metrics()
            if aux_metrics:
                metrics.update(aux_metrics)

        # Compute entropy metrics for epistemic uncertainty
        entropy_metrics = self._compute_entropy_metrics()
        metrics.update(entropy_metrics)

        # Compute classifier confidence metrics
        confidence_metrics = self._compute_confidence_metrics()
        metrics.update(confidence_metrics)

        # Bin-only and delta improvement metrics
        if self.bin_only_token_count > 0:
            t = self.bin_only_token_count
            metrics["bin_only_mae_daltons"] = self.bin_only_running_mae
            metrics["bin_only_pct_within_01da"] = (self.bin_only_tokens_within_01da / t) * 100
            metrics["bin_only_pct_within_1da"] = (self.bin_only_tokens_within_1da / t) * 100
            metrics["bin_only_pct_within_10ppm"] = (self.bin_only_tokens_within_10ppm / t) * 100
            metrics["bin_only_pct_within_20ppm"] = (self.bin_only_tokens_within_20ppm / t) * 100
            if self.bin_only_median_ppm_accumulator:
                metrics["bin_only_median_ae_ppm"] = float(np.median(self.bin_only_median_ppm_accumulator))
            # Improvement metrics (positive = delta helped)
            metrics["delta_improvement_mae_da"] = self.bin_only_running_mae - self.running_mae
            metrics["delta_improvement_pct_within_01da"] = metrics["pct_within_01da"] - metrics["bin_only_pct_within_01da"]
            metrics["delta_improvement_pct10ppm"] = metrics["pct_within_10ppm"] - metrics["bin_only_pct_within_10ppm"]

        return metrics

    @staticmethod
    def _reservoir_update(
        accumulator: list,
        total_seen: int,
        new_items: list,
        max_size: int,
    ) -> None:
        """Reservoir sampling update (Vitter's Algorithm R).

        Mutates *accumulator* in-place.  *total_seen* is the count of items
        processed **before** this batch (caller must increment afterwards).
        """
        for i, item in enumerate(new_items):
            idx = total_seen + i  # global index of this item
            if len(accumulator) < max_size:
                accumulator.append(item)
            else:
                j = random.randint(0, idx)
                if j < max_size:
                    accumulator[j] = item

    @staticmethod
    def _spearman_correlation(pred: torch.Tensor, target: torch.Tensor) -> Optional[float]:
        """Compute Spearman rank correlation between two 1D tensors (torch-based, no scipy)."""
        n = pred.numel()
        if n < 2:
            return None

        # Rank via argsort-of-argsort (all on GPU/device, no CPU transfer)
        def _rank(x: torch.Tensor) -> torch.Tensor:
            return x.argsort().argsort().float()

        rank_p = _rank(pred.detach())
        rank_t = _rank(target.detach())

        # Pearson correlation on ranks = Spearman
        rank_p = rank_p - rank_p.mean()
        rank_t = rank_t - rank_t.mean()
        denom = rank_p.norm() * rank_t.norm()
        if denom < 1e-12:
            return None

        corr = (rank_p * rank_t).sum() / denom
        return float(corr.item())

    @staticmethod
    def _topk_recall(pred: torch.Tensor, target: torch.Tensor, k: int) -> float:
        """Compute Top-K recall: fraction of true top-K found in predicted top-K."""
        true_topk = set(target.topk(k).indices.tolist())
        pred_topk = set(pred.topk(k).indices.tolist())
        return len(true_topk & pred_topk) / k

    def _compute_auxiliary_metrics(self) -> Dict[str, float]:
        """Compute auxiliary task metrics from collected predictions (intensity only)."""
        metrics: Dict[str, float] = {}

        # Intensity regression metrics (MAE and R²)
        if self.aux_intensity_preds and self.aux_intensity_targets:
            intensity_preds = torch.cat(self.aux_intensity_preds)
            intensity_targets = torch.cat(self.aux_intensity_targets)

            # Only compute for non-ignore labels
            valid_mask = intensity_targets != -100
            if valid_mask.any():
                preds_valid = intensity_preds[valid_mask].float()
                targets_valid = intensity_targets[valid_mask].float()
                metrics["intensity_mae"] = (preds_valid - targets_valid).abs().mean().item()

                # R² = 1 - SS_res / SS_tot
                ss_res = ((targets_valid - preds_valid) ** 2).sum().item()
                ss_tot = ((targets_valid - targets_valid.mean()) ** 2).sum().item()
                metrics["intensity_r2"] = 1.0 - ss_res / max(ss_tot, 1e-8)

        # Per-spectrum intensity ranking metrics (annotation-free)
        if self.intensity_spearman_scores:
            scores = np.array(self.intensity_spearman_scores)
            metrics["intensity_spearman_mean"] = float(np.mean(scores))

        if self.intensity_topk_recalls[10]:
            recalls = np.array(self.intensity_topk_recalls[10])
            metrics["intensity_top10_recall_mean"] = float(np.mean(recalls))

        return metrics

    def _compute_entropy_metrics(self) -> Dict[str, float]:
        """Compute entropy metrics for epistemic uncertainty estimation.

        Returns only entropy_total_mean.
        """
        if not self.group_entropies or not self.offset_entropies:
            return {}

        group_entropies = np.array(self.group_entropies)
        offset_entropies = np.array(self.offset_entropies)

        # Ensure same length (truncate to minimum)
        min_len = min(len(group_entropies), len(offset_entropies))
        group_entropies = group_entropies[:min_len]
        offset_entropies = offset_entropies[:min_len]

        # Combined entropy (sum of group and offset entropies)
        total_entropies = group_entropies + offset_entropies

        # Filter NaN/inf values that can arise from extreme logits
        # (e.g., MAE decoder producing occasional outlier decoded tokens)
        finite_mask = np.isfinite(total_entropies)
        if not finite_mask.any():
            return {}
        total_entropies = total_entropies[finite_mask]

        return {
            "entropy_total_mean": float(np.mean(total_entropies)),
        }

    def _compute_confidence_metrics(self) -> Dict[str, float]:
        """Compute classifier confidence metrics: ECE_group, ECE_offset, conf_joint_mean."""
        metrics: dict[str, Any] = {}

        if self.conf_joint_list:
            conf_joint = np.array(self.conf_joint_list)
            metrics["conf_joint_mean"] = float(np.mean(conf_joint))

        # Compute ECE if we have predictions and targets
        if self.group_preds_list and self.group_targets_list and self.conf_group_list:
            from instanovo_fm.trainer.calibration import compute_ece

            # Concatenate predictions and targets
            group_preds = torch.cat(self.group_preds_list, dim=0)
            group_targets = torch.cat(self.group_targets_list, dim=0)
            conf_group_tensor = torch.tensor(self.conf_group_list[: len(group_preds)])

            # Compute ECE for group head
            ece_group, _, _, _ = compute_ece(conf_group_tensor, group_preds, group_targets, n_bins=15)
            metrics["ece_group"] = ece_group

        if self.offset_preds_list and self.offset_targets_list and self.conf_offset_list:
            from instanovo_fm.trainer.calibration import compute_ece

            # Concatenate predictions and targets
            offset_preds = torch.cat(self.offset_preds_list, dim=0)
            offset_targets = torch.cat(self.offset_targets_list, dim=0)
            conf_offset_tensor = torch.tensor(self.conf_offset_list[: len(offset_preds)])

            # Compute ECE for offset head
            ece_offset, _, _, _ = compute_ece(conf_offset_tensor, offset_preds, offset_targets, n_bins=15)
            metrics["ece_offset"] = ece_offset

        return metrics

    def reset(self) -> None:
        """Reset all metrics for a new validation run."""
        self.running_loss = 0.0
        self.running_mae = 0.0
        self.running_mae_squared = 0.0
        self.running_counts = {"losses": 0, "tokens": 0, "spectra": 0}
        self.tokens_within_1da = 0
        self.tokens_within_01da = 0
        self.tokens_within_10ppm = 0
        self.tokens_within_20ppm = 0
        self.spectrum_mae_sum = 0.0
        self.spectrum_std_sum = 0.0
        self.running_median_accumulator = []
        self.running_median_ppm_accumulator = []
        self._median_total_seen = 0
        self._median_ppm_total_seen = 0
        self.aux_intensity_preds = []
        self.aux_intensity_targets = []
        self.intensity_spearman_scores = []
        self.intensity_topk_recalls = {5: [], 10: []}
        self._spearman_spectra_seen = 0

        # Reset bin accuracy counters
        self.bin_acc_group_correct = 0
        self.bin_acc_offset_correct = 0
        self.bin_acc_bin_correct = 0
        self.bin_acc_group_top5_correct = 0
        self.bin_acc_offset_top5_correct = 0
        self.bin_acc_offset_pm1_correct = 0
        self.bin_acc_bin_pm1_correct = 0
        self.bin_acc_total = 0

        # Reset entropy metrics
        self.group_entropies = []
        self.offset_entropies = []

        # Reset confidence metrics
        self.conf_group_list = []
        self.conf_offset_list = []
        self.conf_joint_list = []
        self.group_preds_list = []
        self.offset_preds_list = []
        self.group_targets_list = []
        self.offset_targets_list = []

        # Reset bin-only metrics
        self.bin_only_running_mae = 0.0
        self.bin_only_tokens_within_01da = 0
        self.bin_only_tokens_within_1da = 0
        self.bin_only_tokens_within_10ppm = 0
        self.bin_only_tokens_within_20ppm = 0
        self.bin_only_token_count = 0
        self.bin_only_median_ppm_accumulator = []
        self._bin_only_median_ppm_total_seen = 0
