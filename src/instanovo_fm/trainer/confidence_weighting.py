"""Confidence-guided loss weighting for two-stage training.

This module implements relative confidence weighting that adapts to per-spectrum
quality variations. Instead of hard thresholds, it uses percentile-based
normalization to focus on high-confidence peaks within each spectrum.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional, Literal


class ConfidenceWeightingStrategy:
    """Compute adaptive loss weights based on confidence scores.
    
    Key features:
    - Per-spectrum relative confidence (no hard thresholds)
    - Gradual warmup to prevent sudden distribution shifts
    - Combines intensity and confidence weighting
    - Safety bounds to prevent gradient issues
    """
    
    def __init__(
        self,
        use_relative_confidence: bool = True,
        percentile_threshold: float = 50.0,
        normalization_mode: Literal["percentile", "z_score", "min_max"] = "percentile",
        weighting_function: Literal["sigmoid", "linear", "quadratic"] = "sigmoid",
        sigmoid_center: float = 0.0,
        sigmoid_sharpness: float = 2.0,
        intensity_weight_power: float = 2.0,
        confidence_weight_power: float = 1.0,
        blend_factor: float = 0.5,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
    ):
        """Initialize confidence weighting strategy.
        
        Args:
            use_relative_confidence: Normalize confidence per spectrum
            percentile_threshold: Percentile for thresholding (0-100)
            normalization_mode: How to normalize confidence scores
            weighting_function: Function to convert confidence to weights
            sigmoid_center: Center point for sigmoid (after normalization)
            sigmoid_sharpness: Steepness of sigmoid transition
            intensity_weight_power: Exponent for intensity weighting
            confidence_weight_power: Exponent for confidence weighting
            blend_factor: Weight for confidence vs intensity (0=intensity only, 1=confidence only)
            min_weight: Minimum weight value (prevent zero gradients)
            max_weight: Maximum weight value (prevent exploding gradients)
        """
        self.use_relative_confidence = use_relative_confidence
        self.percentile_threshold = percentile_threshold
        self.normalization_mode = normalization_mode
        self.weighting_function = weighting_function
        self.sigmoid_center = sigmoid_center
        self.sigmoid_sharpness = sigmoid_sharpness
        self.intensity_weight_power = intensity_weight_power
        self.confidence_weight_power = confidence_weight_power
        self.blend_factor = blend_factor
        self.min_weight = min_weight
        self.max_weight = max_weight
    
    def normalize_confidence_per_spectrum(
        self,
        confidence: torch.Tensor,  # (B, L)
        valid_mask: torch.Tensor,  # (B, L)
    ) -> torch.Tensor:
        """Normalize confidence scores per spectrum to handle quality variations.
        
        This ensures that high-quality and low-quality spectra are treated fairly.
        For each spectrum, we normalize confidence relative to that spectrum's
        distribution, not the global distribution.
        
        Args:
            confidence: Raw confidence scores (B, L)
            valid_mask: Mask for valid peaks (B, L)
        
        Returns:
            Normalized confidence scores (B, L), centered around 0
        """
        B, L = confidence.shape
        normalized = torch.zeros_like(confidence)
        
        for i in range(B):
            # Get valid confidence values for this spectrum
            valid_conf = confidence[i, valid_mask[i]]
            
            if len(valid_conf) == 0:
                continue
            
            if self.normalization_mode == "percentile":
                # Center at specified percentile (e.g., median = 50th percentile)
                center = torch.quantile(valid_conf, self.percentile_threshold / 100.0)
                # Scale by IQR for robustness to outliers
                q25 = torch.quantile(valid_conf, 0.25)
                q75 = torch.quantile(valid_conf, 0.75)
                scale = (q75 - q25).clamp(min=1e-6)
                normalized[i, valid_mask[i]] = (valid_conf - center) / scale
            
            elif self.normalization_mode == "z_score":
                # Standard z-score normalization
                mean = valid_conf.mean()
                std = valid_conf.std().clamp(min=1e-6)
                normalized[i, valid_mask[i]] = (valid_conf - mean) / std
            
            elif self.normalization_mode == "min_max":
                # Min-max normalization to [0, 1], then center at 0.5
                min_val = valid_conf.min()
                max_val = valid_conf.max()
                range_val = (max_val - min_val).clamp(min=1e-6)
                normalized[i, valid_mask[i]] = (valid_conf - min_val) / range_val - 0.5
        
        return normalized
    
    def apply_weighting_function(
        self,
        normalized_confidence: torch.Tensor,  # (B, L), centered around 0
    ) -> torch.Tensor:
        """Convert normalized confidence to weights using specified function.
        
        Args:
            normalized_confidence: Normalized confidence (centered around 0)
        
        Returns:
            Weights in range [0, 1] (before power and bounds)
        """
        if self.weighting_function == "sigmoid":
            # Sigmoid: smooth transition around center
            # sigmoid(sharpness * (conf - center))
            weights = torch.sigmoid(
                self.sigmoid_sharpness * (normalized_confidence - self.sigmoid_center)
            )
        
        elif self.weighting_function == "linear":
            # Linear: clamp to [0, 1]
            weights = (normalized_confidence + 1.0) / 2.0  # Map [-1, 1] -> [0, 1]
            weights = weights.clamp(0.0, 1.0)
        
        elif self.weighting_function == "quadratic":
            # Quadratic: emphasize high confidence more
            linear_weights = (normalized_confidence + 1.0) / 2.0
            weights = (linear_weights.clamp(0.0, 1.0)) ** 2
        
        else:
            raise ValueError(f"Unknown weighting function: {self.weighting_function}")
        
        return weights
    
    def compute_weights(
        self,
        confidence: torch.Tensor,  # (B, L) - raw confidence scores
        intensity: torch.Tensor,   # (B, L) - peak intensities
        valid_mask: torch.Tensor,  # (B, L) - mask for valid peaks
        warmup_factor: float = 1.0,  # 0.0 = stage1, 1.0 = full stage2
    ) -> torch.Tensor:
        """Compute adaptive loss weights combining confidence and intensity.
        
        Args:
            confidence: Raw confidence scores from model
            intensity: Peak intensities (normalized)
            valid_mask: Boolean mask for valid (non-padded, masked) peaks
            warmup_factor: Gradual transition factor (0=stage1, 1=stage2)
        
        Returns:
            Loss weights (B, L), normalized per spectrum
        """
        # 1. Compute intensity weights (always active)
        intensity_weights = torch.pow(intensity, self.intensity_weight_power)
        intensity_weights = intensity_weights * valid_mask
        
        # 2. Compute confidence weights (only in stage 2)
        if warmup_factor > 0.0 and self.use_relative_confidence:
            # Normalize confidence per spectrum
            normalized_conf = self.normalize_confidence_per_spectrum(confidence, valid_mask)
            
            # Apply weighting function
            conf_weights_raw = self.apply_weighting_function(normalized_conf)
            
            # Apply power
            conf_weights = torch.pow(conf_weights_raw, self.confidence_weight_power)
            conf_weights = conf_weights * valid_mask
        else:
            # Stage 1: no confidence weighting
            conf_weights = torch.ones_like(intensity_weights)
        
        # 3. Blend intensity and confidence weights
        # blend_factor = 0: intensity only
        # blend_factor = 1: confidence only
        # blend_factor = 0.5: equal blend
        blended_weights = (
            (1.0 - self.blend_factor) * intensity_weights +
            self.blend_factor * conf_weights
        )
        
        # 4. Gradual warmup: transition from intensity-only to blended
        final_weights = (
            (1.0 - warmup_factor) * intensity_weights +
            warmup_factor * blended_weights
        )
        
        # 5. Apply safety bounds
        final_weights = final_weights.clamp(self.min_weight, self.max_weight)
        
        # 6. Normalize per spectrum (sum to 1 for each spectrum)
        # This ensures consistent gradient magnitudes across spectra
        per_spectrum_sum = final_weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        final_weights = final_weights / per_spectrum_sum
        
        # 7. Re-apply valid mask (in case normalization affected it)
        final_weights = final_weights * valid_mask
        
        return final_weights
    
    def get_warmup_factor(
        self,
        current_step: int,
        stage2_start_step: int,
        warmup_steps: int,
    ) -> float:
        """Compute warmup factor for gradual transition into stage 2.
        
        Args:
            current_step: Current training step
            stage2_start_step: Step when stage 2 begins
            warmup_steps: Number of steps for warmup
        
        Returns:
            Warmup factor in [0, 1]
        """
        if current_step < stage2_start_step:
            # Stage 1: no confidence weighting
            return 0.0
        
        steps_into_stage2 = current_step - stage2_start_step
        
        if steps_into_stage2 >= warmup_steps:
            # Warmup complete
            return 1.0
        
        # Linear warmup
        return float(steps_into_stage2) / float(warmup_steps)


def create_confidence_weighting_strategy(config: dict) -> ConfidenceWeightingStrategy:
    """Factory function to create confidence weighting strategy from config.
    
    Args:
        config: Configuration dictionary (from foundational.yaml)
    
    Returns:
        ConfidenceWeightingStrategy instance
    """
    conf_config = config.get("confidence_weighting", {})
    
    return ConfidenceWeightingStrategy(
        use_relative_confidence=conf_config.get("use_relative_confidence", True),
        percentile_threshold=conf_config.get("percentile_threshold", 50.0),
        normalization_mode=conf_config.get("normalization_mode", "percentile"),
        weighting_function=conf_config.get("weighting_mode", "sigmoid"),
        sigmoid_center=conf_config.get("sigmoid_center", 0.0),
        sigmoid_sharpness=conf_config.get("sigmoid_sharpness", 2.0),
        intensity_weight_power=conf_config.get("intensity_weight_power", 2.0),
        confidence_weight_power=conf_config.get("confidence_weight_power", 1.0),
        blend_factor=conf_config.get("blend_factor", 0.5),
        min_weight=conf_config.get("min_weight", 0.1),
        max_weight=conf_config.get("max_weight", 10.0),
    )









