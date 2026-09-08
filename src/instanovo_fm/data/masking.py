from typing import Any

import torch

# ============================================================================
# Constants
# ============================================================================

ISOTOPE_MASS_SHIFT = 1.003355  # 13C - 12C mass difference (Da)
ISOTOPE_DISTANCE_WINDOW = 5.0  # Maximum m/z distance for span-aware isotope detection (Da)


# ============================================================================
# Helper Functions
# ============================================================================


@torch.no_grad()
def _compute_isotope_shift_match(
    candidate_mz: torch.Tensor,  # (C,)
    masked_mz: torch.Tensor,  # (M,)
    charges: torch.Tensor | None,
    ppm: float,
    da_floor: float,
    max_charge: int,
    max_order: int,
    device: torch.device,
) -> torch.Tensor:
    """Check which candidate positions are isotopes of masked peaks (OPTIMIZED).

    Uses adaptive isotopic order based on median m/z of masked peaks:
    - Conservative formula: max_order ≈ m/z / 600, clamped to [1, 3]

    Note on bidirectional checking (lines below):
    We check both forward (candidate > masked) and backward (candidate < masked)
    isotope offsets. This is intentional for masking purposes: Thompson sampling
    may select any peak in an isotope envelope as an anchor. If an M+1 isotope
    is selected, we need backward checking to find and mask the M+0 monoisotopic
    peak. Without this, the most informative peak could remain visible, making
    the reconstruction task trivially easy.

    Returns:
    -------
    is_isotope : (C,) bool tensor, True for candidates that are isotopes
    """
    if candidate_mz.numel() == 0 or masked_mz.numel() == 0:
        return torch.zeros(candidate_mz.numel(), dtype=torch.bool, device=device)

    # Calculate tolerances (vectorized)
    tol_ppm = (ppm * 1e-6) * candidate_mz
    tolerance = torch.maximum(tol_ppm, torch.tensor(da_floor, device=device))

    # Compute pairwise differences
    diff = candidate_mz.unsqueeze(1) - masked_mz.unsqueeze(0)  # (C, M)

    # Determine charge states to check.
    # Fragment ions can have any charge from +1 up to the precursor charge.
    # For a z=3 precursor, fragments can be z=1, z=2, or z=3, each with
    # different isotope spacing (1.003/z Da). We must check all possibilities.
    if charges is not None:
        max_frag_charge = int(charges.item())
        charge_states = torch.arange(1, max_frag_charge + 1, device=device, dtype=torch.float32)
    else:
        charge_states = torch.arange(1, max_charge + 1, device=device, dtype=torch.float32)

    # Determine effective max_order (adaptive)
    if masked_mz.numel() > 0:
        median_mz = torch.median(masked_mz).item()
        effective_max_order = int(min(3, max(1, round(median_mz / 600))))
    else:
        effective_max_order = max_order

    # Ensure we check at least up to max_order for small m/z values
    effective_max_order = max(effective_max_order, max_order)

    # Vectorized isotope checking
    is_isotope = torch.zeros(candidate_mz.numel(), dtype=torch.bool, device=device)

    # Check all charge states and isotopic orders
    for z in charge_states:
        z_val = z.item()
        for n in range(1, effective_max_order + 1):
            shift = (n * ISOTOPE_MASS_SHIFT) / z_val
            # Bidirectional check: see docstring for rationale.
            # Forward: candidate is an isotope of masked (candidate ≈ masked + shift)
            # Backward: masked is an isotope of candidate (candidate ≈ masked - shift)
            within_tol_forward = torch.abs(diff - shift) <= tolerance.unsqueeze(1)
            within_tol_backward = torch.abs(diff + shift) <= tolerance.unsqueeze(1)
            is_isotope |= within_tol_forward.any(dim=1) | within_tol_backward.any(dim=1)

    return is_isotope


@torch.no_grad()
def _identify_contiguous_spans(span_mask: torch.Tensor, intensity: torch.Tensor, device: torch.device) -> Any:
    """Identify all contiguous spans in a 1D mask (OPTIMIZED).

    Parameters
    ----------
    span_mask : (L,) bool tensor
    intensity : (L,) intensity tensor
    device : torch device

    Returns:
    -------
    spans : list of dicts with keys 'start', 'positions', 'length', 'intensity'
    """
    L = span_mask.shape[0]  # noqa: N806
    masked_indices = torch.nonzero(span_mask, as_tuple=False).squeeze(-1)

    if masked_indices.numel() == 0:
        return []

    # Find span boundaries (vectorized)
    if masked_indices.numel() > 1:
        diffs = masked_indices[1:] - masked_indices[:-1]
        span_starts_mask = torch.cat([torch.tensor([True], device=device), diffs > 1])
        span_start_indices = masked_indices[span_starts_mask]
    else:
        span_start_indices = masked_indices

    # Vectorized span identification
    spans = []
    for start_idx in span_start_indices:
        start_pos = int(start_idx.item())

        # Find span end efficiently using vectorized operations
        # Create a mask for positions >= start_pos
        pos_range = torch.arange(start_pos, L, device=device)
        valid_positions = pos_range[span_mask[start_pos:]]

        if valid_positions.numel() == 0:
            continue

        # Find where the span breaks (gap > 1)
        if valid_positions.numel() > 1:
            gaps = valid_positions[1:] - valid_positions[:-1]
            break_points = torch.nonzero(gaps > 1, as_tuple=False).squeeze(-1)

            if break_points.numel() > 0:
                # Span ends at first break point
                valid_positions[break_points[0]].item()
                span_positions = valid_positions[: break_points[0] + 1].tolist()
            else:
                # No breaks, span continues to end
                span_positions = valid_positions.tolist()
        else:
            span_positions = valid_positions.tolist()

        span_length = len(span_positions)
        if span_length == 0:
            continue

        # Vectorized intensity calculation
        span_intensity = intensity[torch.tensor(span_positions, device=device)].mean().item()

        spans.append(
            {
                "start": start_pos,
                "positions": span_positions,
                "length": span_length,
                "intensity": span_intensity,
            }
        )

    return spans


@torch.no_grad()
def _vectorized_isotope_extension(
    span_mask: torch.Tensor,  # (B, L)
    mz: torch.Tensor,  # (B, L)
    intensity: torch.Tensor,  # (B, L)
    valid: torch.Tensor,  # (B, L)
    charges: torch.Tensor | None,  # (B,)
    max_total_mask_ratio: float,
    isotope_ppm: float,
    isotope_da_floor: float,
    isotope_max_charge: int,
    isotope_max_order: int,
    device: torch.device,
) -> torch.Tensor:
    """Fully batched isotope extension for span borders.

    Operates on all B spectra simultaneously using shifted tensor comparisons.
    Since peaks are m/z-sorted, border candidates are the immediate left/right
    neighbours of masked spans. For each border, we check whether the m/z
    difference to the adjacent masked peak matches any isotope shift
    (n * 1.003355 / z) within the configured tolerance.

    Runs ``isotope_max_order`` rounds so that full isotope envelopes (M+1,
    M+2, M+3) are captured layer by layer.
    """
    B, L = span_mask.shape  # noqa: N806
    result_mask = span_mask.clone()

    # Pre-compute all isotope shifts for every (charge, order) combination.
    # Using the global max charge is safe: shifts for charges above the
    # precursor charge simply won't match any real isotope spacing.
    max_z = int(charges.max().item()) if charges is not None else isotope_max_charge
    shifts = []
    for z in range(1, max_z + 1):
        for n in range(1, isotope_max_order + 1):
            shifts.append((n * ISOTOPE_MASS_SHIFT) / z)
    shifts = torch.tensor(shifts, device=device, dtype=mz.dtype)  # (S,)

    # Per-position tolerance: max(ppm * mz * 1e-6, da_floor)
    tol = torch.maximum(
        isotope_ppm * 1e-6 * mz,
        torch.tensor(isotope_da_floor, device=device, dtype=mz.dtype),
    )  # (B, L)

    # Per-batch budget
    valid_counts = valid.sum(dim=1)  # (B,)
    max_allowed = (valid_counts.float() * max_total_mask_ratio).long()  # (B,)

    for _ in range(isotope_max_order):
        # Check budget
        has_budget = result_mask.sum(dim=1) < max_allowed  # (B,)
        if not has_budget.any():
            break

        # Left borders: unmasked[i] & valid[i] & masked[i+1]
        masked_right = torch.zeros_like(result_mask)
        masked_right[:, :-1] = result_mask[:, 1:]
        left_borders = ~result_mask & valid & masked_right  # (B, L)

        # Right borders: unmasked[i] & valid[i] & masked[i-1]
        masked_left = torch.zeros_like(result_mask)
        masked_left[:, 1:] = result_mask[:, :-1]
        right_borders = ~result_mask & valid & masked_left  # (B, L)

        # For left borders, the adjacent masked peak is at i+1
        ref_mz_left = torch.zeros_like(mz)
        ref_mz_left[:, :-1] = mz[:, 1:]
        diff_left = torch.abs(mz - ref_mz_left)  # (B, L)

        # For right borders, the adjacent masked peak is at i-1
        ref_mz_right = torch.zeros_like(mz)
        ref_mz_right[:, 1:] = mz[:, :-1]
        diff_right = torch.abs(mz - ref_mz_right)  # (B, L)

        # Check if diff matches any isotope shift within tolerance
        # (B, L, 1) vs (S,) -> (B, L, S) -> any over S -> (B, L)
        match_left = (torch.abs(diff_left.unsqueeze(-1) - shifts) <= tol.unsqueeze(-1)).any(dim=-1)
        match_right = (torch.abs(diff_right.unsqueeze(-1) - shifts) <= tol.unsqueeze(-1)).any(dim=-1)

        new_isotopes = (left_borders & match_left) | (right_borders & match_right)
        new_isotopes = new_isotopes & has_budget.unsqueeze(1)

        if not new_isotopes.any():
            break

        result_mask = result_mask | new_isotopes

        # Enforce per-batch budget cap: remove lowest-intensity new isotopes
        current_counts = result_mask.sum(dim=1)
        exceeds = current_counts > max_allowed
        if exceeds.any():
            for b in torch.nonzero(exceeds, as_tuple=False).squeeze(-1):
                bi = b.item()
                excess = (current_counts[bi] - max_allowed[bi]).item()
                new_pos = torch.nonzero(new_isotopes[bi], as_tuple=False).squeeze(-1)
                if new_pos.numel() > 0:
                    ints = intensity[bi, new_pos]
                    _, remove_idx = torch.topk(ints, min(excess, new_pos.numel()), largest=False)
                    result_mask[bi, new_pos[remove_idx]] = False

    return result_mask


@torch.no_grad()
def find_isotopic_neighbors(
    mz: torch.Tensor,
    masked_peaks: torch.Tensor,
    valid_mask: torch.Tensor,
    charges: torch.Tensor | None = None,
    ppm: float = 25.0,
    da_floor: float = 0.02,
    max_charge: int = 3,
    max_order: int = 2,
) -> torch.Tensor:
    """Find isotopic neighbors of masked peaks (span-aware, conservative).

    DEPRECATED: This function is primarily for testing/analysis.
    For production use, `thompson_sampling_span_mask` with `include_isotopes=True`
    provides more conservative, span-aware isotope detection.

    Parameters
    ----------
    mz : (B, L) m/z values
    masked_peaks : (B, L) bool, True for masked positions
    valid_mask : (B, L) bool, True for valid (non-padded) positions
    charges : (B,) charge states per spectrum (optional)
    ppm : ppm tolerance for isotopic detection
    da_floor : minimum DA tolerance floor
    max_charge : maximum charge state to consider (fallback)
    max_order : maximum isotopic order (adaptive formula used)

    Returns:
    -------
    isotope_mask : (B, L) bool, True for isotopic neighbors of masked peaks
    """
    B, L = mz.shape  # noqa: N806
    device = mz.device
    isotope_mask = torch.zeros_like(valid_mask, dtype=torch.bool)

    for b in range(B):
        masked_pos = torch.nonzero(masked_peaks[b] & valid_mask[b], as_tuple=False).squeeze(-1)
        candidate_pos = torch.nonzero((~masked_peaks[b]) & valid_mask[b], as_tuple=False).squeeze(-1)

        if masked_pos.numel() == 0 or candidate_pos.numel() == 0:
            continue

        masked_mz = mz[b, masked_pos]
        candidate_mz = mz[b, candidate_pos]

        # SPAN-AWARE: Only check candidates near masked peaks
        mz_distances = torch.abs(candidate_mz.unsqueeze(1) - masked_mz.unsqueeze(0))
        nearby_mask = mz_distances <= ISOTOPE_DISTANCE_WINDOW

        charge_b = charges[b] if charges is not None else None
        is_isotope = torch.zeros(candidate_pos.numel(), dtype=torch.bool, device=device)

        for c_idx in range(candidate_pos.numel()):
            nearby_masked_indices = nearby_mask[c_idx]

            if not nearby_masked_indices.any():
                continue

            nearby_masked_mz = masked_mz[nearby_masked_indices]

            is_iso = _compute_isotope_shift_match(
                candidate_mz=candidate_mz[c_idx : c_idx + 1],
                masked_mz=nearby_masked_mz,
                charges=charge_b,
                ppm=ppm,
                da_floor=da_floor,
                max_charge=max_charge,
                max_order=max_order,
                device=device,
            )

            is_isotope[c_idx] = is_iso.any()

        isotope_mask[b, candidate_pos[is_isotope]] = True

    return isotope_mask


# ============================================================================
# Core Masking Strategies
# ============================================================================


@torch.no_grad()
def thompson_sampling_mask(
    intensity: torch.Tensor,
    spectra_mask: torch.Tensor,
    mask_portion: float = 0.30,
    alpha: float = 0.5,
    beta: float = 0.5,
    kappa: float = 4.0,
    gamma: float = 0.7,
    use_topk: bool = True,
) -> torch.Tensor:
    """Thompson-sampling peak mask (vectorized, intensity-aware).

    Samples peaks based on Beta distribution parameterized by intensity,
    favoring high-intensity peaks while maintaining stochasticity.

    Parameters
    ----------
    intensity : (B, L) intensity values in [0,1]
    spectra_mask : (B, L) bool, True for PAD positions
    mask_portion : fraction of valid tokens to mask
    alpha, beta : Beta distribution base parameters
    kappa : concentration scaling for intensity weighting
    gamma : intensity tempering exponent (I^gamma)
    use_topk : use topk (faster) vs full argsort

    Returns:
    -------
    mask : (B, L) bool, True where masked
    """
    B, L = intensity.shape  # noqa: N806
    device = intensity.device

    valid = ~spectra_mask
    lengths = valid.sum(dim=-1)
    num_to_sample = (lengths.float() * mask_portion).round().long()
    num_to_sample.clamp_(min=1)
    num_to_sample = torch.minimum(num_to_sample, (lengths - 1).clamp(min=1))

    # Beta parameters with tempering & concentration
    eps = 1e-6
    I = intensity.clamp(eps, 1.0 - eps).pow(gamma)  # noqa: E741, N806
    a = alpha + kappa * I
    b = beta + kappa * (1.0 - I)

    # Sample Thompson scores
    scores = torch.distributions.Beta(a, b).sample()
    scores = scores + torch.rand_like(scores) * 1e-4  # tie-breaker
    scores.masked_fill_(~valid, -torch.inf)

    # Select top-k per row
    if use_topk:
        k_max = max(int(num_to_sample.max().item()), 1)
        topk_vals, topk_idx = torch.topk(scores, k=k_max, dim=-1, largest=True, sorted=False)
        rank = torch.arange(k_max, device=device).expand(B, k_max)
        take = rank < num_to_sample.unsqueeze(1)
        mask = torch.zeros_like(valid, dtype=torch.bool)
        mask.scatter_(dim=1, index=topk_idx, src=take)
    else:
        order = torch.argsort(scores, dim=-1, descending=True)
        rank = torch.arange(L, device=device).expand(B, L)
        take = rank < num_to_sample.unsqueeze(1)
        mask = torch.zeros_like(valid, dtype=torch.bool)
        mask.scatter_(dim=1, index=order, src=take)

    return mask


@torch.no_grad()
def uniform_random_mask(
    intensity: torch.Tensor,
    spectra_mask: torch.Tensor,
    mask_portion: float = 0.30,
) -> torch.Tensor:
    """Uniform random masking (vectorized, intensity-agnostic baseline).

    Parameters
    ----------
    intensity : (B, L) intensity values (not used, for API consistency)
    spectra_mask : (B, L) bool, True for PAD positions
    mask_portion : fraction of valid tokens to mask

    Returns:
    -------
    mask : (B, L) bool, True where masked
    """
    B, L = intensity.shape  # noqa: N806
    device = intensity.device

    valid = ~spectra_mask
    lengths = valid.sum(dim=-1)
    num_to_sample = (lengths.float() * mask_portion).round().long()
    num_to_sample.clamp_(min=1)
    num_to_sample = torch.minimum(num_to_sample, (lengths - 1).clamp(min=1))

    scores = torch.rand(B, L, device=device, dtype=intensity.dtype)
    scores.masked_fill_(~valid, -torch.inf)

    order = torch.argsort(scores, dim=-1, descending=True)
    rank = torch.arange(L, device=device).expand(B, L)
    take = rank < num_to_sample.unsqueeze(1)
    mask = torch.zeros_like(valid, dtype=torch.bool)
    mask.scatter_(1, order, take)

    return mask


@torch.no_grad()
def thompson_sampling_span_mask(
    intensity: torch.Tensor,
    spectra_mask: torch.Tensor,
    mask_portion: float = 0.30,
    span_min: int = 4,
    span_max: int = 7,
    alpha: float = 0.5,
    beta: float = 0.5,
    kappa: float = 4.0,
    gamma: float = 0.7,
    oversample_factor: float = 1.5,
    bidirectional: bool = True,
    mz: torch.Tensor | None = None,
    charges: torch.Tensor | None = None,
    include_isotopes: bool = False,
    isotope_ppm: float = 25.0,
    isotope_da_floor: float = 0.02,
    isotope_max_charge: int = 3,
    isotope_max_order: int = 2,
    max_total_mask_ratio: float = 0.40,
    max_mz: float = 2500.0,
    normalize_mz: bool = True,
) -> torch.Tensor:
    """Intensity-aware span masking for contiguous peak sequences.

    Samples anchor peaks using Thompson sampling, expands them into spans,
    and optionally extends with isotopic neighbors. Ensures all masked peaks
    form continuous blocks in m/z-sorted space.

    Parameters
    ----------
    intensity : (B, L) intensity values in [0,1]
    spectra_mask : (B, L) bool, True for PAD positions
    mask_portion : target fraction of peaks to mask
    span_min, span_max : span length range
    alpha, beta : Beta distribution parameters
    kappa : concentration scaling for intensity weighting
    gamma : intensity tempering exponent
    oversample_factor : anchor oversampling multiplier
    bidirectional : center spans on anchors vs start at anchors
    mz : (B, L) m/z values
    charges : (B,) precursor charges
    include_isotopes : extend spans with isotope borders
    isotope_ppm : ppm tolerance for isotope detection
    isotope_da_floor : minimum Da tolerance
    isotope_max_charge : max charge state to consider
    isotope_max_order : max isotopic order
    max_total_mask_ratio : hard cap on masking ratio

    Returns:
    -------
    mask : (B, L) bool, True where masked
    """
    B, L = intensity.shape  # noqa: N806
    device = intensity.device
    valid = ~spectra_mask

    # Always sort by m/z for consistent span masking, then unsort at end
    if mz is not None:
        mz_for_sorting = torch.where(spectra_mask, torch.tensor(float("inf"), device=device, dtype=mz.dtype), mz)
        sort_indices = torch.argsort(mz_for_sorting, dim=1)
        unsort_indices = torch.argsort(sort_indices, dim=1)

        intensity_for_masking = torch.gather(intensity, 1, sort_indices)
        mz_for_isotopes = torch.gather(mz, 1, sort_indices)
        spectra_mask_sorted = torch.gather(spectra_mask, 1, sort_indices)
    else:
        intensity_for_masking = intensity
        mz_for_isotopes = mz
        spectra_mask_sorted = spectra_mask
        unsort_indices = None

    # Estimate number of anchors needed
    exp_span_len = 0.5 * (span_min + span_max)
    anchor_portion = (mask_portion / max(exp_span_len, 1.0)) * oversample_factor
    anchor_portion = float(max(0.01, min(0.99, anchor_portion)))

    # Sample anchors using Thompson sampling
    anchors = thompson_sampling_mask(
        intensity=intensity_for_masking,
        spectra_mask=spectra_mask_sorted,
        mask_portion=anchor_portion,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        gamma=gamma,
        use_topk=True,
    )

    span_mask = torch.zeros_like(valid, dtype=torch.bool)

    num_anchors = int(anchors.sum().item())
    if num_anchors == 0:
        return span_mask

    span_lengths = torch.randint(low=span_min, high=span_max + 1, size=(num_anchors,), device=device, dtype=torch.long)

    anchor_batch_idx, anchor_pos_idx = torch.nonzero(anchors, as_tuple=True)

    # Expand spans around anchors
    if bidirectional:
        left_lengths = span_lengths // 2
        right_lengths = span_lengths - left_lengths
        span_starts = (anchor_pos_idx - left_lengths).clamp(min=0)
        span_ends = (anchor_pos_idx + right_lengths).clamp(max=L)
    else:
        span_starts = anchor_pos_idx
        span_ends = (anchor_pos_idx + span_lengths).clamp(max=L)

    # Apply spans using vectorized operations
    max_span_len = int(span_lengths.max().item())
    offset_range = torch.arange(max_span_len, device=device)
    all_positions = span_starts.unsqueeze(1) + offset_range.unsqueeze(0)
    span_valid_mask = (offset_range.unsqueeze(0) < span_lengths.unsqueeze(1)) & (all_positions < span_ends.unsqueeze(1))
    all_positions = all_positions.clamp(max=L - 1)
    batch_indices = anchor_batch_idx.unsqueeze(1).expand(-1, max_span_len)
    valid_span_mask = span_valid_mask & valid[batch_indices, all_positions]
    span_mask[batch_indices[valid_span_mask], all_positions[valid_span_mask]] = True

    # Vectorized trimming: compute valid counts and targets for all batches
    valid_counts = valid.sum(dim=1)
    current_masked_counts = span_mask.sum(dim=1)
    target_masked_counts = (valid_counts.float() * mask_portion).long()
    needs_trimming = (current_masked_counts > target_masked_counts) & (valid_counts >= span_min)

    # Per-batch trimming (only for batches that need it)
    for b in torch.nonzero(needs_trimming, as_tuple=False).squeeze(-1):
        b_idx = b.item()
        excess = (current_masked_counts[b_idx] - target_masked_counts[b_idx]).item()

        spans = _identify_contiguous_spans(span_mask[b_idx], intensity_for_masking[b_idx], device)
        if not spans:
            continue

        spans.sort(key=lambda s: (s["length"], s["intensity"]))

        removed_count = 0
        for span in spans:
            if removed_count >= excess:
                break

            span_positions = torch.tensor(span["positions"], device=device, dtype=torch.long)
            span_mask[b_idx, span_positions] = False
            removed_count += span["length"]

    # Isotopic co-masking
    if include_isotopes and mz_for_isotopes is not None:
        pre_isotope_mask = span_mask.clone()

        # Denormalize m/z to Daltons for isotope shift matching
        mz_da = mz_for_isotopes * max_mz if normalize_mz else mz_for_isotopes

        span_mask = _vectorized_isotope_extension(
            span_mask=span_mask,
            mz=mz_da,
            intensity=intensity_for_masking,
            valid=valid,
            charges=charges,
            max_total_mask_ratio=max_total_mask_ratio,
            isotope_ppm=isotope_ppm,
            isotope_da_floor=isotope_da_floor,
            isotope_max_charge=isotope_max_charge,
            isotope_max_order=isotope_max_order,
            device=device,
        )

        # Hard cap enforcement: protect isotope peaks by removing span peaks first
        isotope_added = span_mask & ~pre_isotope_mask  # peaks added by isotope extension
        valid_counts = valid.sum(dim=1)
        current_masked_counts = span_mask.sum(dim=1)
        hard_caps = (valid_counts.float() * max_total_mask_ratio).long()
        exceeds_cap = (current_masked_counts > hard_caps) & (valid_counts > 0)

        for b in torch.nonzero(exceeds_cap, as_tuple=False).squeeze(-1):
            b_idx = b.item()
            excess = (current_masked_counts[b_idx] - hard_caps[b_idx]).item()
            masked_positions = torch.nonzero(span_mask[b_idx], as_tuple=False).squeeze(-1)

            # Sort by: isotope flag ascending (remove span peaks first), then intensity ascending
            is_isotope = isotope_added[b_idx, masked_positions].float()
            intensities = intensity_for_masking[b_idx, masked_positions]
            removal_priority = is_isotope * 1e6 + intensities  # isotope peaks sort last
            sorted_by_priority = torch.argsort(removal_priority)
            positions_to_remove = masked_positions[sorted_by_priority[:excess]]

            span_mask[b_idx, positions_to_remove] = False

    # Unsort mask back to original peak order
    if unsort_indices is not None:
        span_mask = torch.gather(span_mask, 1, unsort_indices)

    return span_mask


# ============================================================================
# Signal-Aware Fragment Masking
# ============================================================================


def _annotate_single_spectrum_worker(args: Any) -> Any:
    """Worker function for parallel annotation via multiprocessing.

    Parameters
    ----------
    args : tuple
        (mz_np, intensity_np, peptide, charge, ppm_tol, ion_types, da_tol, frag_type)

    Returns:
    -------
    dict or None
        Annotation result from match_with_conditional_features, or None if failed
    """
    mz_np, intensity_np, peptide, charge, ppm_tol, ion_types, da_tol, frag_type = args

    if peptide is None or len(mz_np) == 0:
        return None

    try:
        from instanovo_fm.utils.theoretical_spectra import match_with_conditional_features

        result = match_with_conditional_features(
            exp_mz=mz_np,
            exp_intensity=intensity_np,
            peptide=peptide,
            precursor_charge=int(charge),
            ppm_tol=ppm_tol,
            da_tol=da_tol,
            ion_types=ion_types,
            add_losses=True,
            loss_types=("H2O", "NH3", "SO3", "H3PO4"),
            add_isotopes=True,
            max_isotope=4,
            add_precursor=False,  # Don't mask precursor
            use_closest=False,
            engine="pyopenms",
            fragmentation_type=frag_type,
        )
        return result
    except Exception:
        return None


def _expand_annotation_to_full_length(result: Any, valid_mask: Any, L: Any) -> Any:  # noqa: N803
    """Expand annotation results from valid peaks to full tensor length.

    Parameters
    ----------
    result : dict
        Annotation result with arrays sized to number of valid peaks
    valid_mask : np.ndarray
        Boolean mask [L] indicating valid (non-padded) positions
    L : int
        Full tensor length

    Returns:
    -------
    dict
        Annotation result with arrays expanded to length L
    """
    import numpy as np

    expanded = {
        "feature_type": [None] * L,
        "parent_annotation": [None] * L,
        "matched_annotation": [None] * L,
        "mask": np.zeros(L, dtype=bool),
        "metrics": result["metrics"],
    }

    valid_indices = np.where(valid_mask)[0]
    for i, valid_idx in enumerate(valid_indices):
        expanded["feature_type"][valid_idx] = result["feature_type"][i]
        expanded["parent_annotation"][valid_idx] = result["parent_annotation"][i]
        expanded["matched_annotation"][valid_idx] = result["matched_annotation"][i]
        expanded["mask"][valid_idx] = result["mask"][i]

    return expanded


def _build_parent_child_mapping(feature_types: Any, parent_annotations: Any, matched_annotations: Any, intensity: Any, spectra_mask: Any) -> Any:
    """Build parent-child mapping from annotation results.

    Parameters
    ----------
    feature_types : list
        Per-peak feature type: "base", "loss", "isotope", "precursor", or None
    parent_annotations : list
        Per-peak parent annotation (for losses/isotopes)
    matched_annotations : list
        Per-peak matched annotation
    intensity : torch.Tensor
        Peak intensities [L]
    spectra_mask : torch.Tensor
        Padding mask [L], True for padded positions

    Returns:
    -------
    dict
        Mapping from parent annotation string to dict with:
        - parent_idx: int
        - parent_intensity: float
        - loss_indices: list of int
        - isotope_indices: list of int
    """
    L = len(feature_types)  # noqa: N806
    parent_groups = {}

    # Pass 1: Identify all base fragment ions (parents)
    for idx in range(L):
        if spectra_mask[idx]:  # Skip padded positions
            continue

        ftype = feature_types[idx]
        annotation = matched_annotations[idx]

        if ftype == "base" and annotation:
            parent_groups[annotation] = {
                "parent_idx": idx,
                "parent_intensity": intensity[idx].item(),
                "loss_indices": [],
                "isotope_indices": [],
            }

    # Pass 2: Assign children to parents
    for idx in range(L):
        if spectra_mask[idx]:
            continue

        ftype = feature_types[idx]
        parent_ann = parent_annotations[idx]

        if ftype == "loss" and parent_ann in parent_groups:
            parent_groups[parent_ann]["loss_indices"].append(idx)
        elif ftype == "isotope" and parent_ann in parent_groups:
            parent_groups[parent_ann]["isotope_indices"].append(idx)

    return parent_groups


def _select_fragment_ions_uniform(parent_groups: Any, mask_portion: Any, n_valid: Any, max_ratio: Any) -> Any:
    """Select fragment ions uniformly at random.

    Parameters
    ----------
    parent_groups : dict
        Parent-child mapping from _build_parent_child_mapping
    mask_portion : float
        Fraction of base fragments to select
    n_valid : int
        Number of valid (non-padded) peaks
    max_ratio : float
        Maximum fraction of total peaks that can be masked

    Returns:
    -------
    list
        List of selected parent annotation keys
    """
    import random

    if len(parent_groups) == 0:
        return []

    # Calculate target number of fragments
    target_fragments = max(1, int(len(parent_groups) * mask_portion))

    # Shuffle parent keys for random selection
    parent_keys = list(parent_groups.keys())
    random.shuffle(parent_keys)

    # Greedily select until target or cap reached
    selected = []
    total_masked = 0
    max_allowed = int(n_valid * max_ratio)

    for key in parent_keys:
        group = parent_groups[key]
        group_size = 1 + len(group["loss_indices"]) + len(group["isotope_indices"])

        # Check if we can fit this group
        if total_masked + group_size <= max_allowed:
            selected.append(key)
            total_masked += group_size

            # Check if we've selected enough fragments
            if len(selected) >= target_fragments:
                break
        else:
            # Can't fit this group - check if we're close enough to target
            if len(selected) >= target_fragments * 0.8:  # Within 80% of target
                break

    return selected


def _enforce_mask_cap(batch_mask: Any, spectra_mask: Any, max_ratio: Any, intensity: Any) -> Any:
    """Enforce hard cap on masking ratio by removing lowest-intensity peaks.

    Parameters
    ----------
    batch_mask : torch.Tensor
        Boolean mask [L] indicating masked positions
    spectra_mask : torch.Tensor
        Boolean mask [L] indicating padded positions
    max_ratio : float
        Maximum fraction of valid peaks that can be masked
    intensity : torch.Tensor
        Peak intensities [L]

    Returns:
    -------
    torch.Tensor
        Updated mask with cap enforced
    """
    n_valid = (~spectra_mask).sum().item()
    max_allowed = int(n_valid * max_ratio)
    n_masked = batch_mask.sum().item()

    if n_masked <= max_allowed:
        return batch_mask  # Under cap

    # Need to remove excess peaks
    excess = n_masked - max_allowed

    # Get masked positions and their intensities
    masked_indices = torch.nonzero(batch_mask, as_tuple=False).squeeze(-1)
    masked_intensities = intensity[masked_indices]

    # Sort by intensity (ascending - remove lowest first)
    sorted_by_intensity = torch.argsort(masked_intensities)

    # Remove lowest-intensity peaks
    positions_to_unmask = masked_indices[sorted_by_intensity[:excess]]
    batch_mask[positions_to_unmask] = False

    return batch_mask


@torch.no_grad()
def signal_aware_fragment_mask(
    intensity: torch.Tensor,
    spectra_mask: torch.Tensor,
    mz: torch.Tensor,
    charges: torch.Tensor,
    peptides: list[str] | None,
    # Signal-aware parameters
    mask_portion: float = 0.30,
    min_backbone_coverage: float = 0.15,
    min_fragment_groups: int = 3,
    annotation_ppm: float = 20.0,
    annotation_cid_da_tol: float = 0.2,
    annotation_ion_types: tuple = ("b", "y"),
    frag_types: list[str | None] | None = None,
    max_mz: float = 2500.0,
    normalize_mz: bool = True,
    # Fallback parameters (thompson_span)
    span_min: int = 4,
    span_max: int = 7,
    alpha: float = 0.5,
    beta: float = 0.5,
    kappa: float = 4.0,
    gamma: float = 0.7,
    bidirectional: bool = True,
    include_isotopes: bool = False,
    isotope_ppm: float = 25.0,
    isotope_da_floor: float = 0.02,
    isotope_max_charge: int = 3,
    isotope_max_order: int = 2,
    max_total_mask_ratio: float = 0.40,
    # Performance
    num_workers: int = 4,
    # Diagnostics
    return_fallback_mask: bool = False,
) -> "torch.Tensor | tuple[torch.Tensor, list[bool]]":
    """Signal-aware fragment masking using theoretical annotation.

    Masks complete fragment ion groups (base + losses + isotopes) based on
    theoretical spectra annotation. Falls back to thompson_span for spectra
    without sequences or insufficient annotated fragments.

    When ``return_fallback_mask=True`` the function returns a
    ``(peak_mask, fallback_used)`` tuple instead of just the tensor;
    ``fallback_used`` is a list of per-spectrum booleans (``True`` if the
    thompson_span fallback was used). The tuple form is intended for the
    analyser/diagnostics path — training callers should leave this flag
    at its default so the existing tensor-only return shape is preserved.

    Parameters
    ----------
    intensity : torch.Tensor
        Intensity values [B, L] normalized to [0, 1]
    spectra_mask : torch.Tensor
        Padding mask [B, L], True for padded positions
    mz : torch.Tensor
        m/z values [B, L], may be normalized to [0, 1]
    charges : torch.Tensor
        Precursor charge states [B]
    peptides : list[str] | None
        Peptide sequences [B], None entries for missing sequences
    mask_portion : float
        Fraction of base fragment ions to mask
    min_backbone_coverage : float
        Minimum backbone cleavage coverage (0-1) for signal-aware mode.
        Spectra below this threshold fall back to thompson_span.
    min_fragment_groups : int
        Minimum unique fragment ion groups for signal-aware mode.
        Spectra below this threshold fall back to thompson_span.
    annotation_ppm : float
        PPM tolerance for annotation
    annotation_cid_da_tol : float
        Da tolerance for CID (ion-trap) spectra; used instead of PPM when frag_type is "CID"
    annotation_ion_types : tuple
        Ion types for annotation (e.g., ("b", "y"))
    frag_types : list[str | None] | None
        Per-spectrum fragmentation types [B] (e.g., "HCD", "CID"). Used to select
        Da tolerance for CID spectra via _da_tol_for_fragmentation().
    max_mz : float
        Maximum m/z value for denormalization
    normalize_mz : bool
        Whether input m/z is normalized
    span_min, span_max : int
        Span length range for fallback
    alpha, beta, kappa, gamma : float
        Thompson sampling parameters for fallback
    bidirectional : bool
        Bidirectional span expansion for fallback
    include_isotopes : bool
        Include isotope extension for fallback
    isotope_ppm, isotope_da_floor, isotope_max_charge, isotope_max_order : float/int
        Isotope detection parameters for fallback
    max_total_mask_ratio : float
        Hard cap on total masking fraction
    num_workers : int
        Number of multiprocessing workers for annotation

    Returns:
    -------
    torch.Tensor
        Boolean mask [B, L], True for masked positions
    """
    from multiprocessing import Pool

    B, L = intensity.shape  # noqa: N806
    device = intensity.device
    peak_mask = torch.zeros_like(spectra_mask, dtype=torch.bool)

    # Check if peptides available
    if peptides is None:
        # No peptides - use fallback for all spectra
        global_fallback = thompson_sampling_span_mask(
            intensity=intensity,
            spectra_mask=spectra_mask,
            mask_portion=mask_portion,
            span_min=span_min,
            span_max=span_max,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            gamma=gamma,
            bidirectional=bidirectional,
            mz=mz,
            charges=charges if include_isotopes else None,
            include_isotopes=include_isotopes,
            isotope_ppm=isotope_ppm,
            isotope_da_floor=isotope_da_floor,
            isotope_max_charge=isotope_max_charge,
            isotope_max_order=isotope_max_order,
            max_total_mask_ratio=max_total_mask_ratio,
            max_mz=max_mz,
            normalize_mz=normalize_mz,
        )
        if return_fallback_mask:
            return global_fallback, [True] * B
        return global_fallback

    # Denormalize m/z if needed
    if normalize_mz:
        mz_daltons = mz * max_mz
    else:
        mz_daltons = mz

    # Prepare annotation arguments for parallel processing
    # Import here to avoid circular imports at module level
    from instanovo_fm.utils.theoretical_spectra import _da_tol_for_fragmentation

    annotation_args: list[Any] = []
    for b in range(B):
        valid_mask = ~spectra_mask[b].cpu().numpy()

        if peptides[b] is None or not valid_mask.any():
            annotation_args.append(None)
            continue

        mz_np = mz_daltons[b].cpu().numpy()[valid_mask]
        intensity_np = intensity[b].cpu().numpy()[valid_mask]

        # Compute per-spectrum Da tolerance for CID spectra
        frag_type = frag_types[b] if frag_types is not None else None
        da_tol = _da_tol_for_fragmentation(frag_type, annotation_cid_da_tol) if annotation_cid_da_tol else None

        annotation_args.append(
            (
                mz_np,
                intensity_np,
                peptides[b],
                charges[b].item(),
                annotation_ppm,
                annotation_ion_types,
                da_tol,
                frag_type,
            )
        )

    # Parallel annotation
    # Check if we're in a daemon worker process (daemon processes can't spawn children)
    in_daemon_process = False
    try:
        # Check for PyTorch DataLoader workers
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()
        if worker_info is not None:
            in_daemon_process = True
    except Exception:
        pass

    # Also check for general multiprocessing daemon processes
    if not in_daemon_process:
        try:
            from multiprocessing import current_process

            in_daemon_process = current_process().daemon
        except Exception:
            pass

    # Use sequential processing if in daemon worker or if num_workers=1
    if num_workers > 1 and not in_daemon_process:
        # Parallel processing (only in main process)
        with Pool(processes=num_workers) as pool:
            annotation_results = pool.map(_annotate_single_spectrum_worker, annotation_args)
    else:
        # Sequential fallback (in daemon worker process or single worker mode)
        annotation_results = [_annotate_single_spectrum_worker(arg) if arg is not None else None for arg in annotation_args]

    # Import quality gate function
    from instanovo_fm.utils.modifications import clean_peptide_sequence
    from instanovo_fm.utils.peak_classification import compute_spectrum_quality

    # Process each spectrum
    n_signal_aware = 0
    n_fallback = 0
    fallback_used: list[bool] = [False] * B

    for b in range(B):
        # Get annotation result
        result = annotation_results[b]

        # Quality gate: check backbone coverage and fragment groups.
        # backbone_coverage is normalised by (clean_seq_len - 1), so the raw peptide
        # string (which includes modification tokens like "M[+15.995]") would inflate
        # the denominator and under-report coverage for modified peptides.
        use_signal_aware = False
        if result is not None and peptides[b] is not None:
            clean_seq = clean_peptide_sequence(peptides[b])
            seq_len = len(clean_seq)
            if seq_len >= 2:
                quality = compute_spectrum_quality(
                    feature_types=result.get("feature_type", []),
                    matched_annotations=result.get("matched_annotation", []),
                    seq_len=seq_len,
                )
                use_signal_aware = quality["backbone_coverage"] >= min_backbone_coverage and quality["n_fragment_groups"] >= min_fragment_groups

        if not use_signal_aware:
            # Fallback to thompson_span
            fallback_mask = thompson_sampling_span_mask(
                intensity=intensity[b : b + 1],
                spectra_mask=spectra_mask[b : b + 1],
                mask_portion=mask_portion,
                span_min=span_min,
                span_max=span_max,
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                gamma=gamma,
                bidirectional=bidirectional,
                mz=mz[b : b + 1],
                charges=charges[b : b + 1] if include_isotopes else None,
                include_isotopes=include_isotopes,
                isotope_ppm=isotope_ppm,
                isotope_da_floor=isotope_da_floor,
                isotope_max_charge=isotope_max_charge,
                isotope_max_order=isotope_max_order,
                max_total_mask_ratio=max_total_mask_ratio,
                max_mz=max_mz,
                normalize_mz=normalize_mz,
            )
            peak_mask[b] = fallback_mask.squeeze(0)
            n_fallback += 1
            fallback_used[b] = True
            continue

        # Expand annotation to full length
        valid_mask = ~spectra_mask[b].cpu().numpy()
        expanded_result = _expand_annotation_to_full_length(result, valid_mask, L)

        # Build parent-child mapping
        parent_groups = _build_parent_child_mapping(
            feature_types=expanded_result["feature_type"],
            parent_annotations=expanded_result["parent_annotation"],
            matched_annotations=expanded_result["matched_annotation"],
            intensity=intensity[b],
            spectra_mask=spectra_mask[b],
        )

        if len(parent_groups) == 0:
            # No valid groups - fallback
            fallback_mask = thompson_sampling_span_mask(
                intensity=intensity[b : b + 1],
                spectra_mask=spectra_mask[b : b + 1],
                mask_portion=mask_portion,
                span_min=span_min,
                span_max=span_max,
                alpha=alpha,
                beta=beta,
                kappa=kappa,
                gamma=gamma,
                bidirectional=bidirectional,
                mz=mz[b : b + 1],
                charges=charges[b : b + 1] if include_isotopes else None,
                include_isotopes=include_isotopes,
                isotope_ppm=isotope_ppm,
                isotope_da_floor=isotope_da_floor,
                isotope_max_charge=isotope_max_charge,
                isotope_max_order=isotope_max_order,
                max_total_mask_ratio=max_total_mask_ratio,
                max_mz=max_mz,
                normalize_mz=normalize_mz,
            )
            peak_mask[b] = fallback_mask.squeeze(0)
            n_fallback += 1
            fallback_used[b] = True
            continue

        # Select fragment ions to mask
        n_valid = (~spectra_mask[b]).sum().item()
        selected_parents = _select_fragment_ions_uniform(parent_groups, mask_portion, n_valid, max_total_mask_ratio)

        # Build mask from selected fragments
        batch_mask = torch.zeros(L, dtype=torch.bool, device=device)
        for parent_key in selected_parents:
            group = parent_groups[parent_key]
            batch_mask[group["parent_idx"]] = True
            if group["loss_indices"]:
                batch_mask[torch.tensor(group["loss_indices"], device=device)] = True
            if group["isotope_indices"]:
                batch_mask[torch.tensor(group["isotope_indices"], device=device)] = True

        # Enforce hard cap
        batch_mask = _enforce_mask_cap(batch_mask, spectra_mask[b], max_total_mask_ratio, intensity[b])

        peak_mask[b] = batch_mask
        n_signal_aware += 1

    # Log statistics
    if B > 0:
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            f"Signal-aware masking: {n_signal_aware}/{B} spectra "
            f"({n_signal_aware / B * 100:.1f}%), {n_fallback}/{B} fallback "
            f"({n_fallback / B * 100:.1f}%)"
        )

    if return_fallback_mask:
        return peak_mask, fallback_used
    return peak_mask


# ============================================================================
# Registry
# ============================================================================


def get_mask_function(strategy: str) -> Any:
    """Get masking function by name.

    Available strategies:
    - 'thompson': Thompson-sampled individual peaks
    - 'thompson_span': Thompson-sampled spans (with optional isotope extension)
    - 'uniform': Uniform random peaks
    - 'signal_aware_fragment': Signal-aware fragment ion masking (requires sequences)
    """
    registry = {
        "thompson": thompson_sampling_mask,
        "thompson_span": thompson_sampling_span_mask,
        "uniform": uniform_random_mask,
        "signal_aware_fragment": signal_aware_fragment_mask,
    }
    if strategy not in registry:
        raise ValueError(f"Unknown masking strategy: '{strategy}'. Available: {list(registry.keys())}")
    return registry[strategy]
