import json
import os
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlsplit

import torch
from jaxtyping import Bool, Float, Integer
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import Tensor, nn

from instanovo.__init__ import console
from instanovo.constants import LEGACY_PTM_TO_UNIMOD, MAX_SEQUENCE_LENGTH
from instanovo_fm.model.embeddings import (
    DualPeakEmbedding,
    FourierPeakEmbedding,
    LinearPeakEmbedding,
    MetaTokenEmbed,
    MultiScalePeakEmbedding,
    RBFPeakEmbedding,
)
from instanovo_fm.model.ion_ladder import IonLadderEncoder
from instanovo_fm.model.pad_token_mixin import PadTokenMixin
from instanovo.inference import Decodable
from instanovo.transformer.layers import (
    ConvPeakEmbedding,
    PositionalEncoding,
)
from instanovo.types import (
    DiscretizedMass,
    Peptide,
    PeptideMask,
    PrecursorFeatures,
    ResidueLogits,
    ResidueLogProbabilities,
    Spectrum,
    SpectrumEmbedding,
    SpectrumMask,
)
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.file_downloader import download_file
from instanovo.utils.residues import ResidueSet

# This model has its own checkpoint family. It was "transformer", inherited from
# the class this was derived from, which made from_pretrained offer InstaNovo's
# checkpoints -- a different architecture from the one load() builds.
MODEL_TYPE = "downstream_denovo"


logger = ColorLog(console, __name__).logger


class DownstreamDeNovo(nn.Module, Decodable, PadTokenMixin):
    """The Instanovo model."""

    def __init__(
        self,
        residue_set: ResidueSet,
        dim_model: int = 768,
        n_head: int = 16,
        dim_feedforward: int = 2048,
        encoder_layers: int = 9,
        decoder_layers: int = 9,
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        conv_peak_encoder: bool = False,
        peak_encoder_type: str = "multiscale",
        max_mz: float = 2500.0,
        min_mz: float = 0.0,
        max_charge: int = 10,
        use_meta_token: bool = True,
        cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or {}
        self._residue_set = residue_set
        self.vocab_size = len(residue_set)
        self.use_flash_attention = use_flash_attention
        self.conv_peak_encoder = conv_peak_encoder
        self.peak_encoder_type = peak_encoder_type
        self.max_mz = max_mz
        self.min_mz = min_mz
        self.max_charge = max_charge
        self.use_meta_token = use_meta_token
        self.dim_model = dim_model

        self.setup_pad_token(self.cfg, self.dim_model)
        logger.info(f"Flash attention is currently set to {self.use_flash_attention}.")

        # ===== Peak Encoder =====
        if self.peak_encoder_type == "fourier":
            fourier_config = self.cfg.get("peak_encoder", {}).get("config", {})
            num_freqs = fourier_config.get("num_freqs", dim_model // 4)
            strategy = fourier_config.get("strategy", "voronov_et_al")
            trainable_fourier = fourier_config.get("trainable_fourier", True)
            x_min = fourier_config.get("x_min", 0.001)
            x_max = fourier_config.get("x_max", 1.5)
            self.peak_encoder = FourierPeakEmbedding(
                dim_model,
                dropout=dropout,
                num_freqs=num_freqs,
                strategy=strategy,
                trainable_fourier=trainable_fourier,
                x_min=x_min,
                x_max=x_max,
            )
        elif self.peak_encoder_type == "rbf":
            rbf_config = self.cfg.get("peak_encoder", {}).get("config", {})
            num_rbf = rbf_config.get("num_rbf", 1024)
            normalize_mz = self.cfg.get("normalize_mz", True)
            self.peak_encoder = RBFPeakEmbedding(
                dim_model,
                dropout=dropout,
                min_mz=min_mz,
                max_mz=max_mz,
                num_rbf=num_rbf,
                normalize_mz=normalize_mz,
            )
        elif self.peak_encoder_type == "dual":
            dual_config = self.cfg.get("peak_encoder", {}).get("config", {})
            num_rbf = dual_config.get("num_rbf", 2048)
            normalize_mz = self.cfg.get("normalize_mz", True)
            self.peak_encoder = DualPeakEmbedding(
                dim_model,
                dropout=dropout,
                min_mz=min_mz,
                max_mz=max_mz,
                num_rbf=num_rbf,
                normalize_mz=normalize_mz,
            )
        elif self.peak_encoder_type == "linear":
            self.peak_encoder = LinearPeakEmbedding(dim_model, dropout=dropout)
        else:
            self.peak_encoder = MultiScalePeakEmbedding(dim_model, dropout=dropout)

        if self.conv_peak_encoder:
            self.conv_encoder = ConvPeakEmbedding(dim_model, dropout=dropout)

        # ===== Ion Ladder Encoder (optional) =====
        ion_ladder_cfg = self.cfg.get("ion_ladder", {})
        self.ion_ladder_enabled = ion_ladder_cfg.get("enabled", False)
        self.ion_ladder: IonLadderEncoder | None
        if self.ion_ladder_enabled:
            logger.info("Ion ladder is enabled.")
            residue_masses_dict = ion_ladder_cfg.get("residue_masses", residue_set.residue_masses)
            if not residue_masses_dict:
                raise ValueError("ion_ladder.enabled=True but no residue_masses provided in config or residue_set.")
            residue_masses_list = list(residue_masses_dict.values())
            neutral_losses: dict[str, float] | None = None
            nl_cfg = ion_ladder_cfg.get("neutral_losses", {})
            if nl_cfg.get("enabled", False):
                loss_types = nl_cfg.get("loss_types", ["H2O", "NH3"])
                neutral_losses = {name: IonLadderEncoder.KNOWN_LOSSES[name] for name in loss_types if name in IonLadderEncoder.KNOWN_LOSSES}
                if not neutral_losses:
                    raise ValueError(f"neutral_losses.enabled=True but none of {loss_types} are known. Known: {list(IonLadderEncoder.KNOWN_LOSSES)}")
            self.ion_ladder = IonLadderEncoder(
                d_model=dim_model,
                residue_masses=residue_masses_list,
                window_k=ion_ladder_cfg.get("window_k", 40),
                sigma=ion_ladder_cfg.get("sigma", 0.35),
                charge_states=tuple(ion_ladder_cfg.get("charge_states", [1, 2])),
                neutral_losses=neutral_losses,
            )
        else:
            self.ion_ladder = None

        # ===== Foundational Model Tokens =====
        self.latent_token = nn.Parameter(torch.randn(1, 1, dim_model))

        if self.use_meta_token:
            logger.info("Setting up meta token.")
            self.meta_embed = self._setup_meta_token()

        # ===== Encoder Stack =====
        self.encoder, self.pairwise_bias = self._setup_encoder(
            dim_model,
            n_head,
            dim_feedforward,
            dropout,
            encoder_layers,
        )

        # Decoder
        self.aa_embed = nn.Embedding(self.vocab_size, dim_model, padding_idx=0)

        self.aa_pos_embed = PositionalEncoding(dim_model, dropout, max_len=MAX_SEQUENCE_LENGTH)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_layers,
        )

        self.head = nn.Linear(dim_model, self.vocab_size)
        self.charge_encoder = nn.Embedding(max_charge, dim_model)

    @property
    def residue_set(self) -> ResidueSet:
        """Every model must have a `residue_set` attribute."""
        return self._residue_set

    @staticmethod
    def _get_causal_mask(seq_len: int, return_float: bool = False) -> PeptideMask:
        mask = (torch.triu(torch.ones(seq_len, seq_len)) == 1).transpose(0, 1)
        if return_float:
            return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return ~mask.bool()

    @staticmethod
    def get_pretrained() -> list[str]:
        """Get a list of pretrained model ids."""
        # Load the models.json file
        with resources.files("instanovo_fm").joinpath("models.json").open("r", encoding="utf-8") as f:
            models_config = json.load(f)

        if MODEL_TYPE not in models_config:
            return []

        return list(models_config[MODEL_TYPE].keys())

    @classmethod
    def load(
        cls, path: str, update_residues_to_unimod: bool = True, override_config: DictConfig | dict | None = None
    ) -> tuple["DownstreamDeNovo", "DictConfig"]:
        """Load a DownstreamDeNovo model from a checkpoint path.

        Args:
            path (str): Path to checkpoint file.
            update_residues_to_unimod (bool): Update residues to unimod, defaults to True.
            override_config (DictConfig | dict | None): Optional override config values with a DictConfig or dict, defaults to None.

        Returns:
            tuple[DownstreamDeNovo, DictConfig]: Tuple of model and config.
        """
        # Add to allow list
        _whitelist_torch_omegaconf()
        ckpt = torch.load(path, map_location="cpu", weights_only=True)

        config = ckpt["config"]

        if override_config is not None:
            if not isinstance(config, DictConfig):
                config = OmegaConf.create(config)
            with open_dict(config):
                config.update(override_config)

        # TODO: Remove
        if "state_dict" not in ckpt:
            ckpt["state_dict"] = ckpt["model"]

        # check if PTL checkpoint
        if all(x.startswith("model") for x in ckpt["state_dict"].keys()):
            ckpt["state_dict"] = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}

        if "residues" not in ckpt:
            # Legacy format
            residues = dict(config["residues"])
        else:
            # TODO: Remove
            residues = ckpt["residues"]

        if update_residues_to_unimod:
            residues = {LEGACY_PTM_TO_UNIMOD[k] if k in LEGACY_PTM_TO_UNIMOD else k: v for k, v in residues.items()}

        residue_set = ResidueSet(residues)

        model = cls(
            residue_set=residue_set,
            dim_model=config["dim_model"],
            n_head=config["n_head"],
            dim_feedforward=config["dim_feedforward"],
            encoder_layers=config.get("encoder_layers", config.get("n_layers", 9)),
            decoder_layers=config.get("decoder_layers", config.get("n_layers", 9)),
            dropout=config["dropout"],
            use_flash_attention=config.get("use_flash_attention", False),
            conv_peak_encoder=config.get("conv_peak_encoder", False),
            peak_encoder_type=config.get("peak_encoder", {}).get("type", "multiscale"),
            max_mz=config.get("max_mz", 2500.0),
            min_mz=config.get("min_mz", 0.0),
            max_charge=config.get("max_charge", 10),
            cfg=config,
            use_meta_token=config.get("use_meta_token", True),
        )
        model.load_state_dict(ckpt["state_dict"])

        return model, config

    @classmethod
    def from_pretrained(
        cls, model_id: str, update_residues_to_unimod: bool = True, override_config: DictConfig | dict | None = None
    ) -> tuple["DownstreamDeNovo", "DictConfig"]:
        """Download and load by model id or model path.

        Args:
            model_id (str): Model id or model path.
            update_residues_to_unimod (bool): Update residues to unimod, defaults to True.
            override_config (DictConfig | dict | None): Optional override config values with a DictConfig or dict, defaults to None.

        Returns:
            tuple[DownstreamDeNovo, DictConfig]: Tuple of model and config.
        """
        # TODO Refactor to use across methods
        # Check if model_id is a local file path
        if "/" in model_id or "\\" in model_id or model_id.endswith(".ckpt"):
            if os.path.isfile(model_id):
                return cls.load(model_id, update_residues_to_unimod=update_residues_to_unimod, override_config=override_config)
            else:
                raise FileNotFoundError(f"No file found at path: {model_id}")

        # Load the models.json file
        with resources.files("instanovo_fm").joinpath("models.json").open("r", encoding="utf-8") as f:
            models_config = json.load(f)

        # Find the model in the config
        if MODEL_TYPE not in models_config or model_id not in models_config[MODEL_TYPE]:
            raise ValueError(f"Model {model_id} not found in models.json, options are [{', '.join(models_config[MODEL_TYPE].keys())}]")

        model_info = models_config[MODEL_TYPE][model_id]
        url = model_info["remote"]

        # Create cache directory if it doesn't exist
        cache_dir = Path.home() / ".cache" / "instanovo-fm"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate a filename for the cached model
        file_name = urlsplit(url).path.split("/")[-1]
        cached_file = cache_dir / file_name

        # Check if the file is already cached
        if not cached_file.exists():
            download_file(url, cached_file, model_id, file_name)

        else:
            logger.info(f"Model {model_id} already cached at {cached_file}")

        try:
            # Load and return the model
            logger.info(f"Loading model {model_id} (remote)")
            return cls.load(str(cached_file), update_residues_to_unimod=update_residues_to_unimod, override_config=override_config)
        except Exception as e:
            logger.warning(f"Failed to load cached model {model_id}, it may be corrupted. Deleting and re-downloading. Error: {e}")
            if cached_file.exists():
                cached_file.unlink()

            download_file(url, cached_file, model_id, file_name)
            logger.info(f"Loading newly downloaded model {model_id}")
            return cls.load(str(cached_file), update_residues_to_unimod=update_residues_to_unimod, override_config=override_config)

    def load_encoder_from_foundation(self, foundation_path: str) -> None:
        """Transfer encoder-side weights from a foundation checkpoint.

        Args:
            foundation_path: Path to the foundation checkpoint file.
        """
        from instanovo_fm.model.encoder import FoundationModel

        foundation_model, _ = FoundationModel.load(foundation_path)
        foundation_state = foundation_model.state_dict()
        own_state = self.state_dict()

        # Encoder-side prefixes to transfer; decoder/head/bin keys ignored.
        encoder_prefixes: tuple[str, ...] = (
            "peak_encoder.",
            "encoder.",
            "pairwise_bias.",
            "ion_ladder.",
            "meta_embed.",
            "latent_token",
            "pad_token",
        )

        # Structural compatibility check: every downstream encoder-prefix
        # submodule must have at least one matching key in the foundation state.
        for prefix in encoder_prefixes:
            downstream_has = any(k.startswith(prefix) for k in own_state)
            foundation_overlap = any(k.startswith(prefix) and k in own_state for k in foundation_state)
            if downstream_has and not foundation_overlap:
                downstream_examples = [k for k in own_state if k.startswith(prefix)][:3]
                foundation_examples = [k for k in foundation_state if k.startswith(prefix)][:3]
                raise RuntimeError(
                    f"Downstream has '{prefix}*' submodule but no foundation key matches. "
                    f"Downstream keys e.g. {downstream_examples}; "
                    f"foundation keys e.g. {foundation_examples or '(none with this prefix)'}. "
                    f"Likely peak_encoder.type / ion_ladder.enabled / architecture.* drift, "
                    f"or renamed submodule in the foundation checkpoint."
                )

        loaded = 0
        skipped_no_slot_keys: list[str] = []

        for key, param in foundation_state.items():
            # Foundation key we don't want to transfer
            if not any(key.startswith(p) for p in encoder_prefixes):
                continue
            if key not in own_state:
                skipped_no_slot_keys.append(key)
                continue
            if own_state[key].shape != param.shape:
                raise RuntimeError(
                    f"Shape mismatch for {key}: foundation={param.shape}, "
                    f"downstream={own_state[key].shape}. Encoder weight transfer "
                    f"would silently fall back to random init for this tensor. "
                    f"Check downstream config against foundation checkpoint config."
                )
            own_state[key].copy_(param)
            loaded += 1

        logger.info(f"Loaded {loaded} encoder parameters from foundation checkpoint")
        if skipped_no_slot_keys:
            logger.warning(f"{len(skipped_no_slot_keys)} foundation keys with no downstream slot: {skipped_no_slot_keys}")

    def _setup_meta_token(self) -> MetaTokenEmbed:
        """Setup meta token embedding for precursor/metadata encoding."""
        meta_cfg = self.cfg.get("meta_token", {})

        # Get vocabulary sizes from metadata builder
        from instanovo_fm.data.metadata_builder import get_vocabulary_sizes

        vocab_sizes = get_vocabulary_sizes()

        # Extract field inclusion flags from config
        include_frag_type = meta_cfg.get("include_frag_type", True)
        include_instrument = meta_cfg.get("include_instrument", True)
        include_acquisition = meta_cfg.get("include_acquisition", True)
        include_detector = meta_cfg.get("include_detector", True)
        include_enzyme = meta_cfg.get("include_enzyme", True)
        include_quant = meta_cfg.get("include_quant", True)
        include_precursor_charge = meta_cfg.get("include_precursor_charge", True)
        include_precursor_mass = meta_cfg.get("include_precursor_mass", True)
        include_collision_energy = meta_cfg.get("include_collision_energy", True)

        return MetaTokenEmbed(
            proj_dim=self.dim_model,
            n_freq=meta_cfg.get("n_freq", 16),
            # Vocabulary sizes from metadata builder
            n_frag=vocab_sizes["n_frag"],
            n_instrument=vocab_sizes["n_instrument"],
            n_acquisition=vocab_sizes["n_acquisition"],
            n_detector=vocab_sizes["n_detector"],
            n_enzyme=vocab_sizes["n_enzyme"],
            n_quant=vocab_sizes["n_quant"],
            n_charge=vocab_sizes["n_charge"],
            # Continuous feature bounds
            precursor_mass_max=meta_cfg.get("precursor_mass_max", 10000.0),
            # Field inclusion
            include_frag_type=include_frag_type,
            include_instrument=include_instrument,
            include_acquisition=include_acquisition,
            include_detector=include_detector,
            include_enzyme=include_enzyme,
            include_quant=include_quant,
            include_precursor_charge=include_precursor_charge,
            include_precursor_mass=include_precursor_mass,
            include_collision_energy=include_collision_energy,
        )

    def _setup_encoder(
        self,
        dim_model: int,
        n_head: int,
        dim_feedforward: int,
        dropout: float,
        encoder_layers: int,
    ) -> Tuple[nn.Module, Optional[nn.Module]]:
        """Setup encoder stack — unified if architecture config present, else standard.

        Args:
            dim_model: Model dimension.
            n_head: Number of attention heads.
            dim_feedforward: Feedforward dimension.
            dropout: Dropout rate.
            encoder_layers: Number of encoder layers.

        Returns:
            Tuple of (encoder, pairwise_bias) where pairwise_bias may be None.
        """
        arch_cfg = self.cfg.get("architecture", {})
        if arch_cfg:
            from instanovo_fm.model.encoder_layers.factories import create_unified_encoder_stack

            result: Tuple[nn.Module, Optional[nn.Module]] = create_unified_encoder_stack(
                cfg=self.cfg,
                d_model=dim_model,
                n_heads=n_head,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                n_layers=encoder_layers,
            )
            return result

        # Fallback: standard PyTorch TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0 if self.use_flash_attention else dropout,
        )
        encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
        )
        return encoder, None

    def _add_special_tokens(
        self,
        x: torch.Tensor,
        meta: Optional[Dict[str, torch.Tensor]],
        spectra_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """Add latent token and optional meta tokens to sequence.

        Args:
            x: Peak embeddings (B, L, D)
            meta: Optional metadata dictionary
            spectra_mask: Optional spectra padding mask (B, L)

        Returns:
            Tuple of (updated_x, num_prepended, updated_spectra_mask)
            where num_prepended includes latent token + all metadata tokens
        """
        batch_size = x.shape[0]

        # Add latent token
        x = torch.cat([self.latent_token.expand(batch_size, -1, -1), x], dim=1)
        num_prepended = 1

        # Add meta tokens if enabled and metadata provided
        if self.use_meta_token and meta is not None:
            # Ensure metadata tensors are on correct device
            tensor_meta = {k: v.to(x.device) if isinstance(v, torch.Tensor) else v for k, v in meta.items()}

            # Get meta token embeddings (B, n_meta_tokens, D)
            meta_tokens = self.meta_embed(tensor_meta)  # (batch_size, n_tokens, D)

            # Insert meta tokens after latent token
            x = torch.cat([x[:, :1], meta_tokens, x[:, 1:]], dim=1)
            num_prepended += meta_tokens.shape[1]  # Add number of meta tokens

            # Pad masks if provided
            if spectra_mask is not None:
                n_meta_tokens = meta_tokens.shape[1]
                spectra_mask = torch.cat([spectra_mask.new_zeros(batch_size, n_meta_tokens), spectra_mask], dim=1)

        return x, num_prepended, spectra_mask

    def _compute_attn_bias(
        self,
        spectra: torch.Tensor,
        spectra_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute attention bias and pairwise features using PA bias if enabled.

        Args:
            spectra: Input spectra (B, L, 2 or 3) [m/z, intensity, (charge)]
            spectra_mask: Optional padding mask (B, L), True for padded positions.
                If provided, pairwise features involving padded positions
                are zeroed out to prevent meaningless distance computations
                (padded positions have m/z=0 which is not a valid value).

        Returns:
            Tuple of (attn_bias, pairwise_feats):
                - attn_bias: None (PA bias applied per-layer via g_pw)
                - pairwise_feats: (B, L, L, hidden_dim) or None
        """
        if self.pairwise_bias is None:
            return None, None

        # Extract m/z and de-normalize to Daltons.
        # spectra[..., 0] is normalized to [0, 1] by the data processor (÷ max_mz).
        # PairwiseAttentionBias lambda_min/lambda_max are calibrated in Daltons,
        # so passing normalized values makes low-freq components near-flat and
        # high-freq components aliased — destroying the Fourier basis.
        mz = spectra[..., 0:1] * self.max_mz  # (B, L, 1) in Daltons

        # Compute pairwise features via PairwiseAttentionBias
        pairwise_feats = self.pairwise_bias(mz=mz)  # (B, L, L, hidden_dim)

        # Zero out pairwise features to prevent information leakage.
        invalid = None
        if spectra_mask is not None:
            invalid = spectra_mask if invalid is None else (invalid | spectra_mask)
        if invalid is not None and invalid.any():
            invalid_2d = invalid.unsqueeze(2) | invalid.unsqueeze(1)  # (B, L, L)
            pairwise_feats = pairwise_feats.masked_fill(invalid_2d.unsqueeze(-1), 0.0)

        # No attn_bias — the per-layer g_pw projection handles it
        return None, pairwise_feats

    def _pad_attn_bias(
        self,
        attn_bias: Optional[torch.Tensor],
        num_prepended: int,
    ) -> Optional[torch.Tensor]:
        """Pad attention bias to account for prepended tokens.

        Args:
            attn_bias: Attention bias (B, H, L, L)
            num_prepended: Number of prepended tokens (latent + meta)

        Returns:
            Padded attention bias (B, H, L+num_prepended, L+num_prepended) or None
        """
        if attn_bias is None:
            return None

        B, H, L, _ = attn_bias.shape  # noqa: N806
        total_len = L + num_prepended

        # Create padded bias with zeros for prepended tokens
        padded = torch.zeros(
            B,
            H,
            total_len,
            total_len,
            device=attn_bias.device,
            dtype=attn_bias.dtype,
        )
        padded[:, :, num_prepended:, num_prepended:] = attn_bias

        return padded

    def forward(
        self,
        x: Float[Spectrum, " batch"],
        p: Float[PrecursorFeatures, " batch"],
        y: Integer[Peptide, " batch"],
        x_mask: Optional[Bool[SpectrumMask, " batch"]] = None,
        y_mask: Optional[Bool[PeptideMask, " batch"]] = None,
        add_bos: bool = True,
        return_encoder_output: bool = False,
        meta: Optional[torch.Tensor] = None,
    ) -> Float[ResidueLogits, "batch token+1"]:
        """Model forward pass.

        Args:
            x: Spectra, float Tensor (batch, n_peaks, 2)
            p: Precursors, float Tensor (batch, 3)
            y: Peptide, long Tensor (batch, seq_len, vocab)
            x_mask: Spectra padding mask, True for padded indices, bool Tensor (batch, n_peaks)
            y_mask: Peptide padding mask, bool Tensor (batch, seq_len)
            add_bos: Force add a <s> prefix to y, bool

        Returns:
            logits: float Tensor (batch, n, vocab_size),
            (batch, n+1, vocab_size) if add_bos==True.
        """
        x, x_mask = self._encoder(x, p, x_mask, meta)
        y = self._decoder(x, y, x_mask, y_mask, p=p, add_bos=add_bos)
        if return_encoder_output:
            return y, x
        return y

    def init(
        self,
        spectra: Float[Spectrum, " batch"],
        precursors: Float[PrecursorFeatures, " batch"],
        spectra_mask: Optional[Bool[SpectrumMask, " batch"]] = None,
    ) -> Tuple[
        Tuple[Float[Spectrum, " batch"], Bool[SpectrumMask, " batch"]],
        Float[ResidueLogProbabilities, "batch token"],
    ]:
        """Initialise model encoder."""
        spectra, spectra_mask = self._encoder(spectra, precursors, spectra_mask)
        logits = self._decoder(spectra, None, spectra_mask, None, p=precursors, add_bos=False)
        return (spectra, spectra_mask), torch.log_softmax(logits[:, -1, :], -1)

    def score_candidates(
        self,
        sequences: Integer[Peptide, " batch"],
        precursor_mass_charge: Float[PrecursorFeatures, " batch"],
        spectra: Float[Spectrum, " batch"],
        spectra_mask: Bool[SpectrumMask, " batch"],
    ) -> Float[ResidueLogProbabilities, "batch token"]:
        """Score a set of candidate sequences."""
        logits = self._decoder(spectra, sequences, spectra_mask, None, p=precursor_mass_charge, add_bos=True)

        return torch.log_softmax(logits[:, -1, :], -1)

    def get_residue_masses(self, mass_scale: int) -> Integer[DiscretizedMass, " residue"]:
        """Get the scaled masses of all residues."""
        residue_masses = torch.zeros(len(self.residue_set), dtype=torch.int64)
        for index, residue in self.residue_set.index_to_residue.items():
            if residue in self.residue_set.residue_masses:
                residue_masses[index] = round(mass_scale * self.residue_set.get_mass(residue))
        return residue_masses

    def get_eos_index(self) -> int:
        """Get the EOS token ID."""
        return int(self.residue_set.EOS_INDEX)

    def get_empty_index(self) -> int:
        """Get the PAD token ID."""
        return int(self.residue_set.PAD_INDEX)

    def decode(self, sequence: Peptide) -> list[str]:
        """Decode a single sequence of AA IDs."""
        # Note: Sequence is reversed as InstaNovo predicts right-to-left.
        # We reverse the sequence again when decoding to ensure
        # the decoder outputs forward sequences.
        return self.residue_set.decode(sequence, reverse=True)  # type: ignore

    def idx_to_aa(self, idx: Peptide) -> list[str]:
        """Decode a single sample of indices to aa list."""
        idx = idx.cpu().numpy()
        t = []
        for i in idx:
            if i == self.eos_id:
                break
            if i == self.bos_id or i == self.pad_id:
                continue
            t.append(i)
        return [self.i2s[x.item()] for x in t]

    def batch_idx_to_aa(self, idx: Integer[Peptide, " batch"], reverse: bool) -> list[list[str]]:
        """Decode a batch of indices to aa lists."""
        return [self.residue_set.decode(i, reverse=reverse) for i in idx]

    def score_sequences(
        self,
        peptides: Integer[Peptide, " batch"] | list[str] | list[list[str]],
        peptides_mask: Bool[PeptideMask, " batch"] | None = None,
        spectra: Float[Spectrum, " batch"] | None = None,
        precursors: Float[PrecursorFeatures, " batch"] | None = None,
        spectra_mask: Bool[SpectrumMask, " batch"] | None = None,
        spectra_embedding: Float[SpectrumEmbedding, " batch"] | None = None,
        max_batch_size: int = 256,
    ) -> Float[ResidueLogProbabilities, "batch token"]:
        """Score a set of peptides."""
        if (spectra is None and precursors is None) and spectra_embedding is None:
            raise ValueError("Either spectra and precursors or spectra_embedding must be provided")

        if not isinstance(peptides, Tensor):
            peptides = [
                self.residue_set.encode(
                    self.residue_set.tokenize(x)[::-1],  # type: ignore # ensure reversed
                    add_eos=True,
                    return_tensor="pt",
                )
                for x in peptides
            ]

            ll = torch.tensor([x.shape[0] for x in peptides], dtype=torch.long)  # type: ignore
            peptides = nn.utils.rnn.pad_sequence(peptides, batch_first=True)
            peptides_mask = (
                torch.arange(peptides.shape[1], dtype=torch.long)[None, :] >= ll[:, None]  # type: ignore
            )

            device = spectra.device if spectra is not None else spectra_embedding.device  # type: ignore

            peptides = peptides.to(device)
            peptides_mask = peptides_mask.to(device)

        # Automatically handle batching if the number of peptides is too large
        if peptides.shape[0] > max_batch_size:
            sequence_scores = []
            for i in range(0, peptides.shape[0], max_batch_size):
                sub_batch = (
                    x[i : i + max_batch_size] if x is not None else None
                    for x in (
                        peptides,
                        peptides_mask,
                        spectra,
                        precursors,
                        spectra_mask,
                        spectra_embedding,
                    )
                )
                sequence_scores.append(self.score_sequences(*sub_batch))  # type: ignore
            return torch.cat(sequence_scores, dim=0)

        with torch.no_grad():
            if spectra_embedding is None:
                spectra_embedding, spectra_mask = self._encoder(spectra, precursors, spectra_mask)  # TODO add meta?

            logits = self._decoder(spectra_embedding, peptides, spectra_mask, peptides_mask, p=precursors, add_bos=True)

        # Get log probabilities for all positions
        log_probs = torch.log_softmax(logits, -1)

        # Gather log probabilities for each token in the sequence
        sequence_log_prob = torch.gather(log_probs, -1, peptides.unsqueeze(-1)).squeeze(-1)

        # Zero out masked positions
        if peptides_mask is not None:
            sequence_log_prob = sequence_log_prob.masked_fill(peptides_mask, 0.0)

        # Sum log probabilities across sequence length
        sequence_log_prob = sequence_log_prob.sum(dim=-1)

        return sequence_log_prob.cpu()

    def _embed_peaks(self, spectra: torch.Tensor) -> torch.Tensor:
        """Embed peaks using configured peak encoder.

        Args:
            spectra: Input spectra (B, L, 2) [m/z, intensity]

        Returns:
            Peak embeddings (B, L, D)
        """
        return self.peak_encoder(spectra)

    def _apply_ion_ladder(
        self,
        x: torch.Tensor,
        spectra: torch.Tensor,
        spectra_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply IonLadderEncoder to peak embeddings (if enabled).

        Args:
            x: Peak embeddings (B, L, D)
            spectra: Raw spectra (B, L, 2) [m/z, intensity]
            spectra_mask: Optional padding mask (B, L), True for padding

        Returns:
            Enhanced peak embeddings (B, L, D), or unchanged x if disabled.
        """
        if self.ion_ladder is None:
            return x

        # De-normalize m/z to Daltons
        mz_daltons = spectra[..., 0] * self.max_mz  # (B, L)

        # Compute padding mask
        if spectra_mask is not None:
            pad_mask = spectra_mask.bool()
        else:
            pad_mask = spectra.sum(dim=-1) == 0  # (B, L)

        return self.ion_ladder(mz_daltons, x, pad_mask=pad_mask)

    def _apply_padding(self, x: torch.Tensor, spectra_mask: Optional[torch.Tensor], spectra: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply padding replacement for Flash Attention compatibility.

        Args:
            x: Peak embeddings (B, L, D)
            spectra_mask: Optional padding mask (B, L)
            spectra: Original spectra for fallback padding detection (B, L, 2)

        Returns:
            Tuple of (padded_embeddings, padding_mask)
        """
        # Determine padding mask
        if spectra_mask is not None:
            pad_mask = spectra_mask.bool()
        else:
            # Fallback: detect padding from zero spectra
            pad_mask = spectra.sum(dim=-1) == 0

        # Apply pad token replacement (no-op unless Flash Attention is enabled)
        x = self.apply_pad_token_replacement(x, pad_mask)

        return x, pad_mask

    def _pad_pairwise_feats(
        self,
        pairwise_feats: Optional[torch.Tensor],
        num_prepended: int,
    ) -> Optional[torch.Tensor]:
        """Pad pairwise features to account for prepended tokens.

        Args:
            pairwise_feats: Pairwise features (B, L, L, R)
            num_prepended: Number of prepended tokens (latent + meta)

        Returns:
            Padded pairwise features (B, L+num_prepended, L+num_prepended, R) or None
        """
        if pairwise_feats is None or num_prepended == 0:
            return pairwise_feats

        B, L, _, R = pairwise_feats.shape  # noqa: N806
        total_len = L + num_prepended

        padded = torch.zeros(
            B,
            total_len,
            total_len,
            R,
            device=pairwise_feats.device,
            dtype=pairwise_feats.dtype,
        )
        padded[:, num_prepended:, num_prepended:, :] = pairwise_feats

        return padded

    def _create_padding_mask(
        self,
        pad_mask: Optional[torch.Tensor],
        num_prepended: int,
    ) -> Optional[torch.Tensor]:
        """Create padding mask for encoder, accounting for prepended tokens.

        Args:
            pad_mask: Original padding mask (B, L)
            num_prepended: Number of prepended tokens

        Returns:
            Padded mask (B, L+num_prepended) or None
        """
        if pad_mask is None:
            return None

        # Prepend zeros for special tokens (not padded)
        prepended_zeros = torch.zeros(
            pad_mask.size(0),
            num_prepended,
            device=pad_mask.device,
            dtype=pad_mask.dtype,
        )
        return torch.cat([prepended_zeros, pad_mask], dim=1)

    def _encoder(
        self,
        spectra: torch.Tensor,
        precursors: Optional[torch.Tensor] = None,
        spectra_mask: Optional[torch.Tensor] = None,
        meta: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Float[SpectrumEmbedding, " batch"], Bool[SpectrumMask, " batch"]]:
        """Encode spectra into embeddings.

        Args:
            spectra: Input spectra (B, L, 2 or 3) [m/z, intensity, (charge)]
            precursors: Optional precursor info (B, 3) [mz, charge, ...] (deprecated, use meta)
            spectra_mask: Optional padding mask (B, L), True for padding
            meta: Optional metadata dictionary for meta token

        Returns:
            x: Embeddings (B, L, D)
            x_mask: Padding mask (B, L), True for padding
        """
        # 1. Embed peaks
        x = self._embed_peaks(spectra)

        # 2. Apply IonLadder (before padding)
        x = self._apply_ion_ladder(x, spectra, spectra_mask)

        # 3. Apply padding replacement
        x, pad_mask = self._apply_padding(x, spectra_mask, spectra)

        # 4. Compute attention bias (before adding special tokens)
        attn_bias, pairwise_feats = self._compute_attn_bias(spectra, pad_mask)

        # 5. Add special tokens (latent + meta)
        x, num_prepended, spectra_mask = self._add_special_tokens(x, meta, spectra_mask)

        # 6. Pad attention bias for special tokens
        attn_bias = self._pad_attn_bias(attn_bias, num_prepended)
        pairwise_feats = self._pad_pairwise_feats(pairwise_feats, num_prepended)

        # 7. Create padding mask for encoder.
        x_mask = self._create_padding_mask(pad_mask, num_prepended)

        # 8. Encode through transformer
        if isinstance(self.encoder, nn.TransformerEncoder):
            # Standard PyTorch TransformerEncoder
            x = self.encoder(
                src=x,
                mask=None,
                src_key_padding_mask=x_mask,
            )
        else:
            x = self.encoder(
                src=x,
                src_mask=None,
                src_key_padding_mask=x_mask,
                attn_bias=attn_bias,
                pairwise_feats=pairwise_feats,
                is_causal=False,
            )

        return x, x_mask

    def _decoder(
        self,
        x: Float[Spectrum, " batch"],
        y: Integer[Peptide, " batch"],
        x_mask: Bool[SpectrumMask, " batch"],
        y_mask: Optional[Bool[PeptideMask, " batch"]] = None,
        p: Optional[torch.Tensor] = None,
        add_bos: bool = True,
    ) -> Float[ResidueLogits, " batch"]:
        # Prepare precursors
        if p is not None:
            masses = self.peak_encoder.encode_mass(p[:, None, [0]])
            charges = self.charge_encoder(p[:, 1].int() - 1)
            precursors = masses + charges[:, None, :]

            # Concatenate precursors
            x = torch.cat([precursors, x], dim=1)
            prec_mask = torch.zeros((x_mask.shape[0], 1), dtype=bool, device=x_mask.device)
            x_mask = torch.cat([prec_mask, x_mask], dim=1)

        if y is None:
            y = torch.full((x.shape[0], 1), self.residue_set.SOS_INDEX, device=x.device)
        elif add_bos:
            bos = torch.ones((y.shape[0], 1), dtype=y.dtype, device=y.device) * self.residue_set.SOS_INDEX
            y = torch.cat([bos, y], dim=1)

            if y_mask is not None:
                bos_mask = torch.zeros((y_mask.shape[0], 1), dtype=bool, device=y_mask.device)
                y_mask = torch.cat([bos_mask, y_mask], dim=1)

        y = self.aa_embed(y)
        if y_mask is None:
            y_mask = ~y.sum(axis=2).bool()

        # concat bos
        y = self.aa_pos_embed(y)

        c_mask = self._get_causal_mask(y.shape[1]).to(y.device)

        y_hat = self.decoder(
            y,
            x,
            tgt_mask=c_mask,
            tgt_key_padding_mask=y_mask,
            memory_key_padding_mask=x_mask,
        )

        return self.head(y_hat)


def _whitelist_torch_omegaconf() -> None:
    """Whitelist specific modules for loading configs from checkpoints."""
    # This is done to safeguard against arbitrary code execution from checkpoints.
    from collections import defaultdict
    from typing import Any

    from omegaconf.base import ContainerMetadata, Metadata
    from omegaconf.listconfig import ListConfig
    from omegaconf.nodes import AnyNode

    torch.serialization.add_safe_globals(
        [
            DictConfig,
            ContainerMetadata,
            Metadata,
            ListConfig,
            AnyNode,
            Any,  # Only used for type hinting in omegaconf.
            defaultdict,
            dict,
            list,
            int,
        ]
    )
