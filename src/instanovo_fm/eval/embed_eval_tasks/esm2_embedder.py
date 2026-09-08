from __future__ import annotations
import logging
import torch
from torch.cuda.amp import autocast
import numpy as np

# Try to import esm, but handle the case where it's not available
try:
    import esm
    ESM_AVAILABLE = True
except ImportError:
    ESM_AVAILABLE = False
    esm = None

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ESM2Embedder:
    """
    A lightweight embedder for generating ESM2 protein sequence embeddings on the fly.

    Example:
        embedder = ESM2Embedder(model_name='esm2_t33_650M_UR50D')
        embeddings = embedder.embed(['MKAILVVLLYTFATANADTQ', 'FAVTVDVDV'], batch_size=16)
    """

    MODEL_DEFAULT_LAYERS: dict[str, int] = {
        'esm2_t48_15B_UR50D': 48,
        'esm2_t36_3B_UR50D': 36,
        'esm2_t33_650M_UR50D': 33,
        'esm2_t30_150M_UR50D': 30,
        'esm2_t12_35M_UR50D': 12,
        'esm2_t6_8M_UR50D': 6,
    }

    def __init__(
        self,
        model_name: str = 'esm2_t33_650M_UR50D',
        device: str | None = None,
    ):
        """
        Load the specified ESM2 model.

        :param model_name: Identifier for the pre-trained model in esm.pretrained
        :param device: Torch device string (e.g. 'cuda:0' or 'cpu'). Auto-detects CUDA if None.
        """
        if not ESM_AVAILABLE:
            raise ImportError(
                "ESM2 is not available. Please install it with: pip install fair-esm"
            )

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model_name = model_name
        self.model, self.batch_converter = self._load_model()
        self.model.eval()

    def _load_model(self):
        logging.info(f"Loading ESM2 model '{self.model_name}' on {self.device}")
        try:
            esm_model, alphabet = getattr(esm.pretrained, self.model_name)()
        except AttributeError as e:
            raise ValueError(f"Unknown model '{self.model_name}'. Available models: {list(esm.pretrained.__dict__.keys())}")

        esm_model = esm_model.to(self.device)
        batch_converter = alphabet.get_batch_converter()
        return esm_model, batch_converter

    def embed(
        self,
        sequences: list[str] | str,
        layer: int | None = None,
        batch_size: int = 32,
        pooling: str = 'mean',
    ) -> np.ndarray:
        """
        Generate embeddings for the provided sequences.

        :param sequences: A single sequence or list of sequences (strings)
        :param layer: Which transformer layer to extract (default is model's final layer)
        :param batch_size: Number of sequences per batch
        :param pooling: Pooling strategy ('mean', 'cls', 'attention')
        :return: NumPy array of shape (n_sequences, embedding_dim)
        """
        # Prepare sequence list
        if isinstance(sequences, str):
            sequences = [sequences]
        if not sequences:
            return np.empty((0, 0))

        # Filter and clean sequences
        valid_sequences = []
        valid_indices = []
        for i, seq in enumerate(sequences):
            if seq is None or not isinstance(seq, str):
                continue
            # Remove any non-amino acid characters and convert to uppercase
            cleaned_seq = ''.join(c.upper() for c in seq if c.upper() in 'ACDEFGHIKLMNPQRSTVWY')
            if len(cleaned_seq) > 0:
                valid_sequences.append(cleaned_seq)
                valid_indices.append(i)

        if not valid_sequences:
            logging.warning("No valid amino acid sequences found")
            return np.empty((0, 0))

        # Determine layer
        if layer is None:
            layer = self.MODEL_DEFAULT_LAYERS.get(self.model_name)
            if layer is None:
                raise ValueError(
                    f"No default layer for model '{self.model_name}'. Please specify layer manually."
                )

        # Create label sequence pairs
        labeled = [(f'seq{i}', seq) for i, seq in enumerate(valid_sequences)]
        embeddings: list[np.ndarray] = []

        # Process in batches
        for start in range(0, len(labeled), batch_size):
            batch = labeled[start : start + batch_size]
            emb = self._embed_batch(batch, layer, pooling)
            embeddings.append(emb)

        # Concatenate
        valid_embeddings = np.vstack(embeddings)

        # If we filtered out some sequences, we need to create a full array with zeros for invalid sequences
        if len(valid_indices) < len(sequences):
            # Get embedding dimension from valid embeddings
            if len(valid_embeddings) > 0:
                embedding_dim = valid_embeddings.shape[1]
                # Create full array with zeros for invalid sequences
                full_embeddings = np.zeros((len(sequences), embedding_dim), dtype=valid_embeddings.dtype)
                # Place valid embeddings at their original indices
                for valid_idx, orig_idx in enumerate(valid_indices):
                    full_embeddings[orig_idx] = valid_embeddings[valid_idx]
                return full_embeddings
            else:
                # No valid embeddings at all
                return np.empty((len(sequences), 0))

        return valid_embeddings

    def _embed_batch(
        self,
        batch: list[tuple[str, str]],
        layer: int,
        pooling: str = 'mean',
    ) -> np.ndarray:
        """
        Compute embeddings for a single batch with specified pooling strategy.

        Args:
            batch: List of (label, sequence) tuples
            layer: Transformer layer to extract
            pooling: Pooling strategy ('mean', 'cls', 'attention')

        Returns:
            Embeddings array
        """
        labels, seqs, tokens = self.batch_converter(batch)
        tokens = tokens.to(self.device)

        with torch.no_grad(), autocast():
            out = self.model(tokens, repr_layers=[layer], return_contacts=False)
            reps = out['representations'][layer]

        embeddings = []
        for i, seq in enumerate(seqs):
            seq_len = len(seq)

            if pooling == 'cls':
                # Extract BOS token (index 0) - this is the ESM-2 CLS analogue
                # BOS is trained to attend to the whole sequence and contains learned summary
                emb = reps[i, 0].cpu().numpy()  # BOS token at position 0
            elif pooling == 'attention':
                # For now, fall back to mean pooling
                # TODO: Implement attention-weighted pooling
                emb = reps[i, 1 : seq_len + 1].mean(dim=0).cpu().numpy()
            else:  # 'mean' (default)
                # Mean pool over sequence tokens (exclude start/end)
                emb = reps[i, 1 : seq_len + 1].mean(dim=0).cpu().numpy()

            embeddings.append(emb)

        return np.stack(embeddings)
