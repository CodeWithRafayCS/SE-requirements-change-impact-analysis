"""
GPU-Accelerated Text Embedder for SEOSS Dataset.
Supports PyTorch CUDA mixed precision (AMP autocast), dynamic batching, and disk caching.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Optional, Union
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class GPUEmbedder:
    """
    High-throughput dense embedding generator utilizing PyTorch CUDA FP16 autocast.
    Avoids memory exhaustion by batching and offloading to disk cache if desired.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        use_fp16: bool = True,
        batch_size: int = 256,
        cache_dir: Optional[str] = None
    ):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model_name = model_name
        self.use_fp16 = use_fp16 and (self.device == "cuda")
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self._model = None

    def _load_model(self):
        if self._model is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is not installed. Run: pip install sentence-transformers"
                )
            # Load model onto target device. We retain native parameters and rely on
            # torch.cuda.amp.autocast during forward pass to prevent FP16 LayerNorm instability.
            self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def embedding_dim(self) -> int:
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    def get_cache_path(self, cache_key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        safe_key = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in cache_key)
        return os.path.join(self.cache_dir, f"{safe_key}_{self.model_name.replace('/', '_')}.pt")

    def encode_texts(
        self,
        texts: List[str],
        cache_key: Optional[str] = None,
        show_progress: bool = True,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Encode a list of text strings into a 2D PyTorch Tensor (N x D) on CPU or GPU.
        Uses FP16 mixed precision autocast when running on CUDA.
        """
        if cache_key and self.cache_dir:
            cache_file = self.get_cache_path(cache_key)
            if cache_file and os.path.exists(cache_file):
                cached_data = torch.load(cache_file, map_location="cpu")
                return cached_data

        self._load_model()

        # Handle empty or whitespace strings
        cleaned_texts = [t if (isinstance(t, str) and t.strip()) else "empty" for t in texts]

        # Use SentenceTransformer encode with device & batch size under autocast if enabled
        with torch.no_grad():
            if self.device == "cuda" and self.use_fp16:
                with torch.cuda.amp.autocast():
                    embeddings = self._model.encode(
                        cleaned_texts,
                        batch_size=self.batch_size,
                        show_progress_bar=show_progress,
                        convert_to_tensor=True,
                        device=self.device,
                        normalize_embeddings=normalize
                    )
            else:
                embeddings = self._model.encode(
                    cleaned_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=show_progress,
                    convert_to_tensor=True,
                    device=self.device,
                    normalize_embeddings=normalize
                )

        # Move to CPU for safety across large repositories and caching
        embeddings = embeddings.cpu().float()

        if cache_key and self.cache_dir:
            cache_file = self.get_cache_path(cache_key)
            if cache_file:
                torch.save(embeddings, cache_file)

        return embeddings
