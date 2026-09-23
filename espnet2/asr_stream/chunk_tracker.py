"""Chunk tracker for streaming ASR inference with delayed decoding."""

from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class ChunkTracker:
    """Manages buffering and delayed decoding logic for streaming ASR.

    This class implements chunk-by-chunk processing with configurable
    past and future context. Key features:

    1. Delayed decoding: With num_right_chunks=F, decoding is delayed
       by F chunks to allow future context.
    2. Latency tracking: Records chunk indices for each emitted token.

    Example with num_left_chunks=2, num_right_chunks=2:
    ```
    Audio chunks arrive:   0    1    2    3    4    5    ...
                           |    |    |    |    |    |
    Chunk ready to decode: -    -    0    1    2    3    ...
                                 ^-- chunk 0 decoded when chunk 2 arrives
    ```

    Attributes:
        chunk_size: Number of encoder frames per chunk.
        num_left_chunks: Past chunks visible (-1 = unlimited).
        num_right_chunks: Future chunks visible (determines delay).
        num_global_tokens: Number of global attention sink tokens.
    """

    chunk_size: int
    num_left_chunks: int = -1
    num_right_chunks: int = 0
    num_global_tokens: int = 0

    # Runtime state (initialized in __post_init__)
    encoder_chunks: List[torch.Tensor] = field(default_factory=list)
    current_chunk_idx: int = -1  # Index of most recent chunk (-1 = none)
    next_decode_idx: int = 0  # Next chunk index to decode

    # Two-pass streaming: raw feature chunks.
    # raw_chunks stores the pre-encoder (subsampled) features so that the
    # previous chunk can be re-encoded with the newly arrived chunk as future
    # context (the "final pass").
    raw_chunks: List[torch.Tensor] = field(default_factory=list)

    # Latency tracking
    token_emission_info: List[dict] = field(default_factory=list)

    def __post_init__(self):
        """Validate configuration."""
        assert self.chunk_size > 0, "chunk_size must be > 0"
        assert self.num_left_chunks >= -1, "num_left_chunks must be >= -1"
        # -1 = unlimited right context (offline-equivalent within a chunk).
        # Required when running the streaming pipeline with a large
        # single-chunk-per-utterance configuration for offline-equivalent testing.
        assert self.num_right_chunks >= -1, "num_right_chunks must be >= -1"
        assert self.num_global_tokens >= 0, "num_global_tokens must be >= 0"

    def reset(self):
        """Reset state for new utterance."""
        self.encoder_chunks = []
        self.raw_chunks = []
        self.current_chunk_idx = -1
        self.next_decode_idx = 0
        self.token_emission_info = []

    def add_chunk(self, enc_chunk: torch.Tensor) -> Optional[int]:
        """Add encoded chunk to buffer.

        Args:
            enc_chunk: Encoder output for this chunk, shape (chunk_size, D)
                       or (1, chunk_size, D).

        Returns:
            decode_chunk_idx if a chunk is ready for decoding, else None.

        With num_right_chunks=F:
        - Returns None until we have at least F+1 chunks
        - When chunk F arrives, returns 0 (decode chunk 0)
        - When chunk F+1 arrives, returns 1 (decode chunk 1)

        Special case: num_right_chunks=0 (immediate decoding)
        - Returns current_chunk_idx immediately
        """
        # Handle batch dimension
        if enc_chunk.dim() == 3 and enc_chunk.size(0) == 1:
            enc_chunk = enc_chunk.squeeze(0)

        self.encoder_chunks.append(enc_chunk)
        self.current_chunk_idx += 1

        # Check if we have enough future context for decoding
        # Chunk i can be decoded when chunk (i + num_right_chunks) arrives
        # That is: current_chunk_idx >= next_decode_idx + num_right_chunks
        if self.current_chunk_idx >= self.next_decode_idx + self.num_right_chunks:
            decode_idx = self.next_decode_idx
            self.next_decode_idx += 1
            return decode_idx

        return None

    # ------------------------------------------------------------------
    # Two-pass asymmetric chunk attention helpers
    # ------------------------------------------------------------------

    def add_raw_chunk(self, raw_chunk: torch.Tensor) -> None:
        """Store a raw (pre-encoder / post-subsampling) feature chunk.

        Must be called **before** :meth:`add_chunk` so that
        ``raw_chunks[i]`` corresponds to ``encoder_chunks[i]``.

        Args:
            raw_chunk: Feature tensor of shape ``(chunk_size, D)`` or
                       ``(1, chunk_size, D)`` (batch dim is squeezed).
        """
        if raw_chunk.dim() == 3 and raw_chunk.size(0) == 1:
            raw_chunk = raw_chunk.squeeze(0)
        self.raw_chunks.append(raw_chunk)

    def record_token_emission(self, token_id: int, decode_idx: int):
        """Record latency info when a token is emitted.

        Args:
            token_id: Emitted token ID.
            decode_idx: Chunk index that was being decoded.
        """
        # Calculate context window bounds
        if self.num_left_chunks < 0:
            oldest = 0
        else:
            oldest = max(0, decode_idx - self.num_left_chunks)

        newest = self.current_chunk_idx  # Most recent chunk available

        self.token_emission_info.append(
            {
                "token_id": token_id,
                "decode_chunk_idx": decode_idx,
                "oldest_chunk_idx": oldest,
                "newest_chunk_idx": newest,
            }
        )

    def get_latency_summary(self) -> dict:
        """Get summary statistics for latency analysis.

        Returns:
            Dictionary with latency metrics.
        """
        if not self.token_emission_info:
            return {}

        future_lookaheads = [
            info["newest_chunk_idx"] - info["decode_chunk_idx"]
            for info in self.token_emission_info
        ]
        past_contexts = [
            info["decode_chunk_idx"] - info["oldest_chunk_idx"]
            for info in self.token_emission_info
        ]

        return {
            "total_chunks": self.current_chunk_idx + 1,
            "total_tokens": len(self.token_emission_info),
            "avg_future_lookahead": sum(future_lookaheads) / len(future_lookaheads),
            "avg_past_context": sum(past_contexts) / len(past_contexts),
            "max_future_lookahead": max(future_lookaheads),
            "max_past_context": max(past_contexts),
        }

    def flush_remaining(self) -> List[int]:
        """Get remaining chunk indices to decode at end of utterance.

        Called when is_final=True to decode any buffered chunks that
        haven't been decoded yet.

        Returns:
            List of chunk indices to decode.
        """
        remaining = []
        while self.next_decode_idx <= self.current_chunk_idx:
            remaining.append(self.next_decode_idx)
            self.next_decode_idx += 1
        return remaining
