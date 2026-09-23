import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from typeguard import typechecked


class CTC(torch.nn.Module):
    """CTC module.

    Args:
        odim: dimension of outputs
        encoder_output_size: number of encoder projection units
        dropout_rate: dropout rate (0.0 ~ 1.0)
        ctc_type: builtin, gtnctc, brctc, or soft_mask_ctc
        reduce: reduce the CTC loss into a scalar
        ignore_nan_grad: Same as zero_infinity (keeping for backward compatiblity)
        zero_infinity:  Whether to zero infinite losses and the associated gradients.
        mask_penalty_value: Large negative value for soft logit masking (optional)
    """

    @typechecked
    def __init__(
        self,
        odim: int,
        encoder_output_size: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: Optional[bool] = None,
        zero_infinity: bool = True,
        brctc_risk_strategy: str = "exp",
        brctc_group_strategy: str = "end",
        brctc_risk_factor: float = 0.0,
        mask_penalty_value: float = -100.0,
        chunk_size: int = 4,
    ):
        super().__init__()
        eprojs = encoder_output_size
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.ctc_type = ctc_type
        self.mask_penalty_value = mask_penalty_value
        # Accepted for config compatibility: saved model config.yaml files pass
        # ctc_conf.chunk_size; the live chunk size reaches forward() per batch.
        self.chunk_size = chunk_size

        if ignore_nan_grad is not None:
            zero_infinity = ignore_nan_grad

        if self.ctc_type == "builtin" or self.ctc_type == "soft_mask_ctc":
            self.ctc_loss = torch.nn.CTCLoss(
                reduction="none", zero_infinity=zero_infinity
            )
        elif self.ctc_type == "builtin2":
            self.ignore_nan_grad = True
            logging.warning("builtin2")
            self.ctc_loss = torch.nn.CTCLoss(reduction="none")

        elif self.ctc_type == "gtnctc":
            from espnet.nets.pytorch_backend.gtn_ctc import GTNCTCLossFunction

            self.ctc_loss = GTNCTCLossFunction.apply

        elif self.ctc_type == "brctc":
            try:
                import k2  # noqa
            except ImportError:
                raise ImportError("You should install K2 to use Bayes Risk CTC")

            from espnet2.asr.bayes_risk_ctc import BayesRiskCTC

            self.ctc_loss = BayesRiskCTC(
                brctc_risk_strategy, brctc_group_strategy, brctc_risk_factor
            )

        else:
            raise ValueError(
                f'ctc_type must be "builtin", "builtin2", "gtnctc", "brctc",'
                f' or "soft_mask_ctc": {self.ctc_type}'
            )

        self.reduce = reduce

    def _create_chunk_boundaries(self, alignments: torch.Tensor, ys_lens: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """Converts chunk index alignments (B, Lmax) into [start_frame, end_frame] boundaries (B, Lmax, 2).

        Args:
            alignments: 0-based chunk indices for each token, shape (B, Lmax)
            ys_lens: Length of each target sequence, shape (B,)
            chunk_size: Number of frames per chunk (dynamically sampled)

        Returns:
            chunk_boundaries: (B, Lmax, 2) with [start_frame, end_frame] for each token
        """
        B, Lmax = alignments.shape
        device = alignments.device

        # Initialize boundaries: (B, Lmax, 2)
        chunk_boundaries = torch.zeros((B, Lmax, 2), dtype=torch.long, device=device)

        for b in range(B):
            L = ys_lens[b] # Use ys_lens for length
            if L == 0:
                continue

            # Alignments are 0-based chunk indices (0, 1, 2, ...)
            chunk_numbers = alignments[b, :L].long()

            # Start Frames: F_start = chunk_num * chunk_size
            start_frames = chunk_numbers * chunk_size

            # End Frames: F_end = (chunk_num + 1) * chunk_size
            end_frames = (chunk_numbers + 1) * chunk_size

            chunk_boundaries[b, :L, 0] = start_frames
            chunk_boundaries[b, :L, 1] = end_frames

        return chunk_boundaries

    def _generate_soft_mask(self, logits_shape: Tuple[int, int, int], ys_pad: torch.Tensor, ys_lens: torch.Tensor, chunk_boundaries: torch.Tensor) -> torch.Tensor:
        """Creates the Soft Logit Mask (M) to penalize predictions outside chunks.

        Args:
            logits_shape: Shape (B, Tmax, D) of the CTC logits
            ys_pad: Batch of padded target token ids (B, Lmax)
            ys_lens: Length of each target sequence (B,)
            chunk_boundaries: Per-token [start_frame, end_frame] pairs (B, Lmax, 2)

        Returns:
            mask: Additive mask (B, Tmax, D). For each target token, frames
                outside its chunk receive mask_penalty_value at that token's
                logit index. If the same label occurs at several target
                positions, the penalties accumulate additively.
        """
        B, Tmax, D = logits_shape
        device = ys_pad.device

        # Initialize mask with zeros
        mask = torch.zeros((B, Tmax, D), dtype=torch.float32, device=device)

        for b in range(B):
            L = ys_lens[b]
            T = Tmax
            if L == 0:
                continue

            targets_l = ys_pad[b, :L]
            boundaries_l = chunk_boundaries[b, :L, :]

            for l in range(L):
                token_label = targets_l[l]
                start_frame, end_frame = boundaries_l[l]

                if start_frame >= end_frame:
                    continue

                # Frames before the chunk: [0, start_frame)
                mask[b, :start_frame, token_label] += self.mask_penalty_value

                # Frames after the chunk: [end_frame, Tmax)
                mask[b, end_frame:T, token_label] += self.mask_penalty_value

        return mask

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen) -> torch.Tensor:
        """Compute the CTC loss for the configured ctc_type.

        For "soft_mask_ctc", th_pred holds raw logits with the soft chunk
        mask already added; log_softmax is applied here and the same
        reduction logic as "builtin" is used.
        """
        if (
            self.ctc_type == "builtin"
            or self.ctc_type == "brctc"
            or self.ctc_type == "soft_mask_ctc"
        ):
            th_pred = th_pred.log_softmax(2).float()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            if self.ctc_type == "builtin" or self.ctc_type == "soft_mask_ctc":
                size = th_pred.size(1)
            else:
                size = loss.size(0)  # some invalid examples will be excluded

            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        # builtin2 ignores nan losses using the logic below, while
        # builtin relies on the zero_infinity flag in pytorch CTC
        elif self.ctc_type == "builtin2":
            th_pred = th_pred.log_softmax(2).float()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)

            if loss.requires_grad and self.ignore_nan_grad:
                # ctc_grad: (L, B, O)
                ctc_grad = loss.grad_fn(torch.ones_like(loss))
                ctc_grad = ctc_grad.sum([0, 2])
                indices = torch.isfinite(ctc_grad)
                size = indices.long().sum()
                if size == 0:
                    # Return as is
                    logging.warning(
                        "All samples in this mini-batch got nan grad."
                        " Returning nan value instead of CTC loss"
                    )
                elif size != th_pred.size(1):
                    logging.warning(
                        f"{th_pred.size(1) - size}/{th_pred.size(1)}"
                        " samples got nan grad."
                        " These were ignored for CTC loss."
                    )

                    # Create mask for target
                    target_mask = torch.full(
                        [th_target.size(0)],
                        1,
                        dtype=torch.bool,
                        device=th_target.device,
                    )
                    s = 0
                    for ind, le in enumerate(th_olen):
                        if not indices[ind]:
                            target_mask[s : s + le] = 0
                        s += le

                    # Calc loss again using maksed data
                    loss = self.ctc_loss(
                        th_pred[:, indices, :],
                        th_target[target_mask],
                        th_ilen[indices],
                        th_olen[indices],
                    )
            else:
                size = th_pred.size(1)

            if self.reduce:
                # Batch-size average
                loss = loss.sum() / size
            else:
                loss = loss / size
            return loss

        elif self.ctc_type == "gtnctc":
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, th_target, th_ilen, 0, "none")

        else:
            raise NotImplementedError

    def forward(self, hs_pad, hlens, ys_pad, ys_lens,
                alignments: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = None):
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
            alignments: The per-token chunk index for each sample (B, Lmax) - 0-based chunk indices
            chunk_size: Dynamic chunk size for converting chunk indices to frame boundaries.
                        If None, soft masking is skipped (non-streaming or full-attention mode).

        Returns:
            torch.Tensor: CTC loss value. A scalar if reduce is True,
                otherwise a per-sample loss of shape (B,).
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab) - Raw Logits
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))

        # soft_mask_ctc: penalize target-token logits outside their aligned chunk
        if self.ctc_type == "soft_mask_ctc":
            if alignments is None or chunk_size is None:
                # Non-streaming mode or full-attention: skip soft masking, use standard CTC
                ys_hat = ys_hat.transpose(0, 1)
                ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])
                loss = self.loss_fn(ys_hat, ys_true, hlens, ys_lens).to(
                    device=hs_pad.device, dtype=hs_pad.dtype
                )
                return loss

            # 1. Convert alignments (chunk indices) to [start, end] boundaries (B, Lmax, 2)
            chunk_boundaries = self._create_chunk_boundaries(alignments, ys_lens, chunk_size)

            # 2. Generate the soft mask M (B, Tmax, D)
            soft_mask = self._generate_soft_mask(ys_hat.size(), ys_pad, ys_lens, chunk_boundaries)

            # 3. Apply the mask to the logits: ys_hat_masked = ys_hat + M
            ys_hat_masked = ys_hat + soft_mask

            # 4. Prepare data for loss function
            # ys_hat_masked: (B, T, D) -> (T, B, D) for PyTorch CTCLoss
            ys_hat_masked_t_b_d = ys_hat_masked.transpose(0, 1)
            # ys_pad: (B, L) -> (BxL,)
            ys_true_flat = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])

            # 5. Compute the loss
            loss = self.loss_fn(
                ys_hat_masked_t_b_d, # (T, B, D) - Masked Logits
                ys_true_flat,        # (BxL,)
                hlens,               # (B,)
                ys_lens              # (B,)
            ).to(
                device=hs_pad.device, dtype=hs_pad.dtype
            )
            return loss

        elif self.ctc_type == "brctc":
            loss = self.loss_fn(ys_hat, ys_pad, hlens, ys_lens).to(
                device=hs_pad.device, dtype=hs_pad.dtype
            )
            return loss

        elif self.ctc_type == "gtnctc":
            # gtn expects list form for ys
            ys_true = [y[y != -1] for y in ys_pad]  # parse padded ys
        else:
            # ys_hat: (B, L, D) -> (L, B, D)
            ys_hat = ys_hat.transpose(0, 1)
            # (B, L) -> (BxL,)
            ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])

        # Note: If ctc_type is 'builtin', the unmasked logic falls here.
        loss = self.loss_fn(ys_hat, ys_true, hlens, ys_lens).to(
            device=hs_pad.device, dtype=hs_pad.dtype
        )

        return loss

    def softmax(self, hs_pad):
        """softmax of frame activations"""
        return F.softmax(self.ctc_lo(hs_pad), dim=2)

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations"""
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations"""
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)
