"""Wait-policy aux head: a tiny MLP over the decoder hidden state (256-d,
input to decoder.output_layer) predicting p_wait = P(deferring >=1 future chunk
fixes this token). Trained jointly with the decoder in fine_tune_waitaux.py;
deployed as the per-token DEFER trigger (signal_type=wait_policy) in
batch_beam_search.py, replacing the beam-feature SR-CEM scorer.
"""
import torch
import torch.nn as nn


class AuxHead(nn.Module):
    """Tiny MLP head over the decoder hidden state predicting a p_wait logit.

    Defaults (d=256, h=64) must match fine_tune_waitaux checkpoints loaded
    by load_wait_aux.
    """

    def __init__(self, d=256, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x):
        """Map hidden states (..., d) to p_wait logits (...)."""
        return self.net(x).squeeze(-1)


class WaitAuxScorer:
    """Deploy-time wrapper: predict_from_hidden(256-d) -> p_wait in [0,1]."""

    def __init__(self, aux_head: AuxHead, device="cpu"):
        self.aux = aux_head.to(device).eval()
        self.device = device

    @torch.no_grad()
    def predict_from_hidden(self, hidden: torch.Tensor) -> float:
        h = hidden.to(self.device).float().reshape(-1)
        return float(torch.sigmoid(self.aux(h)).item())


def load_wait_aux(path, device="cpu"):
    """Load a fine_tune_waitaux checkpoint. Returns (state_dict, WaitAuxScorer).

    state_dict carries 'decoder' (fine-tuned decoder weights to load into the
    model) and 'aux_head' (the MLP weights).
    """
    st = torch.load(path, map_location=device)
    aux = AuxHead()
    aux.load_state_dict(st["aux_head"])
    return st, WaitAuxScorer(aux, device)
