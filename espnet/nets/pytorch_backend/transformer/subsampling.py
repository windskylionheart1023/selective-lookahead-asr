#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""

import torch

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class TooShortUttError(Exception):
    """Raised when the utt is too short for subsampling.

    Args:
        message (str): Message for error catch
        actual_size (int): the short size that cannot pass the subsampling
        limit (int): the limit size for subsampling

    """

    def __init__(self, message, actual_size, limit):
        """Construct a TooShortUttError for error handler."""
        super().__init__(message)
        self.actual_size = actual_size
        self.limit = limit


def check_short_utt(ins, size):
    """Check if the utterance is too short for subsampling.

    If the subsampling module has do_padding enabled, any input of at
    least 1 frame is valid, so the limit is 1 in that case.
    """

    # If do_padding, then the input must be at least 1 frame in size.
    if getattr(ins, "do_padding", False):
        return size <= 0, 1

    if isinstance(ins, Conv1dSubsampling1) and size < 5:
        return True, 5
    if isinstance(ins, Conv1dSubsampling2) and size < 5:
        return True, 5
    if isinstance(ins, Conv1dSubsampling3) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling1) and size < 5:
        return True, 5
    if isinstance(ins, Conv2dSubsampling2) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling) and size < 7:
        return True, 7
    if isinstance(ins, Conv2dSubsampling6) and size < 11:
        return True, 11
    if isinstance(ins, Conv2dSubsampling8) and size < 15:
        return True, 15
    return False, -1


class Conv1dSubsampling1(torch.nn.Module):
    """Convolutional 1D subsampling.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
        do_padding (bool): If True, use symmetric zero-padding so each
            convolution is time-centered.  Default: False (no padding).

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, do_padding=False):
        """Construct an Conv1dSubsampling1 object."""
        super(Conv1dSubsampling1, self).__init__()
        self.do_padding = do_padding
        k1, s1, p1 = 3, 1, (1 if do_padding else 0)
        k2, s2, p2 = 3, 1, (1 if do_padding else 0)
        self.total_stride = s1 * s2

        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, k1, s1, padding=p1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, k2, s2, padding=p2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.transpose(2, 1)  # (#batch, idim, time)
        x = self.conv(x)
        b, c, t = x.size()
        x = self.out(x.transpose(1, 2).contiguous())
        if x_mask is None:
            return x, None

        if self.do_padding:
            x_mask = x_mask[:, :, ::self.total_stride]
        else:
            x_mask = x_mask[:, :, :-2:1][:, :, :-2:1]

        return x, x_mask

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv1dSubsampling2(torch.nn.Module):
    """Convolutional 1D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
        do_padding (bool): If True, use symmetric zero-padding so each
            convolution is time-centered.  Default: False (no padding).

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, do_padding=False):
        """Construct an Conv1dSubsampling2 object."""
        super(Conv1dSubsampling2, self).__init__()
        self.do_padding = do_padding
        k1, s1, p1 = 3, 1, (1 if do_padding else 0)
        k2, s2, p2 = 3, 2, (1 if do_padding else 0)
        self.total_stride = s1 * s2

        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, k1, s1, padding=p1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, k2, s2, padding=p2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.transpose(2, 1)  # (#batch, idim, time)
        x = self.conv(x)
        b, c, t = x.size()
        x = self.out(x.transpose(1, 2).contiguous())
        if x_mask is None:
            return x, None

        if self.do_padding:
            x_mask = x_mask[:, :, ::self.total_stride]
        else:
            x_mask = x_mask[:, :, :-2:1][:, :, :-2:2]

        return x, x_mask

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv1dSubsampling3(torch.nn.Module):
    """Convolutional 1D subsampling (to 1/3 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
        do_padding (bool): If True, use symmetric zero-padding so each
            convolution is time-centered.  Default: False (no padding).

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, do_padding=False):
        """Construct an Conv1dSubsampling3 object."""
        super(Conv1dSubsampling3, self).__init__()
        self.do_padding = do_padding
        k1, s1, p1 = 3, 1, (1 if do_padding else 0)
        k2, s2, p2 = 5, 3, (2 if do_padding else 0)
        self.total_stride = s1 * s2

        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(idim, odim, k1, s1, padding=p1),
            torch.nn.ReLU(),
            torch.nn.Conv1d(odim, odim, k2, s2, padding=p2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.transpose(2, 1)  # (#batch, idim, time)
        x = self.conv(x)
        b, c, t = x.size()
        x = self.out(x.transpose(1, 2).contiguous())
        if x_mask is None:
            return x, None

        if self.do_padding:
            x_mask = x_mask[:, :, ::self.total_stride]
        else:
            x_mask = x_mask[:, :, :-2:1][:, :, :-4:3]

        return x, x_mask

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
        do_padding (bool): If True, use symmetric zero-padding so each
            convolution is time-centered.  Default: False (no padding).

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, do_padding=False):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.do_padding = do_padding
        k1, s1, p1 = 3, 2, (1 if do_padding else 0)
        k2, s2, p2 = 3, 2, (1 if do_padding else 0)
        self.total_stride = s1 * s2

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, k1, s1, padding=p1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, k2, s2, padding=p2),
            torch.nn.ReLU(),
        )

        # ----- compute output freq bins after the two convs --------------
        f_after_c1 = (idim + 2 * p1 - k1) // s1 + 1
        f_after_c2 = (f_after_c1 + 2 * p2 - k2) // s2 + 1
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * f_after_c2, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None

        if self.do_padding:
            x_mask = x_mask[:, :, ::self.total_stride]
        else:
            x_mask = x_mask[:, :, :-2:2][:, :, :-2:2]

        return x, x_mask

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling1(torch.nn.Module):
    """Similar to Conv2dSubsampling module, but without any subsampling performed.

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
        do_padding (bool): If True, use symmetric zero-padding so each
            convolution is time-centered.  Default: False (no padding).

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, do_padding=False):
        """Construct an Conv2dSubsampling1 object."""
        super(Conv2dSubsampling1, self).__init__()
        self.do_padding = do_padding
        k1, s1, p1 = 3, 1, (1 if do_padding else 0)
        k2, s2, p2 = 3, 1, (1 if do_padding else 0)
        self.total_stride = s1 * s2

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, k1, s1, padding=p1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, k2, s2, padding=p2),
            torch.nn.ReLU(),
        )

        # ----- compute output freq bins after the two convs --------------
        f_after_c1 = (idim + 2 * p1 - k1) // s1 + 1
        f_after_c2 = (f_after_c1 + 2 * p2 - k2) // s2 + 1
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * f_after_c2, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Pass x through 2 Conv2d layers without subsampling.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim).
                where time' = time - 4.
            torch.Tensor: Subsampled mask (#batch, 1, time').
                where time' = time - 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None

        if self.do_padding:
            x_mask = x_mask[:, :, ::self.total_stride]
        else:
            x_mask = x_mask[:, :, :-4]

        return x, x_mask

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
        do_padding (bool): If True, use symmetric zero-padding so each
            convolution is time-centered.  Default: False (no padding).

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, do_padding=False):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.do_padding = do_padding
        k1, s1, p1 = 3, 2, (1 if do_padding else 0)
        k2, s2, p2 = 3, 1, (1 if do_padding else 0)
        self.total_stride = s1 * s2

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, k1, s1, padding=p1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, k2, s2, padding=p2),
            torch.nn.ReLU(),
        )

        # ----- compute output freq bins after the two convs --------------
        f_after_c1 = (idim + 2 * p1 - k1) // s1 + 1
        f_after_c2 = (f_after_c1 + 2 * p2 - k2) // s2 + 1
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * f_after_c2, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None

        if self.do_padding:
            x_mask = x_mask[:, :, ::self.total_stride]
        else:
            x_mask = x_mask[:, :, :-2:2][:, :, :-2:1]

        return x, x_mask

    def __getitem__(self, key):
        """Get item.

        When reset_parameters() is called, if use_scaled_pos_enc is used,
            return the positioning encoding.

        """
        if key != -1:
            raise NotImplementedError("Support only `-1` (for `reset_parameters`).")
        return self.out[key]


class Conv2dSubsampling6(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/6 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
        do_padding (bool): If True, use symmetric zero-padding so each
            convolution is time-centered.  Default: False (no padding).

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, do_padding=False):
        """Construct an Conv2dSubsampling6 object."""
        super(Conv2dSubsampling6, self).__init__()
        self.do_padding = do_padding
        k1, s1, p1 = 3, 2, (1 if do_padding else 0)
        k2, s2, p2 = 5, 3, (2 if do_padding else 0)
        self.total_stride = s1 * s2

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, k1, s1, padding=p1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, k2, s2, padding=p2),
            torch.nn.ReLU(),
        )

        # ----- compute output freq bins after the two convs --------------
        f_after_c1 = (idim + 2 * p1 - k1) // s1 + 1
        f_after_c2 = (f_after_c1 + 2 * p2 - k2) // s2 + 1
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * f_after_c2, odim),
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 6.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 6.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None

        if self.do_padding:
            x_mask = x_mask[:, :, ::self.total_stride]
        else:
            x_mask = x_mask[:, :, :-2:2][:, :, :-4:3]

        return x, x_mask


class Conv2dSubsampling8(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/8 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.
        do_padding (bool): If True, use symmetric zero-padding so each
            convolution is time-centered.  Default: False (no padding).

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None, do_padding=False):
        """Construct an Conv2dSubsampling8 object."""
        super(Conv2dSubsampling8, self).__init__()
        self.do_padding = do_padding
        k1, s1, p1 = 3, 2, (1 if do_padding else 0)
        k2, s2, p2 = 3, 2, (1 if do_padding else 0)
        k3, s3, p3 = 3, 2, (1 if do_padding else 0)
        self.total_stride = s1 * s2 * s3

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, k1, s1, padding=p1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, k2, s2, padding=p2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, k3, s3, padding=p3),
            torch.nn.ReLU(),
        )

        # ----- compute output freq bins after the three convs --------------
        f_after_c1 = (idim + 2 * p1 - k1) // s1 + 1
        f_after_c2 = (f_after_c1 + 2 * p2 - k2) // s2 + 1
        f_after_c3 = (f_after_c2 + 2 * p3 - k3) // s3 + 1
        self.out = torch.nn.Linear(odim * f_after_c3, odim)
        self.pos_enc = (
            pos_enc if pos_enc is not None else PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask, prefix_embeds=None):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).
            prefix_embeds (torch.Tensor or None): Prefix token embeddings
                (#batch, prefix_len, odim).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 8.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 8.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is not None:
            if self.do_padding:
                x_mask = x_mask[:, :, ::self.total_stride]
            else:
                x_mask = x_mask[:, :, :-2:2][:, :, :-2:2][:, :, :-2:2]

        if prefix_embeds is not None:
            x = torch.cat([prefix_embeds, x], dim=1)
            if x_mask is not None:
                x_mask = torch.cat(
                    [
                        torch.ones(
                            x_mask.shape[0],
                            1,
                            prefix_embeds.size(1),
                            dtype=x_mask.dtype,
                            device=x_mask.device,
                        ),
                        x_mask,
                    ],
                    dim=-1,
                )

        x = self.pos_enc(x)

        return x, x_mask
