import os
import time
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        if torch.is_tensor(val):
            val = val.detach().item()
        val = float(val)
        n = int(n)
        if n <= 0:
            return
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    if output.ndim != 2:
        raise ValueError(
            f"Classification output must have shape [B, C], got {tuple(output.shape)}."
        )
    if target.ndim > 1:
        target = target.argmax(dim=1)
    target = target.long().reshape(-1)

    if output.shape[0] != target.shape[0]:
        raise ValueError(
            f"Batch-size mismatch: output={output.shape[0]}, target={target.shape[0]}."
        )

    topk = tuple(int(k) for k in topk)
    if not topk or min(topk) <= 0:
        raise ValueError(f"topk must contain positive integers, got {topk}.")

    maxk = min(max(topk), output.shape[1])
    pred = output.topk(maxk, dim=1, largest=True, sorted=True).indices.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    batch_size = max(target.numel(), 1)
    for k in topk:
        k = min(k, output.shape[1])
        correct_k = correct[:k].reshape(-1).float().sum()
        results.append(correct_k / batch_size)
    return results


def _to_bhwc_float01(tensor, name):
    if not torch.is_tensor(tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}.")
    if tensor.ndim != 4:
        raise ValueError(
            f"{name} must have shape [B, C, H, W], got {tuple(tensor.shape)}."
        )
    if tensor.shape[1] not in (1, 3):
        raise ValueError(
            f"{name} must contain 1 or 3 channels, got {tensor.shape[1]}."
        )

    array = tensor.detach().float().cpu().numpy()
    array = np.clip(array, 0.0, 1.0)
    return np.transpose(array, (0, 2, 3, 1))


def _get_ssim_win_size(height, width):
    minimum = min(int(height), int(width))
    if minimum < 3:
        raise ValueError(
            f"SSIM requires image height and width of at least 3, got {(height, width)}."
        )
    win_size = min(7, minimum)
    if win_size % 2 == 0:
        win_size -= 1
    return win_size


def _compute_single_ssim(clean, restored):
    win_size = _get_ssim_win_size(clean.shape[0], clean.shape[1])
    channel_axis = -1 if clean.shape[-1] > 1 else None

    if channel_axis is None:
        clean = clean[..., 0]
        restored = restored[..., 0]

    try:
        return structural_similarity(
            clean,
            restored,
            data_range=1.0,
            channel_axis=channel_axis,
            win_size=win_size,
        )
    except TypeError:
        # Compatibility with older scikit-image versions.
        return structural_similarity(
            clean,
            restored,
            data_range=1.0,
            multichannel=channel_axis is not None,
            win_size=win_size,
        )


def compute_psnr_ssim(recovered, clean):
    if recovered.shape != clean.shape:
        raise ValueError(
            f"Recovered/clean shape mismatch: "
            f"{tuple(recovered.shape)} vs {tuple(clean.shape)}."
        )

    recovered_np = _to_bhwc_float01(recovered, "recovered")
    clean_np = _to_bhwc_float01(clean, "clean")
    batch_size = recovered_np.shape[0]

    if batch_size == 0:
        raise ValueError("Cannot compute PSNR/SSIM for an empty batch.")

    psnr_values = []
    ssim_values = []

    for index in range(batch_size):
        psnr_values.append(
            peak_signal_noise_ratio(
                clean_np[index],
                recovered_np[index],
                data_range=1.0,
            )
        )
        ssim_values.append(
            _compute_single_ssim(
                clean_np[index],
                recovered_np[index],
            )
        )

    return (
        float(np.mean(psnr_values)),
        float(np.mean(ssim_values)),
        batch_size,
    )


def compute_niqe(image):
    try:
        from skvideo.measure import niqe
    except ImportError as error:
        raise ImportError(
            "NIQE requires scikit-video. Install it with: pip install scikit-video"
        ) from error

    image_np = _to_bhwc_float01(image, "image")
    scores = []

    for rgb in image_np:
        if rgb.shape[-1] == 3:
            gray = (
                0.299 * rgb[..., 0]
                + 0.587 * rgb[..., 1]
                + 0.114 * rgb[..., 2]
            )
        else:
            gray = rgb[..., 0]

        score = niqe((gray * 255.0).astype(np.float32))
        scores.append(float(np.asarray(score).mean()))

    return float(np.mean(scores))


class timer:
    def __init__(self):
        self.acc = 0.0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self):
        return time.time() - self.t0

    def hold(self):
        self.acc += self.toc()

    def release(self):
        result = self.acc
        self.acc = 0.0
        return result

    def reset(self):
        self.acc = 0.0
