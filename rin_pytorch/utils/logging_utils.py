from __future__ import annotations

from typing import Any

import torch

try:  # pragma: no cover - wandb optional at runtime
    import wandb
except Exception:  # pragma: no cover - keep training alive without wandb
    wandb = None  # type: ignore[assignment]


def _maybe_log(key: str, payload: Any) -> None:
    if wandb is None or getattr(wandb, "run", None) is None:
        return
    try:
        wandb.log({key: payload}, commit=False)
    except Exception:
        pass


def log_first_tensor(key: str, tensor: torch.Tensor | None) -> None:
    if tensor is None or tensor.numel() == 0:
        return
    value = tensor.detach().cpu()
    if value.dim() > 2:
        value = value.flatten()
    if value.dim() == 1:
        payload = value[: min(1024, value.numel())].tolist()
        _maybe_log(key, payload)
    elif wandb is not None:
        try:
            histogram = wandb.Histogram(value.numpy())
        except Exception:
            return
        _maybe_log(key, histogram)  # type: ignore[arg-type]


def log_first_document(
    key: str,
    tensor: torch.Tensor | None,
    document_ids: torch.Tensor | None,
) -> None:
    if tensor is None or document_ids is None or tensor.shape[0] != document_ids.shape[0]:
        return
    if document_ids.numel() == 0:
        return
    first_doc_id = int(document_ids[0].item())
    mask = document_ids == first_doc_id
    if mask.sum() == 0:
        return
    log_first_tensor(key, tensor[mask])

