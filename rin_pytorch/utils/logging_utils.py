import os
import json
from datetime import datetime

import torch
from torch.utils.data import get_worker_info
import torch.distributed as dist


_LOG_PATH: str | None = None


def set_log_path(path: str) -> None:
    """Initialize the global log file path and truncate it.

    Creates parent directories if needed and clears any previous contents.
    Safe to call multiple times; later calls will reinitialize the file path.
    """
    global _LOG_PATH
    _LOG_PATH = path
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    try:
        with open(path, "w") as f:
            f.write("")
    except Exception:
        # Best-effort; if we cannot write here, leave _LOG_PATH set so later appends can try again
        pass


def _is_primary_process() -> bool:
    """Return True if this is the primary process (rank 0) or if dist is not initialized."""
    try:
        if not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0
    except Exception:
        return True


def log_first_tensor(tag: str, tensor: torch.Tensor | None, step: int | None = None) -> None:
    """Append the first element of a tensor to a JSONL text file for tracing.

    - Only logs on primary process (global rank 0) to avoid duplication in DDP.
    - If tensor has a batch dimension, logs tensor[0]; otherwise logs tensor as-is.
    - Converts tensor to CPU float list for portability.
    """
    if _LOG_PATH is None:
        return
    # Allow Dataloader worker 0 to log, otherwise only primary process
    worker = None
    try:
        worker = get_worker_info()
    except Exception:
        worker = None
    if worker is not None:
        # Only log from (rank 0, worker 0)
        if worker.id != 0 or not _is_primary_process():
            return
    elif not _is_primary_process():
        return
    if tensor is None:
        return

    try:
        t = tensor
        # If the provided object isn't a tensor (e.g., list/tuple), try to take first element
        if not isinstance(t, torch.Tensor):
            try:
                t = t[0]
            except Exception:
                return

        t0 = t[0] if t.ndim >= 1 else t
        data = t0.detach().to("cpu").float().numpy().tolist()
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "tag": tag,
            "step": int(step) if step is not None else None,
            "shape": list(t0.shape),
            "data": data,
        }
        with open(_LOG_PATH, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        # Swallow all logging errors to avoid interfering with training
        pass


