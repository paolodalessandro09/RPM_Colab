import torch


def resolve_device(device_spec: str | None = None) -> str:
    """Resolve a user-facing device string to a torch-compatible device."""
    if device_spec is None:
        device_spec = "auto"

    spec = str(device_spec).strip().lower()

    if spec in {"auto", "gpu", "cuda"}:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if spec.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device '{device_spec}', but CUDA is not available."
            )
        return spec

    if spec == "cpu":
        return "cpu"

    raise ValueError(
        f"Unsupported device '{device_spec}'. Use 'auto', 'cpu', 'cuda', or 'cuda:N'."
    )
