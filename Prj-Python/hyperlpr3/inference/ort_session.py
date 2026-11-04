import os
import onnxruntime as ort


def session(model, default=None):
    device = os.environ.get("HYPERLPR_DEVICE")
    providers = default
    if device:
        provider = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider"}[device]
        if provider not in ort.get_available_providers():
            raise RuntimeError(f"Unavailable inference provider: {provider}")
        providers = [provider]
    engine = ort.InferenceSession(model, providers=providers)
    if device and provider not in engine.get_providers():
        raise RuntimeError(f"Failed to initialize {provider}")
    return engine
