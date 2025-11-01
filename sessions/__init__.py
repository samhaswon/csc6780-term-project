try:
    from .onnx_session.session import Session, BiRefNetSession
except ModuleNotFoundError:
    Session = object
    BiRefNetSession = object
try:
    from .torch_session.patch_session import PatchTorchSession
except ModuleNotFoundError:
    PatchTorchSession = object
