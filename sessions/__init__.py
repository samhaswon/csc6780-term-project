import sys
try:
    from .onnx_session.session import Session, BiRefNetSession
except ModuleNotFoundError:
    # Dummy objects that will throw if used.
    Session = object
    BiRefNetSession = object
    print(
        "Unable to load onnx sessions. "
        "If you are trying to use the onnx servers, make sure you have onnxruntime installed.",
        file=sys.stderr
    )
try:
    from .torch_session.patch_session import PatchTorchSession
    from .torch_session.birefnet_session import BiRefNetTorchSession
    from .torch_session.u2net_session import U2NetTorchSession
except ModuleNotFoundError:
    # Dummy objects that will throw if used.
    PatchTorchSession = object
    BiRefNetTorchSession = object
    print(
        "Unable to load torch sessions. "
        "If you are trying to use the PyTorch servers, make sure you have torch and torchvision "
        "installed.",
        file=sys.stderr
    )
