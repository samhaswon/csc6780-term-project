"""
Server for patch inference.
"""
import sys

from PIL import Image
import numpy as np

from networking.scatter_gather import run_server
from sessions import Session


def usage(err: int = 1) -> None:
    """
    Print a usage message and exit the program.
    :param err: The error number.
    :return: None, exits.
    """
    print("Usage: python patch_server.py <server_port>", file=sys.stderr)
    exit(err)


def main(server_port):
    patch_session = Session("./models/chunks.onnx")

    def patch_process(x: np.ndarray, *_) -> np.ndarray:
        """
        Process the received patch with the patch session.
        This is effectively a fancy lambda.
        :param x: The input patch to process.
        :param _: None, for typing.
        :return: Processed patch.
        """
        return patch_session.predict(Image.fromarray(x), convert_to="RGBA")

    run_server(server_port, patch_process)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage(1)
    try:
        port = int(sys.argv[1])
        main(port)
    except ValueError:
        usage(2)
