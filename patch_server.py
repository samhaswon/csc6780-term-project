"""
Server for patch inference.
"""
from PIL import Image
import numpy as np
import yaml

from networking.scatter_gather import run_server
from sessions import Session


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
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)
    server_port = config['patch_server_port'][0]
    main(int(server_port))
