"""
Server for patch inference using PyTorch.
"""
from PIL import Image
import numpy as np
import yaml

from networking.scatter_gather import run_server
from sessions import TorchSession


def main(server_port):
    patch_session = TorchSession("./models/checkpoint.pth_38.tar")

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
    s_port = config['patch_server_port'][0]
    main(int(s_port))
