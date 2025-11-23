"""
Client of the patch manager.
"""
from itertools import cycle
import socket
import struct
from typing import Union, Tuple, Iterator

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
import yaml

from networking.scatter_gather import (
    parse_server_addr,
    serialize_ndarray,
    deserialize_ndarray,
    recv_exact
)


# Number of benchmark iterations
NUM_ITERATIONS: int = 10
# Whether to do the benchmark (True for cluster testing)
DO_BENCHMARK: bool = True

# Struct stuff
LEN_PREFIX_FMT = "!Q"
LEN_PREFIX_SIZE = struct.calcsize(LEN_PREFIX_FMT)

# The addresses (IP, port) of patch servers
server_addresses: Iterator[Tuple[str, int]]


def _ensure_rgb_numpy(image: Union[np.ndarray, PILImage]) -> np.ndarray:
    """
    Ensure an RGB numpy array (H, W, 3) uint8.

    :param image: PIL image or numpy array
    :returns: numpy RGB array uint8
    """
    if isinstance(image, PILImage):
        rgb = image.convert("RGB")
        arr = np.array(rgb, dtype=np.uint8)
        return arr

    if isinstance(image, np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected numpy array with shape (H, W, 3).")
        if image.dtype != np.uint8:
            return image.astype(np.uint8, copy=False)
        return image

    raise ValueError("Unsupported image type, provide PIL.Image or numpy.ndarray.")


def _send_and_receive_sync(
    host: str,
    port: int,
    payload: bytes,
    *,
    timeout: float = 120.0,
) -> bytes:
    """
    Blocking TCP exchange, length-prefixed request and response.

    :param host: Server hostname or IP.
    :param port: Server port.
    :param payload: Request payload bytes.
    :param timeout: Socket timeout in seconds.
    This should be at least 60s for JIT compile time on the first run.
    :returns: Response payload bytes.
    """
    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(struct.pack(LEN_PREFIX_FMT, len(payload)))
        sock.sendall(payload)

        hdr = recv_exact(sock, LEN_PREFIX_SIZE)
        (resp_len,) = struct.unpack(LEN_PREFIX_FMT, hdr)
        return recv_exact(sock, resp_len)


def segment_image(
    image: Union[np.ndarray, "PILImage.Image"],
    server_addr: Union[Tuple[str, int], str],
) -> np.ndarray:
    """
    Send an RGB image (H, W, 3) to a manager server and receive an (H, W) mask.

    :param image: PIL Image or numpy array (H, W, 3)
    :param server_addr: (host, port) or 'host:port'
    :returns: numpy array (H, W) uint8
    """
    arr = _ensure_rgb_numpy(image)
    payload = serialize_ndarray(arr)
    host, port = parse_server_addr(server_addr)
    # The timeout is really high mostly for the single inference node case.
    resp = _send_and_receive_sync(host, port, payload, timeout=6000)
    out = deserialize_ndarray(resp)

    if not isinstance(out, np.ndarray) or out.ndim != 2 or out.shape[:2] != arr.shape[:2]:
        raise ValueError("Server returned unexpected array shape.")
    return out.astype(np.uint8, copy=False)


if __name__ == '__main__':
    # Parse args
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Make cycle of server addresses (fair-queuing, a.k.a. round robin)
    server_addr_conf = config["manager_servers"]
    server_addr_tmp = [parse_server_addr(x) for x in server_addr_conf]
    server_addresses = cycle(server_addr_tmp)

    # Load test image
    test_image = Image.open("test_inputs/test.jpg")
    if test_image.mode != "RGB":
        test_image = test_image.convert("RGB")
    test_image_np = np.array(test_image)

    # Run inference on the test image
    result_alpha = segment_image(test_image_np, next(server_addresses))

    # Write out the result
    result = np.dstack((cv2.cvtColor(test_image_np, cv2.COLOR_RGB2BGR), result_alpha))
    cv2.imwrite("test.png", result)

    # The previous run should have handled the JIT/eager-mode warmup.
    # So start running the main benchmark (logs are from the manager).
    if DO_BENCHMARK:
        print("Warmup done. Doing inference test.")
        for _ in range(NUM_ITERATIONS):
            segment_image(test_image_np, next(server_addresses))
        print("Done")
