"""
Manager for patch inference.
"""
import asyncio
from itertools import cycle
import socket
import struct
import sys
import time
from typing import Optional, Union, Tuple, Iterator

import numpy as np
from PIL import Image
import yaml

from networking.scatter_gather import (
    request_patch,
    parse_server_addr,
    serialize_ndarray,
    deserialize_ndarray,
    recv_exact,
)
from sessions import BiRefNetTorchSession, U2NetTorchSession
from tile_proc.tiles import select_tiles_edge_mixture, extract_rgb_tiles, stitch_mask_tiles


# The session to do the base inference on the input image
base_session: Optional[Union[BiRefNetTorchSession, U2NetTorchSession]] = None

# The addresses (IP, port) of patch servers
server_addresses: Iterator[Tuple[str, int]]

MANAGER_PORT: int

# Struct stuff
LEN_PREFIX_FMT = "!Q"
LEN_PREFIX_SIZE = struct.calcsize(LEN_PREFIX_FMT)


async def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", MANAGER_PORT))
        srv.listen(100)
        print(f"[manager] listening on 0.0.0.0:{MANAGER_PORT}", file=sys.stderr)

        while True:
            conn, addr = srv.accept()
            try:
                print(f"[manager] connection from {addr}", file=sys.stderr)
                conn_start = time.perf_counter()

                hdr = recv_exact(conn, LEN_PREFIX_SIZE)
                (req_len,) = struct.unpack(LEN_PREFIX_FMT, hdr)
                data = recv_exact(conn, req_len)

                arr = deserialize_ndarray(data)
                if not isinstance(arr, np.ndarray) or arr.ndim != 3 or arr.shape[2] != 3:
                    raise ValueError("Input must be an RGB array (H, W, 3).")

                # Do the base prediction
                base_start = time.perf_counter()
                base_alpha = base_session.remove(Image.fromarray(arr), mask_only=True)
                base_end = time.perf_counter()

                # Generate boxes
                boxes = select_tiles_edge_mixture(base_alpha)
                # Make tiles from the boxes
                tiles = extract_rgb_tiles(np.dstack((arr, base_alpha)), boxes)

                # Scatter and gather tiles to and from patch inference server(s)
                patch_start = time.perf_counter()
                mask_tiles = await asyncio.gather(
                    *(request_patch(tile, next(server_addresses)) for tile in tiles)
                )
                patch_end = time.perf_counter()
                # Put the tiles together to make the result mask
                stitched = stitch_mask_tiles(
                    mask_tiles,
                    boxes,
                    out_shape=arr.shape[:2],
                    window_kind="hann"
                )
                out_bytes = serialize_ndarray(stitched)
                conn.sendall(struct.pack(LEN_PREFIX_FMT, len(out_bytes)))
                conn.sendall(out_bytes)
                conn_end = time.perf_counter()
                print(f"Base time: {base_end-base_start:.4f}s\n"
                      f"Patch time: {patch_end - patch_start:.4f}s\n"
                      f"Total time: {conn_end - conn_start:.4f}s", file=sys.stderr)
            except Exception as exc:
                print(f"[manager] error handling {addr}: {exc!r}", file=sys.stderr)
            finally:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                conn.close()


if __name__ == '__main__':
    # Parse args
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Make cycle of server addresses (fair-queuing, a.k.a. round robin)
    server_addr_conf = config["patch_servers"]
    server_addr_tmp = [parse_server_addr(x) for x in server_addr_conf]
    server_addresses = cycle(server_addr_tmp)

    # Parse the model name and make the session for this manager instance
    session_model_name = config.get("base_session", ["u2net"])[0]
    manager_device = config.get("manager_server_device", ["cpu"])[0]
    if session_model_name == 'u2net':
        base_session = U2NetTorchSession(use_small=False, device=manager_device)
    elif session_model_name == 'u2netp':
        base_session = U2NetTorchSession(use_small=True, device=manager_device)
    elif session_model_name == 'birefnet':
        base_session = BiRefNetTorchSession(net_path="models/birefnet.pth", device=manager_device)
    else:
        raise ValueError(f"Unknown session: {session_model_name}")

    MANAGER_PORT = int(config["manager_server_port"][0])

    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
    loop.close()
