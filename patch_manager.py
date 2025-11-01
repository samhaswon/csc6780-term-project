"""
Manager for patch inference.
"""
import asyncio
from itertools import cycle
from typing import Optional, Union, Tuple, Iterator

import cv2
import numpy as np
from PIL import Image
import yaml

from networking.scatter_gather import request_patch, parse_server_addr
from sessions import Session, BiRefNetSession
from tile_proc.tiles import select_tiles_edge_mixture, extract_rgb_tiles, stitch_mask_tiles


base_session: Optional[Union[Session, BiRefNetSession]] = None
server_addresses: Iterator[Tuple[str, int]]


async def main():
    test_image = Image.open("/home/samuel/da/skindataset/images/01097.png")
    if test_image.mode != "RGB":
        test_image = test_image.convert("RGB")
    test_image_np = np.array(test_image)
    base_alpha = base_session.predict(test_image)

    boxes = select_tiles_edge_mixture(base_alpha)
    tiles = extract_rgb_tiles(np.dstack((test_image_np, base_alpha)), boxes)

    mask_tiles_scatter = [request_patch(tile, next(server_addresses)) for tile in tiles]
    mask_tiles = [await tile for tile in mask_tiles_scatter]
    stitched = stitch_mask_tiles(
        mask_tiles,
        boxes,
        out_shape=test_image_np.shape[:2],
        window_kind="hann"
    )
    result = np.dstack((cv2.cvtColor(test_image_np, cv2.COLOR_RGB2BGR), stitched))
    cv2.imwrite("test.png", result)


if __name__ == '__main__':
    # Parse args
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    # Make cycle of server addresses (fair-queuing, a.k.a. round robin)
    server_addr_conf = config["patch_servers"]
    server_addr_tmp = [parse_server_addr(x) for x in server_addr_conf]
    server_addresses = cycle(server_addr_tmp)

    session_model_name = config.get("base_session", ["u2net"])[0]

    if session_model_name == 'u2net':
        base_session = Session(model_path="models/u2net.onnx")
    elif session_model_name == 'u2netp':
        base_session = Session(model_path="models/u2net.onnx")
    elif session_model_name == 'birefnet':
        base_session = BiRefNetSession(model_path="models/birefnet.onnx")
    else:
        raise ValueError(f"Unknown session: {session_model_name}")


    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
    loop.close()
