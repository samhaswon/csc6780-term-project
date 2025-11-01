"""
Manager for patch inference.
"""
import argparse
import asyncio
from typing import Optional, Union, Tuple

import cv2
import numpy as np
from PIL import Image

from networking.scatter_gather import request_patch, parse_server_addr
from sessions import Session, BiRefNetSession
from tile_proc.tiles import select_tiles_edge_mixture, extract_rgb_tiles, stitch_mask_tiles


base_session: Optional[Union[Session, BiRefNetSession]] = None
server_addr: Optional[Tuple[str, int]] = None


async def main():
    test_image = Image.open("/home/samuel/da/skindataset/images/01097.png")
    if test_image.mode != "RGB":
        test_image = test_image.convert("RGB")
    test_image_np = np.array(test_image)
    base_alpha = base_session.predict(test_image)

    boxes = select_tiles_edge_mixture(base_alpha)
    tiles = extract_rgb_tiles(np.dstack((test_image_np, base_alpha)), boxes)

    mask_tiles_scatter = [request_patch(tile, server_addr) for tile in tiles]
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session",
        type=str,
        help="Session name, either u2net, u2netp, or birefnet",
        default='u2net',
    )
    parser.add_argument(
        "--server_addr",
        type=str,
        help="Server address",
        default="localhost:5432"
    )

    args = parser.parse_args()
    if args.session == 'u2net':
        base_session = Session(model_path="models/u2net.onnx")
    elif args.session == 'u2netp':
        base_session = Session(model_path="models/u2net.onnx")
    elif args.session == 'birefnet':
        base_session = BiRefNetSession(model_path="models/birefnet.onnx")
    else:
        raise ValueError(f"Unknown session: {args.session}")
    server_addr = parse_server_addr(args.server_addr)

    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())
    loop.close()
