"""
Generate boxes for the skin mask for demonstration purposes.
"""
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from tile_proc.tiles import select_tiles_edge_mixture
from sessions import Session


def draw_boxes(image: np.ndarray, boxes: List[Tuple[int, int, int, int]], box_line_thickness: int = 2) -> np.ndarray:
    """
    Draw green rectangles on a copy of the image.

    :param image: BGR image.
    :param boxes: Boxes as (x0, y0, x1, y1) with x1/y1 exclusive.
    :param box_line_thickness: The thickness of the box lines.
    :returns: Annotated image.
    """
    out = image.copy()
    for (x0, y0, x1, y1) in boxes:
        cv2.rectangle(out, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), thickness=box_line_thickness)
    return out



if __name__ == '__main__':
    # AI test image
    image_pil = Image.open("test_inputs/test.jpg")
    if image_pil.mode == "RGBA":
        pil_image = image_pil.convert(mode="RGB")
    image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    base_session = Session("models/u2net.onnx")
    base_alpha = base_session.predict(image_pil)

    boxes = select_tiles_edge_mixture(base_alpha)
    boxed = draw_boxes(image_np, boxes, box_line_thickness=4)

    cv2.imwrite("test_boxed.jpg", boxed)
    print(f"Number of boxes: {len(boxes)}")

