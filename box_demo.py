"""
Generate boxes for the skin mask for demonstration purposes.
"""
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

from tile_proc.tiles import select_tiles_edge_mixture
from sessions import Session, BiRefNetSession


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

    cv2.imwrite("test_boxed.jpg", boxed, [cv2.IMWRITE_JPEG_QUALITY, 75])
    print(f"Number of boxes: {len(boxes)}")

    u2net_alpha = base_alpha
    del base_session
    u2netp_session = Session("models/u2netp.onnx")
    u2netp_alpha = u2netp_session.predict(image_pil)
    del u2netp_session

    birefnet_session = BiRefNetSession("models/birefnet.onnx")
    birefnet_alpha = birefnet_session.predict(image_pil)
    del birefnet_session

    # Extract the same-index box from each alpha and concatenate horizontally
    box_index = 0  # change this to pick a different box index

    crops = []
    model_names = ["u2net", "u2netp", "birefnet"]
    alphas = [u2net_alpha, u2netp_alpha, birefnet_alpha]

    for name, alpha in zip(model_names, alphas):
        model_boxes = select_tiles_edge_mixture(alpha)
        if not model_boxes:
            print(f"No boxes found for {name}; skipping.")
            continue

        # Clamp index to available range
        idx = min(max(0, box_index), len(model_boxes) - 1)
        x0, y0, x1, y1 = model_boxes[idx]
        print(f"{name}: using box {idx} -> {(x0, y0, x1, y1)}")

        crop = image_np[y0:y1, x0:x1].copy()

        # Optionally annotate the crop with the model name
        cv2.putText(
            crop,
            name,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        crops.append(crop)

    if not crops:
        raise SystemExit("No crops were produced; cannot create comparison image.")

    try:
        combined = cv2.hconcat(crops)
    except cv2.error:
        # As a fallback, resize crops to the smallest height and then concat
        min_h = min(c.shape[0] for c in crops)
        resized = [cv2.resize(c, (int(c.shape[1] * (min_h / c.shape[0])), min_h), interpolation=cv2.INTER_AREA) for c in crops]
        combined = cv2.hconcat(resized)

    out_path = "boxes_comparison.jpg"
    cv2.imwrite(out_path, combined, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Saved horizontal comparison to {out_path}")
