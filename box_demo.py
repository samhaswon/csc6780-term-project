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


def colorize_magenta(bgr_image, alpha_map):
    """
    Apply a magenta colorize effect similar to GIMP.

    The grayscale alpha_map controls how strongly the colorization is applied.
    0 leaves the pixel unchanged, 255 applies full colorization.

    :param bgr_image: Input image in BGR, uint8, shape (H, W, 3)
    :param alpha_map: Grayscale control map, uint8, shape (H, W)
    :return: Colorized BGR image, uint8
    """
    if bgr_image.dtype != np.uint8 or alpha_map.dtype != np.uint8:
        raise ValueError("Inputs must be uint8.")
    if bgr_image.ndim != 3 or bgr_image.shape[2] != 3:
        raise ValueError("bgr_image must have shape (H, W, 3).")
    if alpha_map.ndim != 2:
        raise ValueError("alpha_map must have shape (H, W).")
    if bgr_image.shape[:2] != alpha_map.shape:
        raise ValueError("Dimensions must match.")

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    magenta_hue = 300 / 2.0
    hsv[:, :, 0] = magenta_hue
    hsv[:, :, 1] = 255

    colorized = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    alpha = (alpha_map.astype(np.float32) / 255.0)[..., None]

    blended = (
        bgr_image.astype(np.float32) * (1.0 - alpha)
        + colorized.astype(np.float32) * alpha
    )

    return blended.clip(0, 255).astype(np.uint8)


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

    chunk_session = Session("models/u2netp_chunks.onnx")

    # Extract the same-index box from each alpha and concatenate horizontally
    box_index = 0  # change this to pick a different box index

    crops = []
    crops_chunks = []
    crops_alphas = []
    chunk_alphas = []

    model_names = ["u2netp", "u2net", "birefnet"]
    alphas = [u2netp_alpha, u2net_alpha, birefnet_alpha]

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
        crop_copy = crop.copy()
        crops_alphas.append(alpha[y0:y1, x0:x1].copy())
        chunk_alpha = chunk_session.predict(
            Image.fromarray(
                np.dstack(
                    (cv2.cvtColor(crop_copy, cv2.COLOR_BGR2RGB), crops_alphas[-1])
                )
            ),
            convert_to="RGBA",  # RGBA for the chunk session
        )
        chunk_alphas.append(chunk_alpha)

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
        cv2.putText(
            crop_copy,
            name + " + refiner",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )
        crops_chunks.append(crop_copy)

    if not crops:
        raise SystemExit("No crops were produced; cannot create comparison image.")

    combined = cv2.hconcat(crops)
    combined_alpha = cv2.hconcat(crops_alphas)
    chunk_combined = cv2.hconcat(crops_chunks)
    chunk_combined_alpha = cv2.hconcat(chunk_alphas)

    base_img = cv2.vconcat([combined, chunk_combined])
    base_alpha = cv2.vconcat([combined_alpha, chunk_combined_alpha])

    colorized = colorize_magenta(base_img, base_alpha)

    out_path = "boxes_comparison.jpg"
    cv2.imwrite(out_path, colorized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Saved horizontal comparison to {out_path}")
