"""
Quick, messy test for A100 inference.
"""
import csv
from pathlib import Path
import time
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage
import torch

# Input Model
from sessions import BiRefNetTorchSession
# Refiner Model
from sessions.torch_session.u2net import U2NETP
from tile_proc.tiles import select_tiles_edge_mixture, extract_rgb_tiles, stitch_mask_tiles


# One or two GPUs.
# DEVICE 0 takes both the refiner and base model.
# DEVICE 1 takes only the refiner
DEVICE0 = "cuda:0"  # cuda for one or cuda:0 for two
DEVICE0_BATCH_SIZE = 8
DEVICE1 = "cuda:1"  # None or "cuda:1"
DEVICE1_BATCH_SIZE = 8

TEST_PASSES = 20


def load_refiner(device: str) -> torch.nn.Module:
    """Loads the refiner model."""
    net = U2NETP(4, 1)
    model_path = "models/u2netp_chunks.pth"
    if torch.cuda.is_available() and device != "cpu":
        net.load_state_dict(
            torch.load(model_path, weights_only=False)
        )
        net.to(device)
    else:
        net.load_state_dict(
            torch.load(
                model_path,
                map_location=torch.device(device),
                weights_only=False
            )
        )
    net.eval()
    # More JIT
    net = torch.compile(net)
    return net


def norm_one_output(mask: np.ndarray) -> np.ndarray:
    """
    Normalize one output chunk
    :param mask: The mask to process.
    """
    ma = np.max(mask)
    mi = np.min(mask)
    if (ma != mi and (ma != 1.0 and mi != 0.0)) and mi < 0.98:
        mask = (mask - mi) / (ma - mi + 1E-8)
    mask = np.squeeze(mask)
    return (mask * 255).astype(np.uint8)


def run_refiner_on_tiles(
        refiner: torch.nn.Module,
        tiles: list[np.ndarray],
        device: str,
        batch_size: int
) -> list[np.ndarray]:
    """
    Run a refiner on a list of tiles.

    :param refiner: The refiner model.
    :param tiles: List of tiles (H, W, 4) uint8.
    :param device: Torch device string.
    :param batch_size: Batch size to use.
    :return: List of uint8 mask tiles (H, W).
    """
    mask_tiles: list[np.ndarray] = []

    if not tiles:
        return mask_tiles

    for start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[start:start + batch_size]

        batch_arr = np.stack(batch_tiles).astype(np.float32) / 255.0
        batch_tensor = torch.from_numpy(batch_arr)
        batch_tensor = batch_tensor.permute(0, 3, 1, 2).contiguous()
        batch_tensor = batch_tensor.to(device, non_blocking=True)

        with torch.inference_mode():
            out = refiner(batch_tensor)
            if isinstance(out, (tuple, list)):
                out = out[0]

            out_np = out.detach().cpu().numpy()

        for b in range(out_np.shape[0]):
            mask = out_np[b, 0]
            mask_tiles.append(norm_one_output(mask))

    return mask_tiles


def refine_tiles_dual(
        refiner0: torch.nn.Module,
        refiner1: torch.nn.Module,
        tiles: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Run tiles through one or two refiners, depending on configuration.

    Uses DEVICE0/DEVICE1 and DEVICE0_BATCH_SIZE/DEVICE1_BATCH_SIZE.

    :param refiner0: First refiner.
    :param refiner1: Second refiner (may be same object as refiner0).
    :param tiles: List of tiles to process.
    :return: List of mask tiles in the same order as input.
    """
    if not tiles:
        return []

    # Single device or aliased model: just run once.
    if DEVICE0 == DEVICE1 or refiner0 is refiner1:
        return run_refiner_on_tiles(refiner0, tiles, DEVICE0, DEVICE0_BATCH_SIZE)

    # Two different devices, same weights. Split tiles in half by count.
    mid = len(tiles) // 2
    tiles0 = tiles[:mid]
    tiles1 = tiles[mid:]

    masks0 = run_refiner_on_tiles(refiner0, tiles0, DEVICE0, DEVICE0_BATCH_SIZE)
    masks1 = run_refiner_on_tiles(refiner1, tiles1, DEVICE1, DEVICE1_BATCH_SIZE)

    return masks0 + masks1


def write_times_csv(path: str | Path, time_list: list[tuple[float, float, float]]) -> None:
    """
    Write timing results to a CSV file.

    :param path: Output CSV path.
    :param time_list: List of (base, patch, total) timing tuples.
    """
    path = Path(path)

    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["base", "patch", "total"])
        writer.writerows(time_list)


def infer_once(
        base_session: BiRefNetTorchSession,
        refiner0: torch.nn.Module,
        refiner1: torch.nn.Module,
        img: PILImage
):
    """
    Do one inference.

    :param base_session: The base session to use.
    :param refiner0: The first refiner to use.
    :param refiner1: The second refiner to use.
    :param img: The image to process.
    :return: The combined result mask and times (base, patch, total).
    """
    full_start = time.perf_counter()
    arr = np.array(img)
    base_start = time.perf_counter()
    base_alpha = base_session.remove(img, mask_only=True)
    base_end = time.perf_counter()

    # Generate boxes
    boxes = select_tiles_edge_mixture(base_alpha)
    # Make tiles from the boxes
    tiles = extract_rgb_tiles(np.dstack((arr, base_alpha)), boxes)

    # Refine tiles
    refine_start = time.perf_counter()
    mask_tiles = refine_tiles_dual(refiner0, refiner1, tiles)
    refine_end = time.perf_counter()

    stitched = stitch_mask_tiles(
        mask_tiles,
        boxes,
        out_shape=arr.shape[:2],
        window_kind="hann"
    )
    full_end = time.perf_counter()
    return stitched, base_end - base_start, refine_end - refine_start, full_end - full_start


def main():
    """Main function"""
    global DEVICE0, DEVICE1
    if DEVICE1 is None:
        DEVICE1 = DEVICE0

    # Load models
    print(f"Loading models to {DEVICE0} and {DEVICE1}...")
    base_session = BiRefNetTorchSession(half_precision=True, device=DEVICE0)
    # Tell Torch to compile it later (eager mode means it'll take a minute later)
    base_session.compile()
    if DEVICE0 != DEVICE1:
        refiner0, refiner1 = load_refiner(DEVICE0), load_refiner(DEVICE1)
    else:
        refiner0 = load_refiner(DEVICE0)
        # Alias for later
        refiner1 = refiner0
    img = Image.open("./test_inputs/test.jpg")
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Warm up JIT stuff
    print("Running warm-up inference...")
    warm_up_start = time.perf_counter()
    mask, *_ = infer_once(base_session, refiner0, refiner1, img)
    warm_up_end = time.perf_counter()
    print(f"Warm-up took {warm_up_end - warm_up_start:.4f} seconds.")
    img_arr = np.dstack((img_arr, mask))
    cv2.imwrite("test.png", img_arr)

    if TEST_PASSES > 0:
        time_list: List[Tuple[float, float, float]] = []
        print("Running test")
        for _ in range(TEST_PASSES):
            _, *times = infer_once(base_session, refiner0, refiner1, img)
            time_list.append(times)
        write_times_csv("a100_test.csv", time_list)
        print("Done")


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    if "cuda" in DEVICE0:
        torch.backends.cudnn.conv.fp32_precision = 'tf32'
        torch.backends.fp32_precision = "tf32"
        print("Using TF32 for some calculations")
    main()
