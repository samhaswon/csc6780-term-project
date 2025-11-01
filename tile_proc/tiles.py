"""
Code for doing an initial inference, tiling the base image based on the results,
inferencing on the tiles, and combining the results.
"""
import time
from typing import List, Tuple, Sequence, Union, Any, Optional

import cv2
import numpy as np
from PIL import Image
from PIL.Image import Image as PILImage  # For typing

from sessions import Session, BiRefNetSession, TorchSession

# These are global variables here because they may need to be changed,
# but they realistically won't be.

TILE_SIZE: int = 512               # Refiner model expects 512x512x4, so don't change this.
OVERLAP: int = 64                  # 0 <= OVERLAP < TILE_SIZE

# Selection controls
MIN_POS_FRAC: float = 0.01         # min foreground fraction to consider a tile at all
BIN_THRESH: int = 0                # binarize: foreground = mask > BIN_THRESH
REQUIRE_NEGATIVE_OVERLAP: bool = True
MIN_NEG_FRAC: float = 0.01         # only for mixed tiles if REQUIRE_NEGATIVE_OVERLAP

# Edge-focused sampling + include all-white and all-zero tiles
EDGE_BAND_PX: int = 12             # pixels from boundary to count as "edge"
EDGE_TARGET_FRAC: float = 0.8      # target mix across buckets; will renormalize
POS_PURE_FRAC: float = 0.1         # include all-white tiles
NEG_PURE_FRAC: float = 0.1         # include all-zero tiles

# Optional grid jitter to reduce checkerboard bias (keep small)
JITTER_PX: int = 0                 # e.g., 4..8 for large images
RNG_SEED = 1234                    # set None to make it non-deterministic


def _gen_idxs(length: int, window: int, overlap: int, jitter: int) -> List[int]:
    """
    Build grid start indices covering [0, length), honoring overlap and small jitter.

    :param length: Axis length in pixels.
    :param window: Window size (tile side length).
    :param overlap: Overlap in pixels, 0 <= overlap < window.
    :param jitter: Start offset in pixels, clamped to [0, stride-1].
    :returns: List of starting indices.
    """
    if overlap < 0 or overlap >= window:
        raise ValueError("overlap must satisfy 0 <= overlap < window.")
    stride = max(1, window - overlap)
    jit = 0 if stride <= 1 else max(0, min(stride - 1, jitter))
    idxs = list(range(jit, max(1, length - window + 1), stride))
    if not idxs or idxs[-1] != length - window:
        idxs.append(max(0, length - window))
    return idxs


def select_tiles_edge_mixture(
    mask_gray: np.ndarray,
    tile_size: int = TILE_SIZE,
    overlap: int = OVERLAP,
    min_pos_frac: float = MIN_POS_FRAC,
    require_negative_overlap: bool = REQUIRE_NEGATIVE_OVERLAP,
    min_neg_frac: float = MIN_NEG_FRAC,
    edge_band_px: int = EDGE_BAND_PX,
    edge_target_frac: float = EDGE_TARGET_FRAC,
    pos_pure_frac: float = POS_PURE_FRAC,
    neg_pure_frac: float = NEG_PURE_FRAC,
    bin_thresh: int = BIN_THRESH,
    jitter_px: int = JITTER_PX,
    rng_seed: Optional[int] = RNG_SEED,
) -> List[Tuple[int, int, int, int]]:
    """
    Select mostly edge tiles, plus a controlled sample of all-white and all-zero tiles.

    Logic detail:
    - Binarize mask to foreground using mask_gray > bin_thresh.
    - Compute distance to boundary via distance transform on both sides, then min.
    - Classify tiles into three buckets:
      mixed (both fg and bg), pos_pure (all fg), neg_pure (all bg).
    - For mixed tiles only, enforce require_negative_overlap/min_neg_frac.
    - Rank mixed tiles by edge density (fraction within edge band), sample to target ratios.
    - Always allow pos_pure and neg_pure candidates into their buckets, then sample by target ratios.

    :returns: List of (x0, y0, x1, y1), x1/y1 are exclusive.
    """
    h, w = mask_gray.shape[:2]
    bin_mask = (mask_gray > bin_thresh).astype(np.uint8)

    dist_fg = cv2.distanceTransform(bin_mask, distanceType=cv2.DIST_L2, maskSize=3)
    dist_bg = cv2.distanceTransform(1 - bin_mask, distanceType=cv2.DIST_L2, maskSize=3)
    dist_to_b = np.minimum(dist_fg, dist_bg)

    rng = np.random.default_rng(rng_seed)
    jx = int(rng.integers(-jitter_px, jitter_px + 1)) if jitter_px > 0 else 0
    jy = int(rng.integers(-jitter_px, jitter_px + 1)) if jitter_px > 0 else 0

    xs = _gen_idxs(w, tile_size, overlap, jx)
    ys = _gen_idxs(h, tile_size, overlap, jy)

    total = tile_size * tile_size
    edge_tiles: list[tuple[Tuple[int, int, int, int], float]] = []
    pos_pure_tiles: list[Tuple[int, int, int, int]] = []
    neg_pure_tiles: list[Tuple[int, int, int, int]] = []

    for y0 in ys:
        for x0 in xs:
            x1, y1 = x0 + tile_size, y0 + tile_size
            tbin = bin_mask[y0:y1, x0:x1]
            if tbin.size != total:
                continue

            pos = int(np.count_nonzero(tbin))
            neg = total - pos
            if pos / total < min_pos_frac:
                continue

            if pos == total:
                # include all-white tiles regardless of negative-overlap requirement
                pos_pure_tiles.append((x0, y0, x1, y1))
                continue
            if neg == total:
                # include all-zero tiles as candidates
                neg_pure_tiles.append((x0, y0, x1, y1))
                continue

            # Mixed tile: optionally require some zero overlap and min_pos_frac
            if pos / total < min_pos_frac:
                continue

            edge_mask = dist_to_b[y0:y1, x0:x1] <= float(edge_band_px)
            edge_frac = float(np.count_nonzero(edge_mask)) / float(total)
            edge_tiles.append(((x0, y0, x1, y1), edge_frac))

    # Rank mixed tiles by edge density
    edge_tiles.sort(key=lambda t: t[1], reverse=True)
    edge_tiles_only = [t[0] for t in edge_tiles]

    # Determine target per-bucket counts relative to availability
    n_pool = len(edge_tiles_only) + len(pos_pure_tiles) + len(neg_pure_tiles)
    if n_pool == 0:
        return []

    weights = np.array([edge_target_frac, pos_pure_frac, neg_pure_frac], dtype=np.float64)
    weights = weights / max(1e-9, weights.sum())

    tgt_edge = min(len(edge_tiles_only), int(round(weights[0] * n_pool)))
    tgt_pos = min(len(pos_pure_tiles), int(round(weights[1] * n_pool)))
    tgt_neg = min(len(neg_pure_tiles), int(round(weights[2] * n_pool)))

    picked: list[Tuple[int, int, int, int]] = []
    picked.extend(edge_tiles_only[:tgt_edge])
    picked.extend(pos_pure_tiles[:tgt_pos])
    picked.extend(neg_pure_tiles[:tgt_neg])

    # Fill shortfall by priority edge -> pos -> neg
    short = n_pool - len(picked)
    if short > 0:
        leftovers = (
            edge_tiles_only[tgt_edge:]
            + pos_pure_tiles[tgt_pos:]
            + neg_pure_tiles[tgt_neg:]
        )
        if leftovers:
            picked.extend(leftovers[:short])

    # Dedup just in case
    seen = set()
    final = []
    for x0, y0, x1, y1 in picked:
        key = (x0, y0)
        if key not in seen:
            final.append((x0, y0, x1, y1))
            seen.add(key)
    return final

def extract_rgb_tiles(
        image: np.ndarray,
        boxes: Sequence[Tuple[int, int, int, int]]
) -> List[np.ndarray]:
    """
    Extract square RGBA tiles from an image given inclusive-exclusive boxes.

    :param image: BGRA image (H, W, 3) uint8.
    :param boxes: Iterable of (x0, y0, x1, y1), with x1/y1 exclusive.
    :returns: List of BGRA tiles, one per box, in the same order.
    """
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError("image must be a color image with shape (H, W, 4).")

    tiles: List[np.ndarray] = []
    h, w = image.shape[:2]
    for (x0, y0, x1, y1) in boxes:
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise ValueError(f"box out of bounds: {(x0, y0, x1, y1)}")
        tiles.append(image[y0:y1, x0:x1].copy())
    return tiles


def _make_blend_window(size: int, kind: str = "hann") -> np.ndarray:
    """
    Create a 2D separable blending window to reduce seams on overlap.

    :param size: Tile side length.
    :param kind: 'hann' or 'flat'. 'hann' smoothly tapers to edges.
    :returns: Float32 window of shape (size, size) with max 1.0, min > 0.0.
    """
    if kind == "flat":
        w1d = np.ones(size, dtype=np.float32)
    elif kind == "hann":
        # np.hanning gives 0 at ends. Add small floor to avoid zero divide on stitching borders.
        w1d = np.hanning(size).astype(np.float32)
        w1d = np.maximum(w1d, 1e-3)
    else:
        raise ValueError("Unsupported window kind. Use 'hann' or 'flat'.")

    w2d = np.outer(w1d, w1d).astype(np.float32)
    # Normalize so the center is 1.0. Keeps predictions in their original scale.
    w2d /= float(w2d.max(initial=1.0))
    return w2d


def _post_process(mask: np.ndarray) -> np.ndarray:
    """
    Morphs and blurs the mask to make it a bit better (generally speaking).
    :param mask: The mask to post-process.
    :return: The post-processed mask.
    """
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=1, sigmaY=1, borderType=cv2.BORDER_DEFAULT)
    return mask


def stitch_mask_tiles(
        mask_tiles: Sequence[np.ndarray],
        boxes: Sequence[Tuple[int, int, int, int]],
        out_shape: Union[Tuple[int, int], Tuple[int, ...]],
        window_kind: str = "hann"
) -> np.ndarray:
    """
    Combine per-tile masks into a full-sized mask with overlap-aware blending.

    Tiles are placed at the coordinates given by boxes. Overlaps are blended by a
    separable window (default Hann). If multiple tiles cover a pixel, their weighted
    average is used. The result is clipped to [0, 255] and returned as uint8.

    :param mask_tiles: List of mask tiles (each Ht, Wt) or (Ht, Wt, 1), uint8 or float.
    :param boxes: List of (x0, y0, x1, y1) matching mask_tiles order.
    :param out_shape: Target full mask shape (H, W).
    :param window_kind: 'hann' to reduce seams, 'flat' to just average.
    :returns: Full mask as uint8 with shape (H, W).
    """
    if len(mask_tiles) != len(boxes):
        raise ValueError("mask_tiles and boxes must have the same length.")

    H, W = out_shape
    acc = np.zeros((H, W), dtype=np.float32)
    wsum = np.zeros((H, W), dtype=np.float32)

    win_cache: dict[int, np.ndarray] = {}

    for tile, (x0, y0, x1, y1) in zip(mask_tiles, boxes):
        # Normalize tile to float32 2D
        if tile.ndim == 3 and tile.shape[2] == 1:
            tile2d = tile[:, :, 0]
        elif tile.ndim == 2:
            tile2d = tile
        else:
            # If a model returns 3-channel mask by mistake, convert to gray.
            if tile.ndim == 3 and tile.shape[2] == 3:
                tile2d = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            else:
                raise ValueError(f"Unexpected tile mask shape: {tile.shape}")

        tilef = tile2d.astype(np.float32)
        th, tw = tilef.shape[:2]

        if th != (y1 - y0) or tw != (x1 - x0):
            raise ValueError(
                f"Tile size mismatch. tile={tilef.shape} box={(y1 - y0, x1 - x0)}"
            )

        if th not in win_cache:
            win_cache[th] = _make_blend_window(th, window_kind)
        win = win_cache[th]

        # Accumulate weighted values
        acc[y0:y1, x0:x1] += tilef * win
        wsum[y0:y1, x0:x1] += win

    out = np.zeros((H, W), dtype=np.float32)
    np.divide(acc, np.maximum(wsum, 1e-6), out=out)  # safe division
    out_u8 = np.clip(out, 0.0, 255.0).astype(np.uint8)
    return _post_process(out_u8)


def refine_tiles_and_stitch(
        image: np.ndarray,
        boxes: Sequence[Tuple[int, int, int, int]],
        to_mask_tile_fn=None
) -> np.ndarray:
    """
    Example wrapper: extract RGB tiles, run them through your model, then stitch.

    The 'to_mask_tile_fn' is a placeholder that should map an RGB tile (BGR uint8)
    to a single-channel mask tile (uint8). The tiles must be returned in the same
    order as provided. If None, this uses a trivial luminance-based dummy mask.

    :param image: BGR image (H, W, 3) uint8.
    :param boxes: List of (x0, y0, x1, y1) with x1/y1 exclusive.
    :param to_mask_tile_fn: Callable[ndarray -> ndarray] or None.
    :returns: Full stitched mask (H, W) uint8.
    """
    rgb_tiles = extract_rgb_tiles(image, boxes)

    if to_mask_tile_fn is None:
        def _dummy(tile_bgr: np.ndarray) -> np.ndarray:
            # Placeholder: convert to gray as a stand-in for a model prediction.
            gray = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2GRAY)
            return gray
        to_mask_tile_fn = _dummy

    mask_tiles: List[np.ndarray] = [to_mask_tile_fn(t) for t in rgb_tiles]
    stitched = stitch_mask_tiles(mask_tiles, boxes, out_shape=image.shape[:2], window_kind="hann")
    return stitched


def segment_image(
        image: PILImage,
        base_session: Union[Session, BiRefNetSession, TorchSession],
        chunk_session: Union[Session, BiRefNetSession, TorchSession, Any],
        log_time: bool = True,
) -> np.ndarray:
    base_start = time.perf_counter()
    base_alpha = base_session.predict(image)
    base_end = time.perf_counter()
    image_np = np.array(image)

    # Make boxes
    boxed_start = time.perf_counter()
    test_img_boxes = select_tiles_edge_mixture(
        base_alpha,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        min_pos_frac=MIN_POS_FRAC,
        require_negative_overlap=REQUIRE_NEGATIVE_OVERLAP,
        min_neg_frac=MIN_NEG_FRAC,
        edge_band_px=EDGE_BAND_PX,
        edge_target_frac=EDGE_TARGET_FRAC,
        pos_pure_frac=POS_PURE_FRAC,
        neg_pure_frac=NEG_PURE_FRAC,
        bin_thresh=BIN_THRESH,
        jitter_px=JITTER_PX,
        rng_seed=RNG_SEED,
    )

    # Refine the boxes
    new_alpha = refine_tiles_and_stitch(
        image=np.dstack((image_np, base_alpha)),
        boxes=test_img_boxes,
        to_mask_tile_fn=lambda x: chunk_session.predict(Image.fromarray(x), convert_to="RGBA")
    )
    boxed_end = time.perf_counter()
    if log_time:
        print(f"Base time: {base_end - base_start:.4f}s\n"
              f"Boxed time: {boxed_end - boxed_start:.4f}s\n"
              f"Num boxes: {len(test_img_boxes)}\n")
    return new_alpha
