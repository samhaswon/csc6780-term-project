"""
Create an animation that visualizes how grid boxes are selected for refinement.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from box_demo import colorize_magenta, draw_boxes
from sessions import BiRefNetSession
from tile_proc.tiles import (
    BIN_THRESH,
    EDGE_BAND_PX,
    EDGE_TARGET_FRAC,
    JITTER_PX,
    MIN_NEG_FRAC,
    MIN_POS_FRAC,
    NEG_PURE_FRAC,
    OVERLAP,
    POS_PURE_FRAC,
    REQUIRE_NEGATIVE_OVERLAP,
    RNG_SEED,
    TILE_SIZE,
    _gen_idxs,
    select_tiles_edge_mixture,
)


Color = Tuple[int, int, int]
Box = Tuple[int, int, int, int]

PENDING_COLOR: Color = (0, 255, 255)   # Yellow
REJECTED_COLOR: Color = (0, 0, 255)    # Red
SELECTED_COLOR: Color = (0, 200, 0)    # Green
HIGHLIGHT_COLOR: Color = (255, 255, 255)
MAX_SIDE_PX = 2000


@dataclass
class BoxTrace:
    """
    Record of how a tile was handled while running select_tiles_edge_mixture.
    """
    index: int
    box: Box
    category: str
    pos_fraction: float
    reason: str
    edge_fraction: Optional[float] = None
    selected: bool = False
    stage: Optional[str] = None  # e.g., "edge_primary", "pos_leftover"


def trace_tile_selection(
    mask_gray: np.ndarray,
    tile_size: int = TILE_SIZE,
    overlap: int = OVERLAP,
    min_pos_frac: float = MIN_POS_FRAC,
    edge_band_px: int = EDGE_BAND_PX,
    edge_target_frac: float = EDGE_TARGET_FRAC,
    pos_pure_frac: float = POS_PURE_FRAC,
    neg_pure_frac: float = NEG_PURE_FRAC,
    bin_thresh: int = BIN_THRESH,
    jitter_px: int = JITTER_PX,
    rng_seed: Optional[int] = RNG_SEED,
) -> Tuple[List[BoxTrace], List[Box]]:
    """
    Mirror select_tiles_edge_mixture but also emit per-tile metadata for animation.
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
    traces: List[BoxTrace] = []

    edge_tiles: List[Tuple[Box, float]] = []
    pos_pure_tiles: List[Box] = []
    neg_pure_tiles: List[Box] = []

    for y0 in ys:
        for x0 in xs:
            x1, y1 = x0 + tile_size, y0 + tile_size
            box: Box = (x0, y0, x1, y1)
            trace = BoxTrace(
                index=len(traces),
                box=box,
                category="pending",
                pos_fraction=0.0,
                reason="",
            )
            tbin = bin_mask[y0:y1, x0:x1]
            if tbin.size != total:
                trace.category = "skipped"
                trace.reason = "Tile exceeds image bounds."
                traces.append(trace)
                continue

            pos = int(np.count_nonzero(tbin))
            neg = total - pos
            pos_frac = pos / total
            trace.pos_fraction = pos_frac

            if pos_frac < min_pos_frac:
                trace.category = "rejected"
                trace.reason = f"Foreground fraction {pos_frac:.3f} < min {min_pos_frac:.3f}"
                traces.append(trace)
                continue

            if pos == total:
                trace.category = "pos_pure"
                trace.reason = "All mask pixels are foreground."
                pos_pure_tiles.append(box)
                traces.append(trace)
                continue

            if neg == total:
                trace.category = "neg_pure"
                trace.reason = "All mask pixels are background."
                neg_pure_tiles.append(box)
                traces.append(trace)
                continue

            edge_mask = dist_to_b[y0:y1, x0:x1] <= float(edge_band_px)
            edge_frac = float(np.count_nonzero(edge_mask)) / float(total)
            trace.category = "edge"
            trace.edge_fraction = edge_frac
            trace.reason = f"Edge density {edge_frac:.3f}"
            edge_tiles.append((box, edge_frac))
            traces.append(trace)

    # Reproduce the selection logic
    edge_tiles.sort(key=lambda t: t[1], reverse=True)
    edge_boxes = [t[0] for t in edge_tiles]
    edge_rank: Dict[Box, int] = {box: idx for idx, box in enumerate(edge_boxes)}
    pos_rank: Dict[Box, int] = {box: idx for idx, box in enumerate(pos_pure_tiles)}
    neg_rank: Dict[Box, int] = {box: idx for idx, box in enumerate(neg_pure_tiles)}

    n_pool = len(edge_boxes) + len(pos_pure_tiles) + len(neg_pure_tiles)
    if n_pool == 0:
        return traces, []

    weights = np.array([edge_target_frac, pos_pure_frac, neg_pure_frac], dtype=np.float64)
    weights = weights / max(1e-9, weights.sum())

    tgt_edge = min(len(edge_boxes), int(round(weights[0] * n_pool)))
    tgt_pos = min(len(pos_pure_tiles), int(round(weights[1] * n_pool)))
    tgt_neg = min(len(neg_pure_tiles), int(round(weights[2] * n_pool)))

    picked: List[Box] = []
    stage_map: Dict[Box, str] = {}

    for box in edge_boxes[:tgt_edge]:
        picked.append(box)
        stage_map[box] = "edge_primary"

    for box in pos_pure_tiles[:tgt_pos]:
        picked.append(box)
        stage_map[box] = "pos_primary"

    for box in neg_pure_tiles[:tgt_neg]:
        picked.append(box)
        stage_map[box] = "neg_primary"

    short = n_pool - len(picked)
    if short > 0:
        leftovers = (
            edge_boxes[tgt_edge:]
            + pos_pure_tiles[tgt_pos:]
            + neg_pure_tiles[tgt_neg:]
        )
        leftovers = leftovers[:short]
        for box in leftovers:
            picked.append(box)
            stage_map.setdefault(box, "leftover")

    seen = set()
    final_boxes: List[Box] = []
    for box in picked:
        key = (box[0], box[1])
        if key in seen:
            continue
        seen.add(key)
        final_boxes.append(box)

    trace_map: Dict[Tuple[int, int], BoxTrace] = {
        (t.box[0], t.box[1]): t for t in traces if t.category != "skipped"
    }
    for box in final_boxes:
        trace = trace_map.get((box[0], box[1]))
        if trace:
            trace.selected = True
            trace.stage = stage_map.get(box, "dedup")

    # Annotate reasons for candidates that were dropped.
    for trace in traces:
        if trace.category in {"edge", "pos_pure", "neg_pure"} and not trace.selected:
            rank = None
            if trace.category == "edge":
                rank = edge_rank.get(trace.box)
            elif trace.category == "pos_pure":
                rank = pos_rank.get(trace.box)
            elif trace.category == "neg_pure":
                rank = neg_rank.get(trace.box)
            trace.reason = f"{trace.reason}; rank {rank}, not selected after down-selection."

    return traces, final_boxes


def draw_text_overlay(frame: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """
    Draw a semi-transparent text block near the bottom-left corner.
    """
    if not lines:
        return frame
    out = frame.copy()
    margin = 10
    line_height = 28
    width = int(out.shape[1] * 0.65)
    height = line_height * len(lines) + margin * 2
    overlay = out.copy()
    cv2.rectangle(
        overlay,
        (margin // 2, out.shape[0] - height - margin // 2),
        (width, out.shape[0] - margin // 2),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, dst=out)

    y = out.shape[0] - height + margin
    for line in lines:
        cv2.putText(
            out,
            line,
            (margin, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        y += line_height
    return out


def render_boxes(
    base: np.ndarray,
    ordered_boxes: Sequence[Box],
    state_map: Dict[Box, str],
    highlight: Optional[Box] = None,
    text_lines: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Render the current mask with colored box overlays.
    """
    frame = base.copy()
    for box in ordered_boxes:
        color = PENDING_COLOR
        if state_map.get(box) == "selected":
            color = SELECTED_COLOR
        elif state_map.get(box) == "rejected":
            color = REJECTED_COLOR
        x0, y0, x1, y1 = box
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), color, 8)
    if highlight is not None:
        x0, y0, x1, y1 = highlight
        cv2.rectangle(frame, (x0, y0), (x1 - 1, y1 - 1), HIGHLIGHT_COLOR, 16)
    if text_lines:
        frame = draw_text_overlay(frame, text_lines)
    return frame


def fade_frames(start: np.ndarray, end: np.ndarray, steps: int) -> List[np.ndarray]:
    """
    Utility: fade from start to end over the requested number of steps.
    """
    frames: List[np.ndarray] = []
    for i in range(1, steps + 1):
        alpha = i / float(steps)
        blended = cv2.addWeighted(start, 1.0 - alpha, end, alpha, 0)
        frames.append(blended)
    return frames


def resize_to_max_side(image: np.ndarray, max_side: int = MAX_SIDE_PX) -> np.ndarray:
    """
    Resize image so its longest side is at most max_side while preserving aspect ratio.
    """
    h, w = image.shape[:2]
    current_max = max(h, w)
    if current_max <= max_side:
        return image.copy()
    ratio = max_side / float(current_max)
    new_w = max(1, int(round(w * ratio)))
    new_h = max(1, int(round(h * ratio)))
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def describe_trace(trace: BoxTrace) -> List[str]:
    """
    Build textual info for display for a given box trace.
    """
    header = f"Box #{trace.index + 1}: {trace.box}"
    status = "SELECTED" if trace.selected else "NOT SELECTED"
    category = trace.category.replace("_", " ").title()
    lines = [header, f"{category} -> {status}"]
    if trace.edge_fraction is not None:
        lines.append(f"Edge density: {trace.edge_fraction:.3f}")
    lines.append(trace.reason or "Evaluating tile")
    if trace.selected and trace.stage:
        lines.append(f"Stage: {trace.stage}")
    return lines


def build_animation_frames(
    image_bgr: np.ndarray,
    mask_gray: np.ndarray,
    traces: Sequence[BoxTrace],
    final_boxes: Sequence[Box],
    fps: int,
    max_side: int = MAX_SIDE_PX,
) -> List[np.ndarray]:
    """
    Build the complete set of frames for the requested animation.
    """
    frames: List[np.ndarray] = []

    def append_frame(img: np.ndarray) -> None:
        resized = resize_to_max_side(img, max_side=max_side)
        frames.append(resized)

    def hold(img: np.ndarray, count: int) -> None:
        for _ in range(max(0, count)):
            append_frame(img)

    mask_bgr = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)
    colorized = colorize_magenta(image_bgr, mask_gray)

    hold(image_bgr, fps // 2)
    for frame in fade_frames(image_bgr, colorized, fps):
        append_frame(frame)
    hold(colorized, fps // 4)
    for frame in fade_frames(colorized, mask_bgr, fps):
        append_frame(frame)
    hold(mask_bgr, fps // 4)

    ordered_boxes = [trace.box for trace in traces]
    state_map: Dict[Box, str] = {box: "pending" for box in ordered_boxes}

    intro_frame = render_boxes(mask_bgr, ordered_boxes, state_map, text_lines=["All candidate tiles"])
    hold(intro_frame, fps // 3)

    for trace in traces:
        # Highlight evaluation
        highlight_frame = render_boxes(
            mask_bgr,
            ordered_boxes,
            state_map,
            highlight=trace.box,
            text_lines=describe_trace(trace),
        )
        hold(highlight_frame, max(2, fps // 10))

        # Update state using final decision
        state_map[trace.box] = "selected" if trace.selected else "rejected"
        settled = render_boxes(
            mask_bgr,
            ordered_boxes,
            state_map,
            text_lines=describe_trace(trace),
        )
        hold(settled, max(2, fps // 10))

    final_state = render_boxes(mask_bgr, ordered_boxes, state_map, text_lines=["Final tile selection"])
    hold(final_state, fps // 2)

    boxed_image = draw_boxes(image_bgr, final_boxes, box_line_thickness=4)
    for frame in fade_frames(final_state, boxed_image, fps):
        append_frame(frame)
    hold(boxed_image, fps // 2)

    return frames


def write_video(frames: Sequence[np.ndarray], output_path: Path, fps: int) -> None:
    """
    Save frames to disk as an MP4 video.
    """
    if not frames:
        raise ValueError("No frames to write.")
    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def load_mask(image_path: Path, model_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the source image and run BiRefNet to produce the mask.
    """
    image = Image.open(image_path)
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    session = BiRefNetSession(str(model_path))
    alpha_mask = session.predict(image)
    del session
    return image_bgr, alpha_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a grid-box selection animation.")
    parser.add_argument("--image", type=Path, default=Path("test_inputs/test.jpg"), help="Input image path.")
    parser.add_argument("--model", type=Path, default=Path("models/birefnet.onnx"), help="BiRefNet ONNX model path.")
    parser.add_argument("--output", type=Path, default=Path("plots/box_animation.mp4"), help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=30, help="Output frame rate.")
    parser.add_argument("--max-side", type=int, default=MAX_SIDE_PX, help="Clamp frames so the longest edge <= this value.")
    args = parser.parse_args()

    image_bgr, mask = load_mask(args.image, args.model)

    traces, traced_boxes = trace_tile_selection(
        mask,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        min_pos_frac=MIN_POS_FRAC,
        edge_band_px=EDGE_BAND_PX,
        edge_target_frac=EDGE_TARGET_FRAC,
        pos_pure_frac=POS_PURE_FRAC,
        neg_pure_frac=NEG_PURE_FRAC,
        bin_thresh=BIN_THRESH,
        jitter_px=JITTER_PX,
        rng_seed=RNG_SEED,
    )

    official_boxes = select_tiles_edge_mixture(
        mask,
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

    if official_boxes != traced_boxes:
        raise RuntimeError("Trace results do not match select_tiles_edge_mixture output.")

    frames = build_animation_frames(
        image_bgr,
        mask,
        traces,
        traced_boxes,
        fps=args.fps,
        max_side=args.max_side,
    )
    write_video(frames, args.output, args.fps)
    print(f"Wrote animation to {args.output}")


if __name__ == "__main__":
    main()
