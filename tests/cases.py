"""Deterministic inputs shared by make_fixtures.py (legacy code) and test_model_io.py (this package)."""

import hashlib
from pathlib import Path

import numpy as np

FIXTURES = Path(__file__).resolve().parent / "fixtures"
DUMP = FIXTURES / "model_io_dump.npz"  # first images of a real INT8 board dump, written by make_fixtures.py
EXPECTED = FIXTURES / "model_io_expected.npz"
BOARD_ENV_EXPECTED = FIXTURES / "board_env_expected.json"

# (429, 640) and (640, 429) give a padding of 105.5 per side, where round(d - 0.1) and round(d) differ
IMAGE_SHAPES = [(640, 640), (480, 640), (1080, 1920), (200, 300), (1000, 750), (641, 639), (429, 640), (640, 429)]
# (gain, pad_left, pad_top, w0, h0) as returned by preprocess: a square 640 image and a 1280x720 frame
METAS = [(1.0, 0, 0, 640, 640), (0.5, 0, 140, 1280, 720)]
CONFS = [0.25, 0.001]
# np.exp on float32 differs by a few ulp between x86 and aarch64 numpy builds (multiply and divide do not). Measured on
# the board 2026-11-13: decode_split boxes differ by up to 1.2e-4 px, scores by up to 1.2e-7. Outputs that go through
# exp (decode_split, split postprocess) are compared with these absolute tolerances, every other output exactly.
BOX_ATOL = 1e-3  # pixels
SCORE_ATOL = 1e-6
LIST_FILES = ["b.jpg", "a.PNG", "c.jpeg", "d.bmp", "e.txt", "f.JPG", "g.webp", "h"]


def latency_lists() -> list[list[float]]:
    rng = np.random.default_rng(4)
    return [[5.0], [1.0, 2.0], list(rng.gamma(4, 2.5, 1000)), list(rng.uniform(3, 30, 77))]


def images() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in IMAGE_SHAPES]


def synthetic_pred(normalized: bool) -> np.ndarray:
    """(1, 4 + 5, 8400) with overlapping boxes, so NMS has work to do."""
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 640, (2, 8400))
    wh = rng.uniform(10, 200, (2, 8400))
    boxes = np.concatenate([xy, wh]) / (640 if normalized else 1)
    scores = rng.uniform(0, 1, (5, 8400)) ** 3
    return np.concatenate([boxes, scores])[None].astype(np.float32)


def threshold_pred() -> np.ndarray:
    """Few candidates, separate boxes: 20 anchors score exactly CONF, 10 score 0.3, the rest 0.1 (under CONF)."""
    pred = np.zeros((1, 9, 8400), dtype=np.float32)
    pred[0, 0] = np.arange(8400) % 80 * 8 + 4  # cx: an 80 x 105 grid of 6 px boxes, no overlap
    pred[0, 1] = np.arange(8400) // 80 * 6 + 3
    pred[0, 2:4] = 6
    pred[0, 4:] = 0.1
    pred[0, 4, :20] = CONFS[0]
    pred[0, 5, 20:30] = 0.3
    return pred


def split_outputs(reg_max: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2 + reg_max)
    if reg_max == 1:
        boxes = rng.uniform(0, 10, (1, 4, 8400))
    else:
        boxes = rng.normal(0, 4, (1, 4 * reg_max, 8400))
    scores = rng.normal(-3, 3, (1, 5, 8400))
    return boxes.astype(np.float32), scores.astype(np.float32)


def nms_inputs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    xy = rng.uniform(0, 600, (500, 2))
    boxes = np.concatenate([xy, xy + rng.uniform(5, 120, (500, 2))], axis=1).astype(np.float32)
    return boxes, rng.uniform(0, 1, 500).astype(np.float32)


def dump_preds() -> list[np.ndarray]:
    """Dense (1, 9, 8400) outputs rebuilt from the sparse dump (anchors under the dump threshold stay zero)."""
    d = np.load(DUMP)
    preds, start = [], 0
    for n in d["counts"]:
        pred = np.zeros((1, *d["shape"]), dtype=np.float32)
        pred[0][:, d["idx"][start : start + n]] = d["vals"][start : start + n].T
        preds.append(pred)
        start += n
    return preds


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(f"{a.dtype}{a.shape}".encode() + np.ascontiguousarray(a).tobytes()).hexdigest()
