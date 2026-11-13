"""Model input and output contract, numpy and OpenCV only so it runs on the board (Python 3.10) and the PC.

Pre-processing reproduces the evaluator input (Ultralytics val LetterBox). Post-processing reproduces Ultralytics
predict NMS. decode_split reproduces Detect._inference for split-head models (raw boxes and scores outputs).
The C++ port in rk3588-deploy is checked against these functions.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

PROTOCOL_VERSION = "v1"  # evaluation protocol written into every result file
IMGSZ = 640
# Deployment defaults = Ultralytics predict (cfg/default.yaml: conf 0.25, iou 0.7, max_det 300,
# multi_label=False, max_nms 30000, max_wh 7680). AI quality uses val settings (conf 0.001) on the PC instead.
CONF = 0.25
IOU = 0.7
MAX_DET = 300
MAX_NMS = 30000
MAX_WH = 7680
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(images_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def preprocess(bgr: np.ndarray) -> tuple[np.ndarray, tuple]:
    """BGR image -> (1, 640, 640, 3) uint8 RGB, same pixels as the evaluator input."""
    h0, w0 = bgr.shape[:2]
    r = IMGSZ / max(h0, w0)
    img = bgr
    if r != 1:  # BaseDataset.load_image(rect_mode=True)
        img = cv2.resize(
            bgr, (min(math.ceil(w0 * r), IMGSZ), min(math.ceil(h0 * r), IMGSZ)), interpolation=cv2.INTER_LINEAR
        )
    h, w = img.shape[:2]
    # LetterBox(scaleup=False, center=True): r2 <= 1 and new_unpad == (w, h) after the resize above
    dw, dh = (IMGSZ - w) / 2, (IMGSZ - h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gain = min(h / h0, w / w0)
    return img[None], (gain, left, top, w0, h0)


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou <= iou_thres]
    return np.array(keep, dtype=int)


def postprocess(pred: np.ndarray, meta: tuple, normalized: bool, conf_thres: float = CONF) -> np.ndarray:
    """Raw (1, 4 + nc, anchors) -> (N, 6) [x1, y1, x2, y2, conf, cls] in original image pixels."""
    x = pred[0].T.astype(np.float32)  # (anchors, 4 + nc)
    if normalized:  # Ultralytics INT8 RKNN exports divide boxes by the input size
        x[:, :4] *= IMGSZ
    scores = x[:, 4:]
    cls = scores.argmax(1)
    conf = scores[np.arange(len(cls)), cls]
    keep = conf > conf_thres
    x, conf, cls = x[keep], conf[keep], cls[keep]
    if len(x) > MAX_NMS:
        top = conf.argsort()[::-1][:MAX_NMS]
        x, conf, cls = x[top], conf[top], cls[top]
    boxes = np.empty((len(x), 4), dtype=np.float32)
    boxes[:, 0] = x[:, 0] - x[:, 2] / 2
    boxes[:, 1] = x[:, 1] - x[:, 3] / 2
    boxes[:, 2] = x[:, 0] + x[:, 2] / 2
    boxes[:, 3] = x[:, 1] + x[:, 3] / 2
    idx = nms_xyxy(boxes + cls[:, None] * MAX_WH, conf, IOU)[:MAX_DET]
    boxes, conf, cls = boxes[idx], conf[idx], cls[idx]

    gain, padw, padh, w0, h0 = meta  # scale back to the original image (ops.scale_boxes)
    boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - padw) / gain).clip(0, w0)
    boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - padh) / gain).clip(0, h0)
    return np.concatenate([boxes, conf[:, None], cls[:, None].astype(np.float32)], axis=1)


def postprocess_raw(outs, meta: tuple, normalized: bool, split: bool = False, conf_thres: float = CONF) -> np.ndarray:
    """outs is the list returned by RKNNLite.inference. split: the model has two raw outputs (boxes, scores),
    decoded in numpy first (boxes in pixels), then the usual postprocess."""
    if split:
        boxes = np.asarray(outs[0], dtype=np.float32)
        scores = np.asarray(outs[1], dtype=np.float32)
        return postprocess(decode_split(boxes, scores), meta, False, conf_thres)
    return postprocess(outs[0], meta, normalized, conf_thres)


def anchors_and_strides(imgsz: int = 640, strides: tuple[int, ...] = (8, 16, 32)) -> tuple[np.ndarray, np.ndarray]:
    """Anchor points (A x 2, cell centre offset 0.5) and strides (A,), levels concatenated in row-major order
    like make_anchors."""
    pts, st = [], []
    for s in strides:
        n = imgsz // s
        sy, sx = np.meshgrid(np.arange(n) + 0.5, np.arange(n) + 0.5, indexing="ij")
        pts.append(np.stack((sx, sy), -1).reshape(-1, 2))
        st.append(np.full(n * n, s, dtype=np.float32))
    return np.concatenate(pts).astype(np.float32), np.concatenate(st)


def decode_split(
    boxes: np.ndarray,
    scores: np.ndarray,
    imgsz: int = 640,
    reg_max: int | None = None,
    strides: tuple[int, ...] = (8, 16, 32),
) -> np.ndarray:
    """boxes (B, 4*reg_max, A) DFL logits and scores (B, nc, A) class logits -> (B, 4 + nc, A): xywh in pixels and
    sigmoid scores.

    reg_max defaults to boxes.shape[1] // 4. With reg_max 1 there is no DFL (Detect.dfl is Identity) and boxes already
    hold ltrb distances.
    """
    b, ch, a = boxes.shape
    reg_max = ch // 4 if reg_max is None else reg_max
    assert ch == 4 * reg_max, f"{ch} box channels, 4 * reg_max = {4 * reg_max}"
    pts, st = anchors_and_strides(imgsz, strides)
    assert pts.shape[0] == a, f"{a} anchors in the output, {pts.shape[0]} expected for imgsz {imgsz}"
    if reg_max == 1:
        dist = boxes.astype(np.float32).reshape(b, 4, a)  # ltrb distances
    else:
        x = boxes.astype(np.float32).reshape(b, 4, reg_max, a)
        x = np.exp(x - x.max(axis=2, keepdims=True))
        bins = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1)
        dist = (x / x.sum(axis=2, keepdims=True) * bins).sum(axis=2)  # (B, 4, A) ltrb
    lt, rb = dist[:, :2], dist[:, 2:]
    ap = pts.T[None]  # (1, 2, A)
    x1y1, x2y2 = ap - lt, ap + rb
    xywh = np.concatenate(((x1y1 + x2y2) / 2, x2y2 - x1y1), axis=1) * st[None, None, :]
    return np.concatenate((xywh, 1.0 / (1.0 + np.exp(-scores.astype(np.float32)))), axis=1)
