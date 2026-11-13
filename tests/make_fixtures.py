"""Write the expected outputs of model_io and board_env.summarize from the legacy code at tag legacy-final.

Run once with the legacy interpreter, never imported by the tests:
    <legacy>/.venv/bin/python tests/make_fixtures.py <legacy>
<legacy> is the legacy checkout (HEAD at legacy-final, ai/board clean). Its baseline INT8 board dump
(experiments/runs/ppe5_yolov8n_baseline_100ep_s0/weights/npu_dump_int8_css_test.npz, git-ignored) gives real outputs.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cases
import numpy as np

N_DUMP_IMAGES = 3
DUMP_SRC = "experiments/runs/ppe5_yolov8n_baseline_100ep_s0/weights/npu_dump_int8_css_test.npz"


def check_legacy(legacy: Path) -> None:
    tag = subprocess.check_output(["git", "-C", str(legacy), "describe", "--exact-match", "--tags", "HEAD"], text=True)
    assert tag.strip() == "legacy-final", f"legacy HEAD is at {tag.strip()}, not legacy-final"
    dirty = subprocess.check_output(["git", "-C", str(legacy), "status", "--porcelain", "--", "ai/board"], text=True)
    assert not dirty, f"legacy ai/board has local changes:\n{dirty}"


def write_dump(legacy: Path) -> None:
    d = np.load(legacy / DUMP_SRC)
    counts = d["counts"][:N_DUMP_IMAGES]
    n = int(counts.sum())
    keep = {"names": d["names"][:N_DUMP_IMAGES], "counts": counts, "idx": d["idx"][:n], "vals": d["vals"][:n]}
    np.savez_compressed(cases.DUMP, **keep, shape=d["shape"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy", type=Path)
    legacy = parser.parse_args().legacy.resolve()
    check_legacy(legacy)
    sys.path.insert(0, str(legacy))
    from ai.board import common as old  # legacy-final:ai/board/common.py
    from ai.board import split_decode as old_split  # legacy-final:ai/board/split_decode.py

    cases.FIXTURES.mkdir(exist_ok=True)
    write_dump(legacy)
    out = {}
    for i, img in enumerate(cases.images()):
        x, meta = old.preprocess(img)
        out[f"pre{i}_sha"] = np.array(cases.digest(x))
        out[f"pre{i}_meta"] = np.array(meta, dtype=np.float64)
    for m, meta in enumerate(cases.METAS):
        for c, conf in enumerate(cases.CONFS):
            for norm in (False, True):
                out[f"syn_m{m}_c{c}_n{int(norm)}"] = old.postprocess(cases.synthetic_pred(norm), meta, norm, conf)
            out[f"thr_m{m}_c{c}"] = old.postprocess(cases.threshold_pred(), meta, False, conf)
            for k, pred in enumerate(cases.dump_preds()):
                out[f"dump{k}_m{m}_c{c}"] = old.postprocess_raw([pred], meta, True, conf_thres=conf)
            for reg_max in (16, 1):
                outs = list(cases.split_outputs(reg_max))
                out[f"split{reg_max}_m{m}_c{c}"] = old.postprocess_raw(outs, meta, False, split=True, conf_thres=conf)
    for reg_max in (16, 1):
        out[f"dec{reg_max}"] = old_split.decode_split(*cases.split_outputs(reg_max))
    pts, st = old_split.anchors_and_strides()
    out["anchors_sha"] = np.array(cases.digest(pts) + cases.digest(st))
    out["nms"] = old.nms_xyxy(*cases.nms_inputs(), old.IOU)
    with tempfile.TemporaryDirectory() as tmp:
        for name in cases.LIST_FILES:
            (Path(tmp) / name).touch()
        out["list_images"] = np.array([p.name for p in old.list_images(tmp)])
    out["constants"] = np.array(
        [old.PROTOCOL_VERSION, old.IMGSZ, old.CONF, old.IOU, old.MAX_DET, old.MAX_NMS, old.MAX_WH], dtype=str
    )
    out["suffixes"] = np.array(sorted(old.IMAGE_SUFFIXES))
    np.savez_compressed(cases.EXPECTED, **out)
    summaries = [old.summarize(values) for values in cases.latency_lists()]
    cases.BOARD_ENV_EXPECTED.write_text(json.dumps({"summarize": summaries}, indent=1) + "\n")
    print(f"wrote {cases.EXPECTED.name} ({len(out)} arrays), {cases.DUMP.name}, {cases.BOARD_ENV_EXPECTED.name}")


if __name__ == "__main__":
    main()
