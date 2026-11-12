"""rk3588_common.model_io must give exactly the outputs of legacy-final:ai/board/{common,split_decode}.py (fixtures)."""

import cases
import numpy as np
import pytest

from rk3588_common import model_io as io

EXPECTED = np.load(cases.EXPECTED)


@pytest.mark.parametrize("i", range(len(cases.IMAGE_SHAPES)))
def test_preprocess(i):
    x, meta = io.preprocess(cases.images()[i])
    assert x.shape == (1, io.IMGSZ, io.IMGSZ, 3) and x.dtype == np.uint8
    assert cases.digest(x) == str(EXPECTED[f"pre{i}_sha"])
    assert np.array_equal(np.array(meta, dtype=np.float64), EXPECTED[f"pre{i}_meta"])


@pytest.mark.parametrize("m", range(len(cases.METAS)))
@pytest.mark.parametrize("c", range(len(cases.CONFS)))
def test_postprocess(m, c):
    meta, conf = cases.METAS[m], cases.CONFS[c]
    for norm in (False, True):
        got = io.postprocess(cases.synthetic_pred(norm), meta, norm, conf)
        assert np.array_equal(got, EXPECTED[f"syn_m{m}_c{c}_n{int(norm)}"])
    assert np.array_equal(io.postprocess(cases.threshold_pred(), meta, False, conf), EXPECTED[f"thr_m{m}_c{c}"])
    for k, pred in enumerate(cases.dump_preds()):
        assert np.array_equal(io.postprocess_raw([pred], meta, True, conf_thres=conf), EXPECTED[f"dump{k}_m{m}_c{c}"])
    for reg_max in (16, 1):
        outs = list(cases.split_outputs(reg_max))
        got = io.postprocess_raw(outs, meta, False, split=True, conf_thres=conf)
        assert np.array_equal(got, EXPECTED[f"split{reg_max}_m{m}_c{c}"])


def test_fixtures_are_not_trivial():
    assert all(len(EXPECTED[f"dump{k}_m0_c0"]) > 0 for k in range(len(cases.dump_preds())))
    assert len(EXPECTED["thr_m0_c0"]) == 10 and len(EXPECTED["thr_m0_c1"]) == 300
    assert len(EXPECTED["syn_m0_c0_n0"]) > 1 and len(EXPECTED["nms"]) < len(cases.nms_inputs()[0])


@pytest.mark.parametrize("reg_max", [16, 1])
def test_decode_split(reg_max):
    assert cases.digest(io.decode_split(*cases.split_outputs(reg_max))) == str(EXPECTED[f"dec{reg_max}_sha"])


def test_anchors_and_strides():
    pts, st = io.anchors_and_strides()
    assert cases.digest(pts) + cases.digest(st) == str(EXPECTED["anchors_sha"])


def test_nms_xyxy():
    assert np.array_equal(io.nms_xyxy(*cases.nms_inputs(), io.IOU), EXPECTED["nms"])


def test_list_images(tmp_path):
    for name in cases.LIST_FILES:
        (tmp_path / name).touch()
    assert [p.name for p in io.list_images(tmp_path)] == list(EXPECTED["list_images"])


def test_constants():
    got = [io.PROTOCOL_VERSION, io.IMGSZ, io.CONF, io.IOU, io.MAX_DET, io.MAX_NMS, io.MAX_WH]
    assert np.array_equal(np.array(got, dtype=str), EXPECTED["constants"])
    assert sorted(io.IMAGE_SUFFIXES) == list(EXPECTED["suffixes"])
