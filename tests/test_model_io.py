"""Tests for rk3588_common.model_io against legacy-final fixtures."""

import cases
import numpy as np
import pytest

from rk3588_common import model_io as io

EXPECTED = np.load(cases.EXPECTED)


def assert_close_detections(got, expected):
    """Check detections match within the exp tolerances of cases."""
    assert got.shape == expected.shape
    assert np.array_equal(got[:, 5], expected[:, 5])
    np.testing.assert_allclose(
        got[:, :4], expected[:, :4], rtol=0, atol=cases.BOX_ATOL
    )
    np.testing.assert_allclose(
        got[:, 4], expected[:, 4], rtol=0, atol=cases.SCORE_ATOL
    )


@pytest.mark.parametrize("i", range(len(cases.IMAGE_SHAPES)))
def test_preprocess(i):
    x, meta = io.preprocess(cases.images()[i])
    assert x.shape == (1, io.IMGSZ, io.IMGSZ, 3)
    assert x.dtype == np.uint8
    assert cases.digest(x) == str(EXPECTED[f"pre{i}_sha"])
    assert np.array_equal(
        np.array(meta, dtype=np.float64), EXPECTED[f"pre{i}_meta"]
    )


@pytest.mark.parametrize("m", range(len(cases.METAS)))
@pytest.mark.parametrize("c", range(len(cases.CONFS)))
def test_postprocess(m, c):
    meta, conf = cases.METAS[m], cases.CONFS[c]
    for norm in (False, True):
        got = io.postprocess(cases.synthetic_pred(norm), meta, norm, conf)
        assert np.array_equal(got, EXPECTED[f"syn_m{m}_c{c}_n{int(norm)}"])
    assert np.array_equal(
        io.postprocess(cases.threshold_pred(), meta, False, conf),
        EXPECTED[f"thr_m{m}_c{c}"],
    )
    for k, pred in enumerate(cases.dump_preds()):
        assert np.array_equal(
            io.postprocess_raw([pred], meta, True, conf_thres=conf),
            EXPECTED[f"dump{k}_m{m}_c{c}"],
        )
    for reg_max in (16, 1):
        outs = list(cases.split_outputs(reg_max))
        got = io.postprocess_raw(
            outs, meta, False, split=True, conf_thres=conf
        )
        assert_close_detections(got, EXPECTED[f"split{reg_max}_m{m}_c{c}"])


def test_fixtures_are_not_trivial():
    assert all(
        len(EXPECTED[f"dump{k}_m0_c0"]) > 0
        for k in range(len(cases.dump_preds()))
    )
    assert len(EXPECTED["thr_m0_c0"]) == 10
    assert len(EXPECTED["thr_m0_c1"]) == 300
    assert len(EXPECTED["syn_m0_c0_n0"]) > 1
    assert len(EXPECTED["nms"]) < len(cases.nms_inputs()[0])


@pytest.mark.parametrize("reg_max", [16, 1])
def test_decode_split(reg_max):
    got = io.decode_split(*cases.split_outputs(reg_max))
    expected = EXPECTED[f"dec{reg_max}"]
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    np.testing.assert_allclose(
        got[:, :4], expected[:, :4], rtol=0, atol=cases.BOX_ATOL
    )
    np.testing.assert_allclose(
        got[:, 4:], expected[:, 4:], rtol=0, atol=cases.SCORE_ATOL
    )


def test_anchors_and_strides():
    pts, st = io.anchors_and_strides()
    assert cases.digest(pts) + cases.digest(st) == str(EXPECTED["anchors_sha"])


def test_nms_xyxy():
    assert np.array_equal(
        io.nms_xyxy(*cases.nms_inputs(), io.IOU), EXPECTED["nms"]
    )


def test_list_images(tmp_path):
    for name in cases.LIST_FILES:
        (tmp_path / name).touch()
    assert [p.name for p in io.list_images(tmp_path)] == list(
        EXPECTED["list_images"]
    )


def test_constants():
    got = [
        io.PROTOCOL_VERSION,
        io.IMGSZ,
        io.CONF,
        io.IOU,
        io.MAX_DET,
        io.MAX_NMS,
        io.MAX_WH,
    ]
    assert np.array_equal(np.array(got, dtype=str), EXPECTED["constants"])
    assert sorted(io.IMAGE_SUFFIXES) == list(EXPECTED["suffixes"])


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"args": {"quantize": 8}}, True),
        ({"args": {"quantize": 8}, "split_head": True}, False),
        ({"args": {"quantize": 16}}, False),
        ({"args": {}}, False),
        ({}, False),
    ],
)
def test_boxes_normalized(metadata, expected):
    """Rule of legacy-final load_model and RknnDumpBackend."""
    assert io.boxes_normalized(metadata) is expected


def test_eval_conf():
    # legacy-final:ai/core/evaluator.py CONF and infer_dump.py EVAL_CONF
    assert io.EVAL_CONF == 0.001
