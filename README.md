# rk3588-common

Contracts shared by `rk3588-ai` (PC) and `rk3588-deploy` (board), package `rk3588_common`. Only numpy, OpenCV and psutil, no torch, and Python 3.10 compatible so the board can import it. It is the reference that the C++ port is checked against.

| Path | Content | Status |
|---|---|---|
| `python/rk3588_common/model_io.py` | Model I/O constants (`IMGSZ`, `CONF`, `IOU`, `MAX_DET`, `PROTOCOL_VERSION`, ...), `preprocess`, `nms_xyxy`, `postprocess`, `postprocess_raw`, `anchors_and_strides`, `decode_split`, `list_images` | Migrated |
| `python/rk3588_common/board_env.py` | Board telemetry: frequencies, temperatures, system info, resource monitor, latency summary | Planned |
| `python/rk3588_common/provenance.py` | One provenance schema for every result file | Planned |
| `tests/` | Equivalence tests against fixtures produced by the legacy code | |

## Tests

`tests/test_model_io.py` checks that `model_io` gives exactly the outputs of `legacy-final:ai/board/common.py` and `legacy-final:ai/board/split_decode.py`. The expected outputs in `tests/fixtures/` were written once by `tests/make_fixtures.py`, run with the legacy interpreter on the legacy checkout at tag `legacy-final` (see its docstring). The inputs are built in `tests/cases.py`: synthetic images and outputs, and the first 3 images of the baseline INT8 board dump.

Run from the meta-repo: `make check`, or `cd common && ../.venv/bin/python -m pytest tests`.

## How it is consumed

The meta-repo `rk3588` is the only place that pins this repo. `rk3588-ai` and `rk3588-deploy` do not contain a copy, so work from the `rk3588` checkout.

- PC: editable install through the meta-repo uv workspace (`make sync`). Depends on `opencv-python` (the one ultralytics installs), not `-headless`, so the PC has a single cv2.
- Board: the meta-repo `make sync-board` copies `common/python/` to the board, the board venv provides numpy and `opencv-python-headless` of the same versions.
