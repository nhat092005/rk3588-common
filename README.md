# rk3588-common

Code shared by `rk3588-ai` (PC) and `rk3588-deploy` (board). Must stay numpy only, no torch, so the board can import it. It is the reference that the C++ port is checked against.

Planned content (nothing migrated yet):

| Path | Content |
|---|---|
| `python/` | Pre/post-processing and split-head box decode |
| `tests/` | Golden checks that PC and board give identical outputs |

## How it is consumed

The meta-repo `rk3588` is the only place that pins this repo. `rk3588-ai` and `rk3588-deploy` do not contain a copy, so work from the `rk3588` checkout.

- PC: `uv pip install -e common` from the `rk3588` root.
- Board: `make deploy-sync` (planned) rsyncs `deploy/` and `common/python/` and sets `PYTHONPATH`.
