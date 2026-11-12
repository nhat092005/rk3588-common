# rk3588-common

Code shared by `rk3588-ai` (PC) and `rk3588-deploy` (board). Must stay numpy only, no torch, so the board can import it.

Planned content (nothing migrated yet):

| Path | Content |
|---|---|
| `python/` | Pre/post-processing and split-head box decode, reference for the C++ port |
| `tests/` | Golden checks that PC and board give identical outputs |

Used as a submodule of `rk3588`. Consumers pin a commit.
