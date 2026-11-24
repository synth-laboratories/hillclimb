# synth-hillclimb

Rust-first prompt, context, skill, and agent harness optimization engine (`import hillclimb`, CLI `hillclimb`, Apache-2.0).

## Install

Install the prebuilt CLI tool with `uv` (no Rust toolchain required):

```sh
uv tool install --no-build synth-hillclimb
```

Or add `synth-hillclimb` to a Python project:

```sh
uv add synth-hillclimb
```

Optional extras:

- `tui` — interactive terminal observer and live visuals panels
- `mcp` — Model Context Protocol stdio server (`hillclimb mcp`)
- `all` — all optional extras

```sh
uv tool install --no-build 'synth-hillclimb[all]'
```

## Quick start

```sh
hillclimb --help
hillclimb mcp --help
```

```python
import hillclimb
```

See the [repository](https://github.com/synth-laboratories/hillclimb) and [`docs/INSTALL.md`](https://github.com/synth-laboratories/hillclimb/blob/main/docs/INSTALL.md) for full documentation and provenance verification.
