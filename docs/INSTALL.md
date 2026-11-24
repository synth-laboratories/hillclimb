# Install and start Hillclimb

The package distribution is **`synth-hillclimb`** (`import hillclimb`, CLI `hillclimb`, Apache-2.0). Check [official releases](https://github.com/synth-laboratories/hillclimb/releases) for availability, qualified artifacts, provenance manifests, and exact checksums; do not install a similarly named unverified package.

Python 3.12.16+ (or another security-patched Python) and `uv` 0.13.4+ are recommended. Installing a prebuilt wheel with `--no-build` requires no Rust toolchain.

## Install from PyPI

Install the CLI tool:

```sh
uv tool install --no-build 'synth-hillclimb==0.1.0'
```

Run a one-off invocation with `uvx`:

```sh
uvx --no-build --from 'synth-hillclimb==0.1.0' hillclimb --version
```

Add `synth-hillclimb` as a project library dependency:

```sh
uv add 'synth-hillclimb==0.1.0'
```

## Optional extras

`synth-hillclimb` separates optional UI and protocol integrations into extras:

- `tui` — interactive terminal observer and live visuals panels
- `mcp` — Model Context Protocol stdio server support
- `all` — all optional extras

Install with extras:

```sh
uv tool install --no-build 'synth-hillclimb[tui]==0.1.0'
uv tool install --no-build 'synth-hillclimb[mcp]==0.1.0'
uv tool install --no-build 'synth-hillclimb[all]==0.1.0'
uv add 'synth-hillclimb[all]==0.1.0'
```

## Install a downloaded release wheel

For an offline or pre-verified release wheel from [official releases](https://github.com/synth-laboratories/hillclimb/releases):

```sh
uv tool install --no-build --from /absolute/path/to/synth_hillclimb-0.1.0-<tags>.whl synth-hillclimb
```

## Verify capabilities and MCP entrypoint

Inspect installed optimizers, agent backends, sandbox substrates, and optional extras:

```sh
hillclimb --version
hillclimb capabilities
```

Start the stdio MCP server (requires `[mcp]` or `[all]`):

```sh
hillclimb mcp
```

## Channels and provenance

- **Stable releases (`X.Y.Z`)**: published manually from protected `main` via GitHub OIDC Trusted Publishing (`pypi` environment) with Sigstore/PyPI attestations.
- **Nightly prereleases (`X.Y.Z.devN`)**: opt-in prereleases selected explicitly; never automatic upgrades for stable pins.
- **Release verification**: publication binds `publish-manifest.json`, canonical `cohort.json`, wheel digests, and OIDC identity via `scripts/verify_publish.py` and `scripts/verify_attestation.py`.
