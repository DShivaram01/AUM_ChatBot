# AUM Academic Assistant

Reproducible local deployment of the AUM Academic Assistant prototype: COS research and Housing policy answers through FastAPI, Gradio, and Electron. Open-ended mode uses local Mistral pretrained knowledge and is explicitly disclosed.

## Fresh-clone commands

1. git clone https://github.com/DShivaram01/AUM_ChatBot.git
2. cd AUM_ChatBot
3. bash scripts/setup_lab.sh
4. bash scripts/verify_install.sh
5. bash scripts/run_api.sh
6. In another terminal: bash scripts/run_desktop_dev.sh

The API runs at http://127.0.0.1:8000. First launch downloads Hugging Face models and creates local ignored retrieval indexes.

## Layout

- server, pipeline, models: Python application
- desktop/client: Electron source
- data: versioned input data
- scripts: lab setup, verification, and launch commands
- docs: architecture, guides, decisions, and frozen baseline manifest
- archive: legacy material, not the active build

## Packaging

Build Electron with: cd desktop/client && npm run dist

Optional flash-attn can be installed manually with the active lab version: python -m pip install --no-build-isolation flash-attn==2.8.3.post1. The copied source now falls back to PyTorch SDPA when it is unavailable.

Review data provenance and redistribution rights before public release. See docs/project_guides/source_provenance_and_versioning.md.
