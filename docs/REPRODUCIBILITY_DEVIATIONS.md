# Reproducibility deviations from the frozen Desktop baseline

The checksum manifest preserves the unchanged Desktop baseline. Two repository-only changes make a clone portable:

1. config.py derives the project root from AUM_CHATBOT_HOME or its own location instead of the Desktop-specific absolute path.
2. models/loader.py prefers FlashAttention when installed and explicitly falls back to PyTorch SDPA when it is unavailable.

These changes do not alter the Desktop baseline. They are required so a fresh lab clone can run at an arbitrary filesystem location and does not fail solely because an optional CUDA extension is absent.

## Post-baseline dependencies

- `pymupdf4llm==1.28.2` was added after the frozen Desktop baseline for
  page-aware Markdown extraction of Housing policy PDFs. It brings in PyMuPDF
  (AGPL-licensed) and is a new capability, not a baseline deviation.
