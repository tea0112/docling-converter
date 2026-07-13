# VLM PDF & Image Text Extractor

Extract text from PDFs and images using a Vision Language Model via NVIDIA NIM or any OpenAI-compatible API endpoint.

## Requirements

- Python 3.9+
- `pip install pillow requests`

Optional for PDF support:
- `pip install docling` — preferred, higher-quality rendering
- or `pip install pymupdf` — fallback if docling is not installed

## Quick start

### NVIDIA NIM

```bash
# 1. Get your API key from https://build.nvidia.com/
export NVIDIA_API_KEY=nvapi-xxxxx

# 2. Run
python3 run_nim_vlm.py document.pdf
```

### OpenAI-compatible (OpenRouter, LM Studio, Ollama, etc.)

```bash
export VLM_PROVIDER=openai
export OPENAI_BASE_URL=https://openrouter.ai/api/v1
export OPENAI_API_KEY=sk-...
python3 run_nim_vlm.py document.pdf
```

Output is written to a file with the same base name but `.md` extension:

```bash
python3 run_nim_vlm.py document.pdf   # → document.md
python3 run_nim_vlm.py image.png      # → image.md
```

Use `--output -` to force output to stdout (for piping):

```bash
python3 run_nim_vlm.py document.pdf --output - | other-command
```

## Providers

Set `VLM_PROVIDER` (Pass 1, VLM) and `TEXT_PROVIDER` (Pass 2, text post-processing) to choose the backend:

| Provider | Value | Description |
|---|---|---|
| `nvidia` | *(default)* | NVIDIA NIM at `https://integrate.api.nvidia.com/v1` |
| `openai` | | Any OpenAI-compatible API (OpenRouter, LM Studio, Ollama, etc.) |

## Environment variables

### VLM Provider (Pass 1)

| Variable | Required | Default | Description |
|---|---|---|---|
| `VLM_PROVIDER` | No | `nvidia` | Provider: `nvidia` or `openai` |
| `NVIDIA_API_KEY` | For NVIDIA | — | Your NVIDIA API key |
| `OPENAI_API_KEY` | For OpenAI | — | Your OpenAI-compatible API key |
| `VLM_BASE_URL` | No | `https://integrate.api.nvidia.com/v1` | API base URL |
| `VLM_MODEL` | No | `meta/llama-3.2-11b-vision-instruct` | VLM model |
| `VLM_SYSTEM_PROMPT` | No | *(empty)* | System prompt for VLM — empty recommended for figure/diagram extraction (using `/think` may reduce visual detail) |

### Text Provider (Pass 2 — post-processing)

| Variable | Required | Default | Description |
|---|---|---|---|
| `TEXT_PROVIDER` | No | `nvidia` | Provider: `nvidia` or `openai` |
| `TEXT_API_KEY_ENV` | No | *(varies)* | Env var name holding the text API key |
| `TEXT_BASE_URL` | No | *(same as VLM)* | API base URL |
| `TEXT_MODEL` | No | `meta/llama-3.1-nemotron-32b-instruct` | Text model |
| `TEXT_SYSTEM_PROMPT` | No | `/think` | System prompt for text pass |

### Post-processing

| Variable | Required | Default | Description |
|---|---|---|---|
| `POST_PROCESS_PROMPT` | No | *(built-in)* | Prompt for Obsidian formatting in Pass 2 |

### Backward-compatible aliases

These old `NIM_*` / `NIM_TEXT_*` / `NIM_*` vars still work:

| Old variable | New variable |
|---|---|
| `NIM_ENDPOINT` | `VLM_BASE_URL` |
| `NIM_VLM_MODEL` | `VLM_MODEL` |
| `NIM_SYSTEM_PROMPT` | `VLM_SYSTEM_PROMPT` |
| `NIM_TEXT_ENDPOINT` | `TEXT_BASE_URL` |
| `NIM_TEXT_MODEL` | `TEXT_MODEL` |
| `NIM_TEXT_SYSTEM_PROMPT` | `TEXT_SYSTEM_PROMPT` |
| `NIM_TEXT_API_KEY` | Set `TEXT_API_KEY_ENV=NIM_TEXT_API_KEY` |
| `NIM_PAGE_RANGE` | `PAGE_RANGE` |

You can copy `.env.example` to `.env` and fill in your values, then source it:

```bash
cp .env.example .env
# edit .env with your key
source .env && python3 run_nim_vlm.py document.pdf
```

## Supported input types

- **PDF** — each page is rendered and sent to the VLM individually
- **Images** — PNG, JPG, JPEG, WebP

## Page selection (PDF only)

```bash
python3 run_nim_vlm.py doc.pdf --pages 1       # single page
python3 run_nim_vlm.py doc.pdf --pages 1-3    # page range (inclusive)
python3 run_nim_vlm.py doc.pdf --pages 1,3,5  # multiple pages/ranges
```

Or via environment variable:

```bash
PAGE_RANGE=1-3 python3 run_nim_vlm.py doc.pdf
```

## Two-pass mode: VLM + text post-processing

Post-processing is enabled **by default**. The script first calls the VLM to extract text from each image,
then (if `TEXT_API_KEY` is configured) calls a text-only model to format the output as **Obsidian Flavored Markdown**
(callouts, `%%` page separators, wikilink-safe headings, etc.):

```bash
# Post-processing is on by default; disable it with --no-post-process
python3 run_nim_vlm.py doc.pdf --no-post-process
```

If `TEXT_API_KEY` is not set, the script prints a warning and falls back to VLM-only (single-pass) mode.

**Generation settings per pass:**

| Pass | Temperature | top_p | Notes |
|---|---|---|---|
| Pass 1 — VLM | `0.1` | `0.9` | Low temperature for accurate extraction |
| Pass 2 — text | `0.0` | — | Deterministic formatting |

**Retry logic:** Both passes retry up to 3 times on timeout or connection error before failing.

**VLM prompt:** The VLM is instructed to describe figures and diagrams in full prose text — no ASCII art, no backticks/code fences, and no Obsidian `![[...]]` wikilinks (which would reference non-existent files).

**Post-process prompt:** The text pass is instructed to preserve the original verbatim — it must not add, remove, rephrase, or condense any content. Formatting it applies only (callouts, page separators, heading cleanup).

## Obsidian output format

When post-processing is enabled (default), the text pass formats output as Obsidian Flavored Markdown:

- **Callouts**: `> [!note]`, `> [!warning]`, `> [!tip]`, etc.
- **Page separators**: `%% Page N — ... %%`
- **Clean headings**: wikilink-safe (no special characters)
- **Tables / code blocks / LaTeX math**: preserved

```markdown
%% Page 1 — Introduction %%

> [!note] Key Finding
> This is an important callout.

| Column 1 | Column 2 |
|---|---|
| Value | Value |

```python
def hello():
    pass
```

> [!warning] Watch Out
> Something to be careful about.
```

## Exit codes

| Code | Meaning |
|---|---|
| `1` | Required API key not set, or no input file given |
| `2` | Input file not found |
| `3` | API returned an error |

## Example usage

### NVIDIA NIM (default)

```bash
NVIDIA_API_KEY=xxx python3 run_nim_vlm.py doc.pdf

# Use a different model
VLM_MODEL=nvidia/nemotron-nano-12b-v2-vl NVIDIA_API_KEY=xxx python3 run_nim_vlm.py doc.pdf
```

### OpenRouter

```bash
VLM_PROVIDER=openai \
  OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
  OPENAI_API_KEY=sk-... \
  OPENAI_MODEL=anthropic/claude-3.5-sonnet \
  python3 run_nim_vlm.py doc.pdf
```

### LM Studio (local)

```bash
VLM_PROVIDER=openai \
  OPENAI_BASE_URL=http://localhost:1234/v1 \
  OPENAI_API_KEY=sk-12345 \
  OPENAI_MODEL=llama-3.2-vision \
  python3 run_nim_vlm.py doc.pdf
```