#!/usr/bin/env python3
"""
run_nim_vlm.py - Extract text from PDFs and images using a VLM provider.

Supports NVIDIA NIM and any OpenAI-compatible API endpoint.

Usage:
    python run_nim_vlm.py <file.pdf|image.png|image.jpg|...> [options]

    Page selection (PDF only):
    python run_nim_vlm.py doc.pdf --pages 1       # single page
    python run_nim_vlm.py doc.pdf --pages 1-3    # page range (inclusive)
    python run_nim_vlm.py doc.pdf --pages 1,3,5  # multiple pages/ranges

Environment variables (VLM / TEXT providers):
    VLM_PROVIDER         Provider for VLM pass: nvidia | openai (default: nvidia)
    VLM_BASE_URL         Base URL for VLM provider
    VLM_MODEL            Model for VLM pass
    VLM_SYSTEM_PROMPT    System prompt for VLM pass

    TEXT_PROVIDER        Provider for text pass: nvidia | openai (default: nvidia)
    TEXT_BASE_URL        Base URL for text provider
    TEXT_MODEL           Model for text pass
    TEXT_SYSTEM_PROMPT   System prompt for text pass

    PAGE_RANGE           Alternative to --pages flag (e.g. "1-3" or "1,3,5")

    --post-process flag enables the text pass which formats output as Obsidian Flavored Markdown.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import json
import os
import re
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import requests
from PIL import Image

# Load .env file if it exists (no external dependency — pure stdlib)
_dotenv = Path(__file__).parent / ".env"
if _dotenv.is_file():
    for line in _dotenv.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Optional docling imports - handled gracefully
if TYPE_CHECKING:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.pipeline.base_pipeline import BasePipeline
    from docling.conversion import PdfContext

try:
    from docling.datamodel.base_models import InputFormat  # type: ignore[attr-defined]
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore[attr-defined]
    from docling.pipeline.base_pipeline import BasePipeline  # type: ignore[attr-defined]
    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_VLM_MODEL = "meta/llama-3.2-11b-vision-instruct"
DEFAULT_VLM_SYSTEM_PROMPT = ""

DEFAULT_TEXT_MODEL = "meta/llama-3.1-nemotron-32b-instruct"
DEFAULT_TEXT_SYSTEM_PROMPT = "/think"

# Obsidian-formatted post-processing prompt
DEFAULT_POST_PROCESS_PROMPT = (
    "You are an OBSIDIAN MARKDOWN FORMATTER. Your ONLY task is to apply Obsidian formatting "
    "to existing markdown — DO NOT modify, correct, reword, summarize, or omit ANY content. "
    "Preserve every word, character, and line exactly as-is. "
    "If there are OCR errors in the original text, preserve them as-is — do not fix them. "
    "If content is garbled or unclear, still preserve it exactly. "
    "Formatting you MAY apply: "
    "  - Convert important text to callouts: > [!note], > [!warning], > [!tip] "
    "  - Add %% page separators: %% Page N — ... %% "
    "  - Clean headings to be wikilink-safe (no #, *, or special chars in heading text) "
    "  - Preserve tables, code blocks, LaTeX math, and all text exactly "
    "Formatting you MUST NOT do: "
    "  - Do NOT rephrase, rewrite, or change any original words "
    "  - Do NOT summarize or condense content "
    "  - Do NOT drop paragraphs or sections "
    "  - Do NOT fix OCR errors or punctuation "
    "  - Do NOT add any new content, explanations, or commentary "
    "Output ONLY the complete markdown — nothing else. Preserve every word, character, "
    "punctuation mark, and line from the original exactly as-is. Do not add any header, "
    "footer, note, or explanation beyond the formatted content itself."
)

SUPPORTED_IMAGE_EXTS = {"png", "jpg", "jpeg", "webp"}
SUPPORTED_MEDIA_TYPES = {
    "png": ("image/png", "image_url"),
    "jpg": ("image/jpeg", "image_url"),
    "jpeg": ("image/jpeg", "image_url"),
    "webp": ("image/webp", "image_url"),
}

EXIT_MISSING_ENV = 1
EXIT_BAD_INPUT = 2
EXIT_API_ERROR = 3

PROMPT_VLM = (
    "Extract ALL content from this image as markdown. "
    "Include any text, equations, and tables. "
    "For any diagrams, figures, charts, or visual elements: describe them ONLY in prose text. "
    "For example: 'A right triangle with vertices labeled A, B, and the origin. A horizontal arrow points right labeled x. A diagonal arrow slopes upward from the origin toward A, with a perpendicular dashed line dropping to the x-axis.' "
    "Do NOT draw ASCII art, do NOT use backticks or code fences for diagrams, "
    "and do NOT create Obsidian wikilinks like ![[...]] for figures. "
    "Preserve all text, equations, and formatting exactly as shown."
)

# ---------------------------------------------------------------------------
# Provider abstraction
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """Abstract protocol for LLM API providers."""

    @property
    def base_url(self) -> str:
        """Base URL for the API (without trailing slash)."""
        ...

    @property
    def api_key_env(self) -> str:
        """Environment variable name that holds the API key."""
        ...

    @property
    def model(self) -> str:
        """Default model name."""
        ...

    @property
    def default_system_prompt(self) -> str:
        """Default system prompt."""
        ...

    @property
    def auth_header_value(self) -> str:
        """Full auth header value (e.g. 'Bearer sk-...' or 'sk-...')."""
        ...

    def chat_url(self) -> str:
        """Full URL for the chat completions endpoint."""
        ...

    def make_request(
        self,
        content: Any,
        stream: bool,
        model: str | None,
        system_prompt: str | None,
    ) -> requests.Response:
        ...


class _BaseProvider:
    """Base class providing common request-building logic."""

    @property
    def base_url(self) -> str:
        """Override in subclasses. Base URL for the API (without trailing slash)."""
        return "https://api.example.com/v1"

    @property
    def api_key_env(self) -> str:
        return "NVIDIA_API_KEY"

    @property
    def auth_scheme(self) -> str:
        """Auth scheme prefix (e.g. 'Bearer ' or ''). Override in subclasses."""
        return "Bearer "

    @property
    def model(self) -> str:
        """Override in subclasses. Default model name."""
        return "default-model"

    @property
    def default_system_prompt(self) -> str:
        return "/think"

    @property
    def auth_header_value(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        return f"{self.auth_scheme}{key}"

    def chat_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def _build_headers(self, stream: bool) -> dict[str, str]:
        headers = {
            "Authorization": self.auth_header_value,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if stream:
            headers["Accept"] = "text/event-stream"
        return headers

    def _build_messages(
        self,
        content: Any,
        system_prompt: str | None,
    ) -> list[dict[str, Any]]:
        prompt = system_prompt if system_prompt is not None else self.default_system_prompt
        return [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]

    def _build_payload(
        self,
        content: Any,
        stream: bool,
        model: str | None,
        system_prompt: str | None,
        gen_overrides: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base = {
            "max_tokens": 8192,
            "temperature": 0.3,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "messages": self._build_messages(content, system_prompt),
            "stream": stream,
            "model": model or self.model,
        }
        if gen_overrides:
            base.update(gen_overrides)
        return base

    def make_request(
        self,
        content: Any,
        stream: bool,
        model: str | None,
        system_prompt: str | None,
        gen_overrides: dict[str, Any] | None = None,
    ) -> requests.Response:
        headers = self._build_headers(stream)
        payload = self._build_payload(content, stream, model, system_prompt, gen_overrides)
        response = requests.post(
            self.chat_url(),
            headers=headers,
            json=payload,
            stream=stream,
            timeout=(300, 600),  # (connect_timeout, read_timeout) — 5min connect, 10min read
        )
        return response


class NVIDIAProvider(_BaseProvider):
    """NVIDIA NIM provider (existing behavior)."""

    @property
    def base_url(self) -> str:
        return (
            os.environ.get("VLM_BASE_URL")
            or os.environ.get("NIM_ENDPOINT")
            or "https://integrate.api.nvidia.com/v1"
        )

    @property
    def api_key_env(self) -> str:
        return "NVIDIA_API_KEY"

    @property
    def auth_scheme(self) -> str:
        return "Bearer "

    @property
    def model(self) -> str:
        return (
            os.environ.get("VLM_MODEL")
            or os.environ.get("NIM_VLM_MODEL")
            or DEFAULT_VLM_MODEL
        )

    @property
    def default_system_prompt(self) -> str:
        return (
            os.environ.get("VLM_SYSTEM_PROMPT")
            or os.environ.get("NIM_SYSTEM_PROMPT")
            or DEFAULT_VLM_SYSTEM_PROMPT
        )


class OpenAICompatibleProvider(_BaseProvider):
    """OpenAI-compatible provider (OpenRouter, LM Studio, local Ollama, etc.)."""

    @property
    def base_url(self) -> str:
        return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    @property
    def api_key_env(self) -> str:
        return "OPENAI_API_KEY"

    @property
    def auth_scheme(self) -> str:
        return ""

    @property
    def model(self) -> str:
        return os.environ.get("OPENAI_MODEL", "gpt-4o")

    @property
    def default_system_prompt(self) -> str:
        return os.environ.get("OPENAI_SYSTEM_PROMPT", "")


# Text provider subclasses (use TEXT_* vars)

class NVIDIATextProvider(_BaseProvider):
    """NVIDIA provider configured via TEXT_* vars for the text post-processing pass."""

    @property
    def base_url(self) -> str:
        return (
            os.environ.get("TEXT_BASE_URL")
            or os.environ.get("NIM_TEXT_ENDPOINT")
            or os.environ.get("VLM_BASE_URL")
            or os.environ.get("NIM_ENDPOINT")
            or "https://integrate.api.nvidia.com/v1"
        )

    @property
    def api_key_env(self) -> str:
        return os.environ.get("TEXT_API_KEY_ENV") or "NIM_TEXT_API_KEY" or "NVIDIA_API_KEY"

    @property
    def auth_scheme(self) -> str:
        return "Bearer "

    @property
    def model(self) -> str:
        return (
            os.environ.get("TEXT_MODEL")
            or os.environ.get("NIM_TEXT_MODEL")
            or DEFAULT_TEXT_MODEL
        )

    @property
    def default_system_prompt(self) -> str:
        return (
            os.environ.get("TEXT_SYSTEM_PROMPT")
            or os.environ.get("NIM_TEXT_SYSTEM_PROMPT")
            or DEFAULT_TEXT_SYSTEM_PROMPT
        )


class OpenAICompatibleTextProvider(_BaseProvider):
    """OpenAI-compatible provider configured via TEXT_* vars for text post-processing."""

    @property
    def base_url(self) -> str:
        return os.environ.get("TEXT_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    @property
    def api_key_env(self) -> str:
        return os.environ.get("TEXT_API_KEY_ENV") or "TEXT_API_KEY" or "OPENAI_API_KEY"

    @property
    def auth_scheme(self) -> str:
        return os.environ.get("TEXT_AUTH_SCHEME", "Bearer ").rstrip() + " "

    @property
    def model(self) -> str:
        return os.environ.get("TEXT_MODEL") or os.environ.get("OPENAI_MODEL", "gpt-4o")

    @property
    def default_system_prompt(self) -> str:
        return os.environ.get("TEXT_SYSTEM_PROMPT") or os.environ.get("OPENAI_SYSTEM_PROMPT", "")


_VLM_PROVIDER_CLASSES: dict[str, type[_BaseProvider]] = {
    "nvidia": NVIDIAProvider,
    "openai": OpenAICompatibleProvider,
}

_TEXT_PROVIDER_CLASSES: dict[str, type[_BaseProvider]] = {
    "nvidia": NVIDIATextProvider,
    "openai": OpenAICompatibleTextProvider,
}


def _make_vlm_provider() -> LLMProvider:
    name = os.environ.get("VLM_PROVIDER", "nvidia").lower()
    cls = _VLM_PROVIDER_CLASSES.get(name, NVIDIAProvider)
    return cls()  # type: ignore[return-value]


def _make_text_provider() -> LLMProvider:
    name = os.environ.get("TEXT_PROVIDER", "nvidia").lower()
    cls = _TEXT_PROVIDER_CLASSES.get(name, NVIDIATextProvider)
    return cls()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_page_range(raw: str) -> set[int]:
    """Parse a page range string like '1', '1-3', '5-7' into a set of 0-based page indices.

    Raises SystemExit(2) on invalid input (page <= 0, non-numeric, malformed range).
    Returns an empty set if the string is empty (meaning "all pages" — handled elsewhere).
    """
    s = raw.strip()
    if not s:
        return set()  # caller treats empty as "all pages"

    result: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_raw, hi_raw = part.split("-", 1)
            lo_str, hi_str = lo_raw.strip(), hi_raw.strip()
            if not lo_str.isdigit() or not hi_str.isdigit():
                sys.stderr.write(f"Error: invalid page range '{part}': numbers only.\n")
                sys.stderr.flush()
                sys.exit(EXIT_BAD_INPUT)
            lo, hi = int(lo_str), int(hi_str)
            if lo < 1 or hi < 1:
                sys.stderr.write(f"Error: page numbers must be >= 1, got '{part}'.\n")
                sys.stderr.flush()
                sys.exit(EXIT_BAD_INPUT)
            if lo > hi:
                sys.stderr.write(f"Error: invalid range '{part}': start > end.\n")
                sys.stderr.flush()
                sys.exit(EXIT_BAD_INPUT)
            result.update(range(lo - 1, hi))  # convert to 0-based, end-exclusive
        else:
            if not part.isdigit():
                sys.stderr.write(f"Error: invalid page number '{part}': must be a positive integer.\n")
                sys.stderr.flush()
                sys.exit(EXIT_BAD_INPUT)
            n = int(part)
            if n < 1:
                sys.stderr.write(f"Error: page numbers must be >= 1, got '{part}'.\n")
                sys.stderr.flush()
                sys.exit(EXIT_BAD_INPUT)
            result.add(n - 1)  # convert to 0-based

    return result


def _get_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename)
    return ext[1:].lower()


def _mime_type(ext: str) -> str:
    return SUPPORTED_MEDIA_TYPES[ext][0]


def _media_type(ext: str) -> str:
    return SUPPORTED_MEDIA_TYPES[ext][1]


def _encode_base64(media_file: str) -> str:
    with open(media_file, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# PDF → List[PIL.Image]
# ---------------------------------------------------------------------------


def _load_pdf_pages(file_path: str, page_indices: set[int] | None = None) -> list[Image.Image]:
    """Convert each page of a PDF into a list of PIL Images using docling or PyMuPDF.

    Args:
        file_path: Path to the PDF file.
        page_indices: Optional set of 0-based page indices to include.
                     If None, all pages are returned.
                     Out-of-range indices cause exit with code 2.
    """

    if not _DOCLING_AVAILABLE:
        # Minimal fallback using PyMuPDF (fitz) if docling is not installed
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(file_path)
            total_pages = len(doc)
            if page_indices is not None:
                max_idx = max(page_indices)
                if max_idx >= total_pages:
                    sys.stderr.write(f"Error: page {max_idx + 1} is out of range (PDF has {total_pages} pages).\n")
                    sys.stderr.flush()
                    sys.exit(EXIT_BAD_INPUT)
            pages: list[Image.Image] = []
            page_nums = page_indices if page_indices is not None else range(total_pages)
            for page_num in page_nums:
                page = doc[page_num]
                mat = fitz.Matrix(3.0, 3.0)  # 3x zoom (~216 DPI) for better text/math readability
                pix = page.get_pixmap(matrix=mat)
                data = pix.tobytes("png")
                pages.append(Image.open(BytesIO(data)))
            return pages
        except ImportError:
            sys.stderr.write(
                "Error: Processing PDF requires either the 'docling' or 'pymupdf' library.\n"
            )
            sys.stderr.flush()
            sys.exit(EXIT_API_ERROR)


def _extract_pdf_images(
    file_path: str, page_indices: set[int] | None = None
) -> dict[int, list[str]]:
    """Extract embedded images from PDF pages. Returns {page_idx: [filepath, ...]}."""
    import fitz

    doc = fitz.open(file_path)
    base = Path(file_path).stem
    attach_dir = Path("Attachments") / base
    attach_dir.mkdir(parents=True, exist_ok=True)
    result: dict[int, list[str]] = {}
    indices = page_indices if page_indices is not None else range(len(doc))
    for page_idx in indices:
        page = doc[page_idx]
        imgs = page.get_images(full=True)
        page_imgs: list[str] = []
        for img_idx, img_info in enumerate(imgs):
            xref = img_info[0]
            try:
                img_data = doc.extract_image(xref)
                ext = img_data.get("ext", "png")
                img_bytes = img_data.get("image")
                if not img_bytes:
                    continue
                filename = f"page{page_idx+1}_fig{img_idx+1}.{ext}"
                out_path = attach_dir / filename
                out_path.write_bytes(img_bytes)
                page_imgs.append(str(out_path))
            except Exception:
                pass
        result[page_idx] = page_imgs
    return result


    # --- docling path ---
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    from docling.pipeline.base_pipeline import BasePipeline

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_in_separate_pages = True

    pipeline = BasePipeline.from_options(
        pipeline_options=pipeline_options,
        format_options={
            InputFormat.PDF: pipeline_options,
        },
    )

    from docling.conversion import PdfContext

    ctx = PdfContext(file_path, [], pipeline=pipeline)
    pages_info = ctx.get_pages()

    # Check out-of-range for docling path
    if page_indices is not None:
        total_pages = len(pages_info)
        max_idx = max(page_indices)
        if max_idx >= total_pages:
            sys.stderr.write(f"Error: page {max_idx + 1} is out of range (PDF has {total_pages} pages).\n")
            sys.stderr.flush()
            sys.exit(EXIT_BAD_INPUT)

    images = []
    for page_info in pages_info.values():
        # page_info.page_number is 1-based; convert to 0-based for comparison
        zero_based = page_info.page_number - 1
        if page_indices is not None and zero_based not in page_indices:
            continue
        try:
            from docling.util.picture import convert_picture_to_pil_image

            img = convert_picture_to_pil_image(page_info)
            images.append(img)
        except Exception as e:
            sys.stderr.write(f"Warning: failed to render page {page_info.page_number}: {e}\n")
            sys.stderr.flush()
    return images


def _open_image(file_path: str) -> Image.Image:
    """Open an image file and convert RGBA → RGB if needed."""
    img = Image.open(file_path)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return img


# ---------------------------------------------------------------------------
# Response streaming
# ---------------------------------------------------------------------------


def _stream_response(response: requests.Response, buffer: list[str] | None = None) -> None:
    """Stream SSE response chunks to stdout, optionally accumulating into buffer."""
    if response.status_code != 200:
        sys.stderr.write(f"API error ({response.status_code}): {response.text}\n")
        sys.stderr.flush()
        sys.exit(EXIT_API_ERROR)

    for line in response.iter_lines():
        if line:
            decoded = line.decode("utf-8")
            if decoded.startswith("data:") or decoded.startswith("{"):
                try:
                    payload = json.loads(decoded.lstrip("data:").strip())
                    content = str(payload.get("choices", [{}])[0].get("delta", {}).get("content", ""))
                    if content:
                        if buffer is not None:
                            buffer.append(content)
                        else:
                            sys.stdout.write(content)
                            sys.stdout.flush()
                except (json.JSONDecodeError, KeyError, IndexError):
                    pass


# ---------------------------------------------------------------------------
# Core API call helpers
# ---------------------------------------------------------------------------


def _chat_with_media(
    provider: LLMProvider,
    media_files: list[str],
    query: str,
    stream: bool = False,
    model: str | None = None,
    system_prompt: str | None = None,
    buffer: list[str] | None = None,
    gen_overrides: dict[str, Any] | None = None,
) -> None:
    """Send media files + query to the VLM provider and stream/print the response."""

    # Build content list
    if len(media_files) == 0:
        content: object = query
    else:
        content = [{"type": "text", "text": query}]
        for media_file in media_files:
            ext = _get_extension(media_file)
            base64_data = _encode_base64(media_file)
            mtype = _media_type(ext)
            media_obj: dict = {
                "type": mtype,
                mtype: {"url": f"data:{_mime_type(ext)};base64,{base64_data}"},
            }
            content.append(media_obj)

    response = provider.make_request(
        content=content,
        stream=stream,
        model=model,
        system_prompt=system_prompt,
        gen_overrides=gen_overrides,
    )

    if not stream:
        if response.status_code != 200:
            sys.stderr.write(f"API error ({response.status_code}): {response.text}\n")
            sys.stderr.flush()
            sys.exit(EXIT_API_ERROR)
        result = response.json()
        sys.stdout.write(str(result))
        sys.stdout.flush()
        return

    # Streamed response
    _stream_response(response, buffer=buffer)


def _chat_text_only(
    provider: LLMProvider,
    query: str,
    stream: bool = True,
    model: str | None = None,
    system_prompt: str | None = None,
    gen_overrides: dict[str, Any] | None = None,
) -> None:
    """Send a text-only query to the text provider and stream the response."""
    response = provider.make_request(
        content=query,
        stream=stream,
        model=model,
        system_prompt=system_prompt,
        gen_overrides=gen_overrides,
    )

    if not stream:
        if response.status_code != 200:
            sys.stderr.write(f"API error ({response.status_code}): {response.text}\n")
            sys.stderr.flush()
            sys.exit(EXIT_API_ERROR)
        result = response.json()
        sys.stdout.write(str(result))
        sys.stdout.flush()
        return

    _stream_response(response)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from PDFs and images using a VLM provider (NVIDIA NIM or OpenAI-compatible)."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Path to a PDF or image file (PNG, JPG, JPEG, WebP).",
    )
    parser.add_argument(
        "--pages",
        dest="pages",
        default=None,
        help=(
            "One or more pages / page ranges to process (1-based). "
            "Examples: --pages 1     (single page), "
            "--pages 1-3   (pages 1 through 3), "
            "--pages 1,3,5 (pages 1, 3 and 5). "
            "Only valid for PDF input."
        ),
    )
    parser.add_argument(
        "--no-post-process",
        dest="no_post_process",
        action="store_true",
        default=False,
        help="Disable the post-processing pass (post-processing is enabled by default).",
    )
    parser.add_argument(
        "--output",
        dest="output",
        default=None,
        help=(
            "Output file. Default: <input>.md (auto-derived from input file). "
            "Use '-' to write to stdout."
        ),
    )

    # Build and validate provider early so we can check API key before parsing args
    vlm_provider = _make_vlm_provider()

    # Check required API key
    _api_key = os.environ.get(vlm_provider.api_key_env, "")
    if not _api_key:
        sys.stderr.write(f"Error: {vlm_provider.api_key_env} environment variable is not set.\n")
        sys.stderr.flush()
        sys.exit(EXIT_MISSING_ENV)

    args = parser.parse_args()

    if args.file is None:
        sys.stderr.write(
            "Usage: python run_nim_vlm.py <file.pdf|image.png|...> [--pages RANGE] [--post-process]\n"
        )
        sys.stderr.flush()
        sys.exit(EXIT_MISSING_ENV)

    file_path = args.file
    if not os.path.isfile(file_path):
        sys.stderr.write(f"Error: file not found: {file_path}\n")
        sys.stderr.flush()
        sys.exit(EXIT_BAD_INPUT)

    # Determine output destination
    if args.output == "-":
        out_file = None
    elif args.output is not None:
        out_file = open(args.output, "w")
    else:
        # Auto-derive from input file
        input_path = Path(file_path)
        out_path = input_path.with_suffix(".md")
        out_file = open(out_path, "w")

    # Load images from PDF or single image
    images: list[Image.Image] = []
    ext = _get_extension(file_path)

    if ext == "pdf":
        page_indices: set[int] | None = None
        raw_pages = (
            args.pages
            if args.pages is not None
            else os.environ.get("PAGE_RANGE") or os.environ.get("NIM_PAGE_RANGE")
        )
        if raw_pages is not None:
            page_indices = _parse_page_range(raw_pages)
        try:
            images = _load_pdf_pages(file_path, page_indices=page_indices)
        except Exception as e:
            sys.stderr.write(f"Error loading PDF: {e}\n")
            sys.stderr.flush()
            sys.exit(EXIT_API_ERROR)
        # Extract embedded images from PDF pages for potential later reference
        try:
            _extract_pdf_images(file_path, page_indices=page_indices)
        except Exception:
            pass  # Non-fatal — proceed without image extraction
    elif ext in SUPPORTED_IMAGE_EXTS:
        try:
            images = [_open_image(file_path)]
        except Exception as e:
            sys.stderr.write(f"Error opening image: {e}\n")
            sys.stderr.flush()
            sys.exit(EXIT_BAD_INPUT)
    else:
        sys.stderr.write(
            f"Error: unsupported file type '.{ext}'. "
            f"Supported: PDF, {', '.join(sorted(SUPPORTED_IMAGE_EXTS))}\n"
        )
        sys.stderr.flush()
        sys.exit(EXIT_MISSING_ENV)

    if not images:
        sys.stderr.write("Error: no pages/images could be extracted.\n")
        sys.stderr.flush()
        sys.exit(EXIT_API_ERROR)

    # Save images to temporary files so we can base64-encode them
    tmp_paths: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, img in enumerate(images):
            tmp_path = os.path.join(tmpdir, f"page_{i}.png")
            img.save(tmp_path, format="PNG")
            tmp_paths.append(tmp_path)

        # Determine whether to run post-processing pass
        post_process_enabled = not args.no_post_process

        text_provider: LLMProvider | None = None
        if post_process_enabled:
            text_provider = _make_text_provider()
            text_api_key = os.environ.get(text_provider.api_key_env, "")
            if not text_api_key:
                sys.stderr.write(
                    f"\nWarning: TEXT_API_KEY not set (tried env var: {text_provider.api_key_env}). "
                    "Post-processing DISABLED — falling back to VLM-only pass. "
                    "Set TEXT_API_KEY to enable formatting.\n"
                )
                sys.stderr.flush()
                post_process_enabled = False
                text_provider = None

        # NIM VLM API accepts at most 1 image per request — call once per page
        total = len(tmp_paths)

        # Redirect stdout to file if output was auto-derived or specified
        redirect_context = contextlib.redirect_stdout(out_file) if out_file else contextlib.nullcontext()
        with redirect_context:
            # Pass 1: VLM extraction — buffer each page's raw output
            page_buffers: list[str] = []
            for i, tmp_path in enumerate(tmp_paths):
                sys.stderr.write(f"\rPass 1/2 — Page {i+1}/{total}...")
                sys.stderr.flush()
                page_buf: list[str] = []
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        _chat_with_media(
                            provider=vlm_provider,
                            media_files=[tmp_path],
                            query=PROMPT_VLM,
                            stream=True,
                            buffer=page_buf,
                            gen_overrides={"temperature": 0.1, "top_p": 0.9, "max_tokens": 16384},
                        )
                        break
                    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                        if attempt < max_retries - 1:
                            sys.stderr.write(f" (retry {attempt+2}/{max_retries}...)")
                            sys.stderr.flush()
                        else:
                            sys.stderr.write(f"\nAPI error after {max_retries} attempts: {e}\n")
                            sys.stderr.flush()
                            sys.exit(EXIT_API_ERROR)
                page_text = "".join(page_buf)
                # Remove any ![[...]] Obsidian wikilinks — they reference non-existent files
                page_text = re.sub(r'\n*!\[\[.*?\]\]\s*\n?', '\n', page_text)
                page_buffers.append(page_text)
            sys.stderr.write(f"\r{' ' * 50}\r")
            sys.stderr.flush()

            # Pass 2: optional text-only post-processing (Obsidian formatting)
            if post_process_enabled and text_provider is not None:
                post_process_prompt = (
                    os.environ.get("POST_PROCESS_PROMPT")
                    or os.environ.get("NIM_POST_PROCESS_PROMPT")
                    or DEFAULT_POST_PROCESS_PROMPT
                )
                for i, raw_markdown in enumerate(page_buffers):
                    sys.stderr.write(f"\rPass 2/2 — Page {i+1}/{total}...")
                    sys.stderr.flush()
                    page_markdown = raw_markdown.strip()
                    query = f"{post_process_prompt}\n\n---\n\n{page_markdown}"
                    max_retries_pp = 3
                    for attempt in range(max_retries_pp):
                        try:
                            _chat_text_only(
                                provider=text_provider,
                                query=query,
                                stream=True,
                                gen_overrides={"temperature": 0.0},
                            )
                            break
                        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, requests.exceptions.ReadTimeout) as e:
                            if attempt < max_retries_pp - 1:
                                sys.stderr.write(f" (retry {attempt+2}/{max_retries_pp}...)")
                                sys.stderr.flush()
                            else:
                                sys.stderr.write(f"\nPost-process error after {max_retries_pp} attempts: {e}\n")
                                sys.stderr.flush()
                                sys.exit(EXIT_API_ERROR)
                sys.stderr.write(f"\r{' ' * 50}\r")
                sys.stderr.flush()
            else:
                # No post-processing — output raw VLM buffers directly
                for raw_markdown in page_buffers:
                    sys.stdout.write(raw_markdown)
                    sys.stdout.flush()

    # Close output file if we opened one
    if out_file is not None:
        out_file.close()


if __name__ == "__main__":
    main()