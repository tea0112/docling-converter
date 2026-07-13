#!/usr/bin/env python3
"""
Unit tests for run_nim_vlm.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from io import BytesIO
from typing import Any
from unittest.mock import ANY, MagicMock, patch

# Determine path to module
_script_dir = os.path.dirname(os.path.abspath(__file__))
_module_path = os.path.join(_script_dir, "run_nim_vlm.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_temp_image(fmt="PNG", suffix=".png") -> tuple[str, bytes]:
    """Create a temp image file and return (path, raw_bytes)."""
    try:
        from PIL import Image
    except ImportError:
        raise unittest.SkipTest("PIL not available")

    img = Image.new("RGB", (100, 100), color="red")
    buf = BytesIO()
    img.save(buf, format=fmt)
    raw = buf.getvalue()

    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, raw)
    os.close(fd)
    return path, raw


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestCliArgumentValidation(unittest.TestCase):
    """Test (a): no args → exit 1 with API key missing in stderr."""

    def test_no_args_exits_1_without_api_key(self):
        """Running with no args and no API key must exit 1."""
        env = {k: v for k, v in os.environ.items() if k != "NVIDIA_API_KEY"}
        env["NVIDIA_API_KEY"] = ""  # empty string is falsy; .env setdefault won't override
        result = subprocess.run(
            [sys.executable, _module_path],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("NVIDIA_API_KEY", result.stderr)

    def test_no_args_exits_1_with_api_key_but_no_file(self):
        """Running with API key but no file argument must exit 1."""
        env = {**os.environ, "NVIDIA_API_KEY": "test_key"}
        result = subprocess.run(
            [sys.executable, _module_path],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 1, result.stderr)


class TestMissingFile(unittest.TestCase):
    """Test (b): missing file → exit 2."""

    def test_missing_file_exits_2(self):
        """Passing a non-existent file must exit 2."""
        env = {**os.environ, "NVIDIA_API_KEY": "test_key"}
        result = subprocess.run(
            [sys.executable, _module_path, "/nonexistent/file.pdf"],
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("not found", result.stderr)


class TestModuleImport(unittest.TestCase):
    """Test (c): module imports cleanly."""

    def test_import_run_nim_vlm(self):
        """Importing the module must succeed."""
        import run_nim_vlm

        # Test that key provider classes are importable and satisfy LLMProvider
        from run_nim_vlm import (
            NVIDIAProvider,
            OpenAICompatibleProvider,
            NVIDIATextProvider,
            OpenAICompatibleTextProvider,
            LLMProvider,
        )

        # Provider classes exist
        self.assertTrue(callable(NVIDIAProvider))
        self.assertTrue(callable(OpenAICompatibleProvider))

        # Default VLM model preserved
        self.assertEqual(
            run_nim_vlm.DEFAULT_VLM_MODEL, "meta/llama-3.2-11b-vision-instruct"
        )
        self.assertEqual(
            run_nim_vlm.DEFAULT_TEXT_MODEL, "meta/llama-3.1-nemotron-32b-instruct"
        )

        # Exit codes preserved
        self.assertEqual(run_nim_vlm.EXIT_MISSING_ENV, 1)
        self.assertEqual(run_nim_vlm.EXIT_BAD_INPUT, 2)
        self.assertEqual(run_nim_vlm.EXIT_API_ERROR, 3)

        prompt = run_nim_vlm.PROMPT_VLM
        self.assertIn("Extract ALL content from this image", prompt)
        self.assertIn("describe them ONLY in prose text", prompt)
        self.assertIn("Do NOT draw ASCII art", prompt)
        self.assertIn("do NOT use backticks or code fences for diagrams", prompt)
        print("import ok")


class TestProviderInterface(unittest.TestCase):
    """Test (d): provider classes satisfy the LLMProvider protocol."""

    def test_nvidia_provider_instance_check(self):
        """NVIDIAProvider instance satisfies LLMProvider."""
        import run_nim_vlm

        # Clear env to get defaults
        with patch.dict(os.environ, {}, clear=True):
            provider = run_nim_vlm.NVIDIAProvider()
            self.assertIsInstance(provider, run_nim_vlm.LLMProvider)
            self.assertEqual(provider.api_key_env, "NVIDIA_API_KEY")
            self.assertEqual(provider.model, "meta/llama-3.2-11b-vision-instruct")
            self.assertEqual(provider.base_url, "https://integrate.api.nvidia.com/v1")
            self.assertEqual(provider.auth_scheme, "Bearer ")

    def test_openai_provider_instance_check(self):
        """OpenAICompatibleProvider instance satisfies LLMProvider."""
        import run_nim_vlm

        with patch.dict(os.environ, {
            "OPENAI_BASE_URL": "https://openrouter.ai/api/v1",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "claude-3.5-sonnet",
        }, clear=True):
            provider = run_nim_vlm.OpenAICompatibleProvider()
            self.assertIsInstance(provider, run_nim_vlm.LLMProvider)
            self.assertEqual(provider.api_key_env, "OPENAI_API_KEY")
            self.assertEqual(provider.model, "claude-3.5-sonnet")
            self.assertEqual(provider.base_url, "https://openrouter.ai/api/v1")
            self.assertEqual(provider.auth_scheme, "")

    def test_nvidia_provider_backward_compat_nim_vars(self):
        """NVIDIAProvider reads NIM_* vars as backward-compatible aliases."""
        import run_nim_vlm

        with patch.dict(os.environ, {
            "NIM_ENDPOINT": "https://custom.nim.endpoint.com/v1",
            "NIM_VLM_MODEL": "custom/nim-model",
            "NIM_SYSTEM_PROMPT": "/think",
            "NVIDIA_API_KEY": "test-key",
        }, clear=True):
            provider = run_nim_vlm.NVIDIAProvider()
            self.assertEqual(provider.base_url, "https://custom.nim.endpoint.com/v1")
            self.assertEqual(provider.model, "custom/nim-model")
            self.assertEqual(provider.default_system_prompt, "/think")

    def test_vlm_provider_new_vars_override_nim(self):
        """VLM_* vars take precedence over NIM_* vars."""
        import run_nim_vlm

        with patch.dict(os.environ, {
            "VLM_BASE_URL": "https://new.url.com/v1",
            "VLM_MODEL": "new-model",
            "NIM_ENDPOINT": "https://old.nim.com/v1",
            "NIM_VLM_MODEL": "old-model",
            "NVIDIA_API_KEY": "test-key",
        }, clear=True):
            provider = run_nim_vlm.NVIDIAProvider()
            self.assertEqual(provider.base_url, "https://new.url.com/v1")
            self.assertEqual(provider.model, "new-model")

    def test_nvidia_text_provider_uses_text_vars(self):
        """NVIDIATextProvider reads TEXT_* / NIM_TEXT_* vars."""
        import run_nim_vlm

        with patch.dict(os.environ, {
            "TEXT_BASE_URL": "https://text.url.com/v1",
            "TEXT_MODEL": "custom-text-model",
            "NIM_TEXT_ENDPOINT": "https://nim-text.url.com/v1",
            "NIM_TEXT_MODEL": "nim-text-model",
            "NVIDIA_API_KEY": "test-key",
        }, clear=True):
            provider = run_nim_vlm.NVIDIATextProvider()
            self.assertEqual(provider.base_url, "https://text.url.com/v1")
            self.assertEqual(provider.model, "custom-text-model")

    def test_make_vlm_provider_selects_nvidia(self):
        """_make_vlm_provider() returns NVIDIAProvider by default."""
        import run_nim_vlm

        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test"}, clear=True):
            provider = run_nim_vlm._make_vlm_provider()
            self.assertIsInstance(provider, run_nim_vlm.NVIDIAProvider)
            self.assertIsInstance(provider, run_nim_vlm.LLMProvider)

    def test_make_vlm_provider_selects_openai(self):
        """_make_vlm_provider() returns OpenAICompatibleProvider when VLM_PROVIDER=openai."""
        import run_nim_vlm

        with patch.dict(os.environ, {
            "VLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        }, clear=True):
            provider = run_nim_vlm._make_vlm_provider()
            self.assertIsInstance(provider, run_nim_vlm.OpenAICompatibleProvider)
            self.assertIsInstance(provider, run_nim_vlm.LLMProvider)

    def test_provider_chat_url(self):
        """Provider.chat_url() returns the full endpoint URL."""
        import run_nim_vlm

        with patch.dict(os.environ, {
            "NIM_ENDPOINT": "https://integrate.api.nvidia.com/v1",
            "NVIDIA_API_KEY": "test",
        }, clear=True):
            provider = run_nim_vlm.NVIDIAProvider()
            self.assertEqual(
                provider.chat_url(),
                "https://integrate.api.nvidia.com/v1/chat/completions",
            )

    def test_provider_make_request_posts_correct_url(self):
        """Provider.make_request() calls requests.post with the right URL."""
        import run_nim_vlm

        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": "test-key",
        }, clear=True):
            provider = run_nim_vlm.NVIDIAProvider()
            mock_response = MagicMock(status_code=200, json=lambda: {})

            with patch("requests.post", return_value=mock_response) as mock_post:
                provider.make_request(
                    content="hello",
                    stream=False,
                    model=None,
                    system_prompt=None,
                )

            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            self.assertEqual(
                args[0],
                "https://integrate.api.nvidia.com/v1/chat/completions",
            )

    def test_provider_make_request_uses_bearer_auth(self):
        """Provider.make_request() includes Bearer Authorization header."""
        import run_nim_vlm

        with patch.dict(os.environ, {
            "NVIDIA_API_KEY": "nvapi-testkey123",
        }, clear=True):
            provider = run_nim_vlm.NVIDIAProvider()
            mock_response = MagicMock(status_code=200, json=lambda: {})

            with patch("requests.post", return_value=mock_response) as mock_post:
                provider.make_request(
                    content="hello",
                    stream=True,
                    model=None,
                    system_prompt=None,
                )

            args, kwargs = mock_post.call_args
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer nvapi-testkey123")

    def test_openai_provider_uses_no_auth_scheme(self):
        """OpenAICompatibleProvider uses API key directly (no 'Bearer ' prefix)."""
        import run_nim_vlm

        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-12345",
        }, clear=True):
            provider = run_nim_vlm.OpenAICompatibleProvider()
            self.assertEqual(provider.auth_header_value, "sk-12345")


class TestProviderWithMedia(unittest.TestCase):
    """Test (e): provider-based _chat_with_media with media (mocked provider)."""

    def setUp(self):
        """Set up environment and a temporary image file."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")

        self.img_path, _ = _create_temp_image()

    def tearDown(self):
        if os.path.exists(self.img_path):
            os.remove(self.img_path)

    def test_chat_with_media_via_provider_mock(self):
        """_chat_with_media uses provider.make_request() with list content (image)."""
        import run_nim_vlm
        import shutil
        import tempfile

        mock_response = MagicMock(status_code=200, iter_lines=lambda: iter([]))

        # Mock the provider directly
        mock_provider = MagicMock(spec=run_nim_vlm.LLMProvider)
        mock_provider.make_request.return_value = mock_response
        mock_provider.api_key_env = "NVIDIA_API_KEY"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_img = shutil.copy(self.img_path, os.path.join(tmpdir, "test.png"))

            run_nim_vlm._chat_with_media(
                provider=mock_provider,
                media_files=[tmp_img],
                query=run_nim_vlm.PROMPT_VLM,
                stream=True,
                buffer=[],
            )

        mock_provider.make_request.assert_called_once()
        call_kwargs = mock_provider.make_request.call_args.kwargs
        content = call_kwargs["content"]

        # When media files are present, content must be a list
        self.assertIsInstance(content, list)
        # First element should be the text query
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[0]["text"], run_nim_vlm.PROMPT_VLM)
        # Second element should be the image
        image_item = content[1]
        self.assertEqual(image_item["type"], "image_url")
        self.assertTrue(image_item["image_url"]["url"].startswith("data:image/png;base64,"))


class TestProviderTextOnly(unittest.TestCase):
    """Test (f): provider-based _chat_text_only (mocked provider)."""

    def test_chat_text_only_via_provider_mock(self):
        """_chat_text_only uses provider.make_request() with string content."""
        import run_nim_vlm

        mock_response = MagicMock(status_code=200, iter_lines=lambda: iter([]))

        mock_provider = MagicMock(spec=run_nim_vlm.LLMProvider)
        mock_provider.make_request.return_value = mock_response
        mock_provider.api_key_env = "NVIDIA_API_KEY"

        run_nim_vlm._chat_text_only(
            provider=mock_provider,
            query="Improve this markdown:\n\n## Hello",
            stream=True,
        )

        mock_provider.make_request.assert_called_once()
        call_kwargs = mock_provider.make_request.call_args.kwargs
        content = call_kwargs["content"]

        # Text-only: content must be a plain string
        self.assertIsInstance(content, str)
        self.assertEqual(content, "Improve this markdown:\n\n## Hello")

    def test_chat_text_only_passes_model_and_prompt(self):
        """_chat_text_only passes model and system_prompt to provider.make_request()."""
        import run_nim_vlm

        mock_response = MagicMock(status_code=200, iter_lines=lambda: iter([]))

        mock_provider = MagicMock(spec=run_nim_vlm.LLMProvider)
        mock_provider.make_request.return_value = mock_response
        mock_provider.api_key_env = "NVIDIA_API_KEY"

        run_nim_vlm._chat_text_only(
            provider=mock_provider,
            query="test query",
            stream=True,
            model="custom-model",
            system_prompt="/think",
        )

        call_kwargs = mock_provider.make_request.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "custom-model")
        self.assertEqual(call_kwargs["system_prompt"], "/think")


class TestPostProcessFlag(unittest.TestCase):
    """Test (g): --post-process flag triggers two-pass flow (mocked provider)."""

    def setUp(self):
        """Create a temporary image file."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("PIL not available")
        self.img_path, _ = _create_temp_image()

    def tearDown(self):
        if os.path.exists(self.img_path):
            os.remove(self.img_path)

    def test_post_process_flag_enables_two_passes(self):
        """With --post-process, two provider.make_request() calls per page (VLM + text)."""
        import run_nim_vlm
        import shutil
        import tempfile

        vlm_response = MagicMock(status_code=200, iter_lines=lambda: iter([]))
        text_response = MagicMock(status_code=200, iter_lines=lambda: iter([]))

        call_count = [0]

        def fake_make_request(content: Any, stream: bool, model: str | None, system_prompt: str | None, gen_overrides: Any = None):
            call_count[0] += 1
            if call_count[0] == 1:
                return vlm_response
            return text_response

        mock_vlm_provider = MagicMock(spec=run_nim_vlm.LLMProvider)
        mock_vlm_provider.make_request.side_effect = fake_make_request
        mock_vlm_provider.api_key_env = "NVIDIA_API_KEY"

        mock_text_provider = MagicMock(spec=run_nim_vlm.LLMProvider)
        mock_text_provider.make_request.side_effect = fake_make_request
        mock_text_provider.api_key_env = "NIM_TEXT_API_KEY"

        with patch.object(run_nim_vlm, "_make_vlm_provider", return_value=mock_vlm_provider):
            with patch.object(run_nim_vlm, "_make_text_provider", return_value=mock_text_provider):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_img = shutil.copy(self.img_path, os.path.join(tmpdir, "test.png"))

                    with patch.object(sys, "argv", [
                        "run_nim_vlm.py",
                        tmp_img,
                        "--post-process",
                    ]):
                        with patch.dict(os.environ, {
                            "NVIDIA_API_KEY": "test_key",
                            "NIM_TEXT_API_KEY": "test_text_key",
                        }, clear=False):
                            run_nim_vlm.main()

        # Two passes: VLM + text (--post-process enables pass 2)
        self.assertEqual(call_count[0], 2)

    def test_no_post_process_flag_single_pass(self):
        """Without --post-process (default), only the VLM pass runs."""
        import run_nim_vlm
        import shutil
        import tempfile

        vlm_response = MagicMock(status_code=200, iter_lines=lambda: iter([]))

        call_count = [0]

        def fake_make_request(content: Any, stream: bool, model: str | None, system_prompt: str | None, gen_overrides: Any = None):
            call_count[0] += 1
            return vlm_response

        mock_vlm_provider = MagicMock(spec=run_nim_vlm.LLMProvider)
        mock_vlm_provider.make_request.side_effect = fake_make_request
        mock_vlm_provider.api_key_env = "NVIDIA_API_KEY"

        with patch.object(run_nim_vlm, "_make_vlm_provider", return_value=mock_vlm_provider):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_img = shutil.copy(self.img_path, os.path.join(tmpdir, "test.png"))

                with patch.object(sys, "argv", [
                    "run_nim_vlm.py",
                    tmp_img,
                ]):
                    with patch.dict(os.environ, {"NVIDIA_API_KEY": "test_key"}, clear=False):
                        run_nim_vlm.main()

        # Only VLM pass — post-processing is disabled by default
        self.assertEqual(call_count[0], 1)

    def test_post_process_without_text_api_key_skips_pass2(self):
        """With --post-process but no text API key, only VLM pass runs."""
        import run_nim_vlm
        import shutil
        import tempfile

        vlm_response = MagicMock(status_code=200, iter_lines=lambda: iter([]))

        call_count = [0]

        def fake_make_request(content: Any, stream: bool, model: str | None, system_prompt: str | None, gen_overrides: Any = None):
            call_count[0] += 1
            return vlm_response

        mock_vlm_provider = MagicMock(spec=run_nim_vlm.LLMProvider)
        mock_vlm_provider.make_request.side_effect = fake_make_request
        mock_vlm_provider.api_key_env = "NVIDIA_API_KEY"

        # Text provider has no API key
        mock_text_provider = MagicMock(spec=run_nim_vlm.LLMProvider)
        mock_text_provider.api_key_env = "NIM_TEXT_API_KEY"  # will return "" from env

        with patch.object(run_nim_vlm, "_make_vlm_provider", return_value=mock_vlm_provider):
            with patch.object(run_nim_vlm, "_make_text_provider", return_value=mock_text_provider):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_img = shutil.copy(self.img_path, os.path.join(tmpdir, "test.png"))

                    with patch.object(sys, "argv", [
                        "run_nim_vlm.py",
                        tmp_img,
                    ]):
                        # No TEXT_API_KEY set — should skip pass 2
                        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test_key"}, clear=False):
                            run_nim_vlm.main()

        # Only VLM pass — TEXT_API_KEY not set, so pass 2 is skipped
        self.assertEqual(call_count[0], 1)


if __name__ == "__main__":
    unittest.main()