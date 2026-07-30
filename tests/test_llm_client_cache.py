"""The unified client's opt-in disk cache is deterministic and never re-calls."""
from __future__ import annotations

import pytest

pytest.importorskip("anthropic")

from src.llm.anthropic_client import AnthropicClient


class _ExplodingClient:
    """Stand-in for anthropic.Anthropic — fails if the API is ever touched."""

    class messages:
        @staticmethod
        def create(**kwargs):
            raise AssertionError("API called despite a warm cache")


def _client(tmp_path):
    c = AnthropicClient(api_key="test", cache_dir=str(tmp_path), temperature=0.0)
    c._client = _ExplodingClient()          # any API call now raises
    return c


def test_a_warm_cache_is_returned_without_calling_the_api(tmp_path):
    c = _client(tmp_path)
    path = c._cache_path("sys", "usr", None)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('{"cached": true}')
    assert c.complete(system="sys", user="usr") == '{"cached": true}'
    assert c.cached_calls == 1
    assert c.calls == 0


def test_cache_key_separates_tool_name_temperature_and_prompt(tmp_path):
    c = AnthropicClient(api_key="test", cache_dir=str(tmp_path), temperature=0.0)
    base = c._cache_path("sys", "usr", None)
    assert base != c._cache_path("sys", "usr", "submit_view")   # tool name matters
    assert base != c._cache_path("sys", "other", None)          # prompt matters
    warm = AnthropicClient(api_key="test", cache_dir=str(tmp_path), temperature=0.7)
    assert base != warm._cache_path("sys", "usr", None)         # temperature matters
    # Same inputs → same path (deterministic reruns).
    assert base == c._cache_path("sys", "usr", None)


def test_no_cache_dir_means_no_cache_path():
    c = AnthropicClient(api_key="test")
    assert c._cache_path("sys", "usr", None) is None


def test_preflight_passes_cache_dir_through(tmp_path, monkeypatch):
    """The PM runners opt into the disk cache via ``preflight_llm``; the analyst
    path passes nothing and must stay byte-identical (cache_dir=None)."""
    import src.layered.analysts.build as build

    received = {}

    class _FakeClient:
        def __init__(self, **kw):
            received.update(kw)

        def validate(self):
            return None

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr("src.llm.anthropic_client.AnthropicClient", _FakeClient)

    build.preflight_llm("m", max_tokens=100, cache_dir=str(tmp_path))
    assert received["cache_dir"] == str(tmp_path)

    received.clear()
    build.preflight_llm("m", max_tokens=100)
    assert received["cache_dir"] is None
