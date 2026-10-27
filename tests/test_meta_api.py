"""Muse Spark client without network: a fake OpenAI client checks request shape and failure handling."""

import time
from types import SimpleNamespace

import numpy as np

from streetsense.config import VisionLLMConfig
from streetsense.meta_api import MetaVisionAssistant, encode_jpeg_data_url


class FakeCompletions:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply, exc, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


def make(monkeypatch, comp):
    monkeypatch.setenv("MODEL_API_KEY", "test-key")
    a = MetaVisionAssistant(VisionLLMConfig(interval_s=5))
    a._client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    return a


def frame():
    return np.zeros((480, 640, 3), np.uint8)


def test_disabled_without_key(monkeypatch):
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    monkeypatch.delenv("LLAMA_API_KEY", raising=False)
    a = MetaVisionAssistant(VisionLLMConfig())
    assert not a.enabled
    assert a.maybe_request(frame(), now=100) is False
    assert "MODEL_API_KEY" in a.ask_sync(frame(), "q")["error"]


def test_image_goes_in_user_message(monkeypatch):
    comp = FakeCompletions('{"summary": "crossing ahead", "safe_to_cross": "unsure"}')
    a = make(monkeypatch, comp)
    out = a.ask_sync(frame(), "Can I cross?")
    assert out["safe_to_cross"] == "unsure"
    msgs = comp.calls[0]["messages"]
    assert comp.calls[0]["model"] == "muse-spark-1.3"
    assert comp.calls[0].get("reasoning_effort") == "minimal"
    assert msgs[0]["role"] == "system" and isinstance(msgs[0]["content"], str)  # no image in system
    parts = msgs[1]["content"]
    assert msgs[1]["role"] == "user"
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_background_rate_limit_and_errors_do_not_raise(monkeypatch):
    comp = FakeCompletions(exc=RuntimeError("quota"))
    a = make(monkeypatch, comp)
    assert a.maybe_request(frame(), now=0)
    for _ in range(50):
        if a.error:
            break
        time.sleep(0.01)
    assert "quota" in a.error
    assert a.maybe_request(frame(), now=1) is False  # interval not elapsed
    assert a.maybe_request(frame(), now=6) is True


def test_jpeg_downscaled():
    big = np.zeros((2000, 3000, 3), np.uint8)
    url = encode_jpeg_data_url(big, max_side=768)
    import base64

    import cv2

    img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert max(img.shape[:2]) == 768
