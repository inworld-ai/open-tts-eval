import io
import json
import struct
import wave
from dataclasses import replace
from pathlib import Path

import pytest

from tts_assess.config import AssessmentConfig
from tts_assess.pipeline import run_assessment
from tts_assess.reporting.compare import run_comparison
from tts_assess.sampling.providers import GradiumProvider, available_providers, build_provider
from tts_assess.sampling.providers.base import ProviderError, SynthesisRequest
from tts_assess.sampling.sampler import SamplingConfig, run_sampling


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, headers, body, timeout))
        return self.responses.pop(0)


def _wav(sample_rate=48000, frames=4800):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(b"\x00\x01" * frames)
    return buffer.getvalue()


def _streaming_wav():
    audio = bytearray(_wav())
    struct.pack_into("<I", audio, 4, 0xFFFFFFFF)
    struct.pack_into("<I", audio, 40, 0xFFFFFFFF)
    return bytes(audio)


REQ = SynthesisRequest(text="Hello world.", voice_id="YTpq7expH9539ERJ", model_id="")


def test_gradium_registered_and_requires_key():
    assert "gradium" in available_providers()
    assert isinstance(build_provider("gradium", "k"), GradiumProvider)
    with pytest.raises(ValueError, match="API key"):
        GradiumProvider("")


@pytest.mark.parametrize("sample_rate", [48000, 24000])
def test_synthesize_payload_and_actual_audio_format(sample_rate):
    audio = _wav(sample_rate)
    transport = FakeTransport([(200, audio)])
    provider = GradiumProvider("key", timeout=15, transport=transport)
    result = provider.synthesize(replace(REQ, audio_encoding="MP3", sample_rate_hz=16000))
    assert result.audio == audio
    assert result.audio_encoding == provider.output_encoding("MP3") == "WAV"
    assert result.sample_rate_hz == sample_rate
    method, url, headers, body, timeout = transport.calls[0]
    assert method == "POST" and url == "https://api.gradium.ai/api/post/speech/tts"
    assert headers["x-api-key"] == "key" and timeout == 15
    assert json.loads(body) == {
        "text": REQ.text,
        "voice_id": REQ.voice_id,
        "model_name": "default",
        "output_format": "wav",
        "only_audio": True,
    }


@pytest.mark.parametrize("temperature", [0.0, 0.7, 1.5])
def test_model_temperature_and_custom_voice(temperature):
    transport = FakeTransport([(200, _wav())])
    result = GradiumProvider("k", transport=transport).synthesize(
        replace(REQ, model_id="custom-model", voice_id="custom-uid", temperature=temperature)
    )
    body = json.loads(transport.calls[0][3])
    assert body["model_name"] == "custom-model" and body["voice_id"] == "custom-uid"
    assert isinstance(body["json_config"], str)
    assert json.loads(body["json_config"]) == {"temp": temperature}
    assert result.request_sent == {key: value for key, value in body.items() if key != "text"}


@pytest.mark.parametrize("temperature", [-0.1, 1.6, float("nan"), float("inf")])
def test_invalid_temperature_rejected_before_http(temperature):
    transport = FakeTransport([])
    with pytest.raises(ProviderError, match="temperature"):
        GradiumProvider("k", transport=transport).synthesize(replace(REQ, temperature=temperature))
    assert not transport.calls


def test_speaking_rate_rejected_before_http():
    transport = FakeTransport([])
    with pytest.raises(ProviderError, match="speaking_rate"):
        GradiumProvider("k", transport=transport).synthesize(replace(REQ, speaking_rate=1.0))
    assert not transport.calls


@pytest.mark.parametrize(
    "audio",
    [
        b"",
        b"not a wav",
        _wav(frames=0),
        _wav()[:-2],
        _wav()[:20],
        b"RIFF" + struct.pack("<I", 36) + b"WAVEJUNK" + struct.pack("<I", 100) + b"x",
    ],
)
def test_invalid_audio_rejected(audio):
    with pytest.raises(ProviderError, match="invalid WAV"):
        GradiumProvider("k", transport=FakeTransport([(200, audio)])).synthesize(REQ)


def test_streaming_wav_with_unknown_length():
    audio = _streaming_wav()
    result = GradiumProvider("k", transport=FakeTransport([(200, audio)])).synthesize(REQ)
    # The 0xFFFFFFFF placeholder sizes are rewritten with the real PCM length.
    assert result.audio == _wav() and result.sample_rate_hz == 48000
    with wave.open(io.BytesIO(result.audio), "rb") as fixed:
        assert fixed.getnframes() == 4800
    with pytest.raises(ProviderError, match="incomplete frame"):
        GradiumProvider("k", transport=FakeTransport([(200, audio[:-1])])).synthesize(REQ)


def test_invalid_wav_sample_rate():
    audio = bytearray(_wav())
    struct.pack_into("<I", audio, 24, 0)
    with pytest.raises(ProviderError, match="invalid WAV"):
        GradiumProvider("k", transport=FakeTransport([(200, bytes(audio))])).synthesize(REQ)


@pytest.mark.parametrize("offset", [22, 34])
def test_invalid_wav_channels_or_sample_width(offset):
    audio = bytearray(_streaming_wav())
    struct.pack_into("<H", audio, offset, 0)
    with pytest.raises(ProviderError, match="invalid WAV"):
        GradiumProvider("k", transport=FakeTransport([(200, bytes(audio))])).synthesize(REQ)


@pytest.mark.parametrize("operation", ["synthesize", "list_voices"])
def test_http_errors_and_bounded_retry(operation, monkeypatch):
    monkeypatch.setattr("tts_assess.sampling.providers._http.time.sleep", lambda _: None)
    for status, attempts in [(401, 1), (429, 2), (503, 2)]:
        transport = FakeTransport([(status, b"error")] * attempts)
        provider = GradiumProvider("k", transport=transport, max_retries=2)
        with pytest.raises(ProviderError, match=f"HTTP {status}"):
            if operation == "synthesize":
                provider.synthesize(REQ)
            else:
                provider.list_voices()
        assert len(transport.calls) == attempts


def test_transient_error_recovers(monkeypatch):
    monkeypatch.setattr("tts_assess.sampling.providers._http.time.sleep", lambda _: None)
    transport = FakeTransport([(503, b"busy"), (200, _wav())])
    assert GradiumProvider("k", transport=transport).synthesize(REQ).audio == _wav()
    assert len(transport.calls) == 2


def test_list_public_voices_only():
    entries = [
        {"uid": "public", "name": "Emma", "language": "en", "is_catalog": True},
        {"uid": "unknown-language", "name": "Other", "language": None, "is_catalog": True},
        {"uid": "private", "name": "Clone", "language": "en", "is_catalog": False},
    ]
    transport = FakeTransport([(200, json.dumps(entries).encode())])
    voices = GradiumProvider("k", timeout=15, transport=transport).list_voices()
    assert [voice.voice_id for voice in voices] == ["public", "unknown-language"]
    assert voices[0].languages == ("en",) and voices[0].metadata == entries[0]
    assert voices[1].languages == ()
    method, url, headers, body, timeout = transport.calls[0]
    assert method == "GET" and body is None and timeout == 15
    assert url == "https://api.gradium.ai/api/voices/?include_catalog=true&limit=0"
    assert headers["x-api-key"] == "k"


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"{}",
        b"[null]",
        b"[{}]",
        b'[{"uid":"a","name":"A","is_catalog":"true"}]',
        b'[{"uid":"a","name":"A","is_catalog":true,"language":3}]',
    ],
)
def test_invalid_voice_response(body):
    with pytest.raises(ProviderError, match="Gradium"):
        GradiumProvider("k", transport=FakeTransport([(200, body)])).list_voices()


def test_empty_voice_catalog():
    assert GradiumProvider("k", transport=FakeTransport([(200, b"[]")])).list_voices() == []


def test_sampling_assessment_and_comparison(tmp_path):
    dataset = tmp_path / "texts.txt"
    dataset.write_text("Hello world.\n")
    transport = FakeTransport([(200, _wav()), (200, _streaming_wav())])
    provider = build_provider("gradium", "k", transport=transport)
    summary = run_sampling(
        dataset,
        tmp_path / "samples",
        provider,
        [("default", "first"), ("default", "second")],
        voices=[REQ.voice_id],
        config=SamplingConfig(audio_encoding="MP3", concurrency=1),
    )
    config = AssessmentConfig.default()
    config.asr.backend = "mock"
    config.reporting.bootstrap_resamples = 20
    runs = []
    for run in summary["runs"]:
        assert run["samples"] == 1 and run["errors"] == 0
        directory = Path(run["run_dir"])
        manifest = Path(run["manifest"])
        entry = json.loads(manifest.read_text())
        assert entry["audio_path"].endswith(".wav")
        assert entry["metadata"]["sample_rate_hz"] == 48000
        assert entry["metadata"]["requested_sample_rate_hz"] == 24000
        assert entry["metadata"]["sent_request"] == {
            "voice_id": REQ.voice_id,
            "model_name": "default",
            "output_format": "wav",
            "only_audio": True,
        }
        rows, report = run_assessment(manifest, directory, config, use_cache=False)
        assert report["sample_count"] == 1 and rows[0]["wer"] == 0
        assert (directory / "report.html").exists()
        runs.append(directory)
    comparison = run_comparison(runs, tmp_path / "comparison", config)
    assert comparison["labels"] == ["gradium-first", "gradium-second"]
    assert (tmp_path / "comparison" / "comparison.html").exists()
