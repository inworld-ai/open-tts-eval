"""Sampling provenance: unique files, actual audio properties, model identity, sent controls."""

import io
import json
import wave

import pytest
import soundfile as sf

from tts_assess.sampling.providers.base import (
    ProviderError,
    SynthesisRequest,
    SynthesisResult,
    TTSProvider,
    Voice,
)
from tts_assess.sampling.providers.elevenlabs import ElevenLabsProvider
from tts_assess.sampling.providers.hume import HumeProvider
from tts_assess.sampling.providers.inworld import InworldProvider
from tts_assess.sampling.sampler import SamplingConfig, _unique_filenames, run_sampling


def _wav(seconds: float, sample_rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frame = int(0.1 * 32767).to_bytes(2, "little", signed=True)
        handle.writeframes(frame * int(sample_rate * seconds))
    return buffer.getvalue()


class TextLengthProvider(TTSProvider):
    """Audio duration encodes the text length, so a clip can be traced to its text."""

    name = "fake"
    default_model = "fake-1"

    def __init__(self, *, served_model=None, actual_rate=24000):
        self.served_model = served_model
        self.actual_rate = actual_rate
        self.requests = []

    def synthesize(self, request):
        self.requests.append(request)
        return SynthesisResult(
            audio=_wav(0.1 * len(request.text), sample_rate=self.actual_rate),
            audio_encoding="WAV",
            sample_rate_hz=request.sample_rate_hz,
            model_id=self.served_model,
        )

    def list_voices(self):
        return [Voice("v")]


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append({"method": method, "url": url, "headers": headers, "body": body})
        return self.responses.pop(0)


# --- H4: filename collisions ------------------------------------------------------------


def test_unique_filenames_disambiguate_colliding_ids():
    names = _unique_filenames(["v__a/b", "v__a:b", "v__plain"], "wav")
    assert names["v__plain"] == "v__plain.wav"
    assert names["v__a/b"] != names["v__a:b"]
    assert names["v__a/b"].startswith("v__a_b-") and names["v__a:b"].startswith("v__a_b-")
    with pytest.raises(ValueError, match="duplicate sample ids"):
        _unique_filenames(["v__x", "v__x"], "wav")


def test_colliding_text_ids_keep_their_own_audio(tmp_path):
    dataset = tmp_path / "texts.jsonl"
    dataset.write_text(
        json.dumps({"id": "a/b", "text": "short"}) + "\n"
        + json.dumps({"id": "a:b", "text": "a much longer sentence here"}) + "\n"
    )
    provider = TextLengthProvider()
    run_sampling(dataset, tmp_path / "out", provider, ["m1"], voices=["v"])

    rows = [
        json.loads(line)
        for line in (tmp_path / "out" / "fake-m1" / "manifest.jsonl").read_text().splitlines()
    ]
    assert len({row["audio_path"] for row in rows}) == 2
    for row in rows:
        info = sf.info(tmp_path / "out" / "fake-m1" / row["audio_path"])
        assert info.frames / info.samplerate == pytest.approx(0.1 * len(row["text"]), abs=1e-3)

    # Resuming makes no provider calls and keeps the association.
    provider.requests.clear()
    run_sampling(dataset, tmp_path / "out", provider, ["m1"], voices=["v"])
    assert provider.requests == []


def test_model_labels_sharing_a_run_dir_are_rejected(tmp_path):
    dataset = tmp_path / "texts.txt"
    dataset.write_text("hi\n")
    with pytest.raises(ValueError, match="collide"):
        run_sampling(
            dataset, tmp_path / "out", TextLengthProvider(), ["org/a", "org:a"], voices=["v"]
        )


# --- H5: recorded model and audio properties --------------------------------------------


def test_returned_model_mismatch_is_an_error_unless_allowed(tmp_path):
    dataset = tmp_path / "texts.txt"
    dataset.write_text("hello\n")
    provider = TextLengthProvider(served_model="fake-2")

    summary = run_sampling(dataset, tmp_path / "out", provider, ["fake-1"], voices=["v"])
    run = summary["runs"][0]
    assert run["samples"] == 0 and run["errors"] == 1
    meta = json.loads((tmp_path / "out" / "fake-fake-1" / "sampling_meta.json").read_text())
    assert "served model 'fake-2'" in meta["errors"][0]["error"]

    summary = run_sampling(
        dataset, tmp_path / "out2", provider, ["fake-1"], voices=["v"],
        config=SamplingConfig(allow_model_mismatch=True),
    )
    assert summary["runs"][0]["samples"] == 1
    row = json.loads((tmp_path / "out2" / "fake-fake-1" / "manifest.jsonl").read_text())
    assert row["metadata"]["api_model"] == "fake-1"
    assert row["metadata"]["returned_model"] == "fake-2"
    meta = json.loads((tmp_path / "out2" / "fake-fake-1" / "sampling_meta.json").read_text())
    assert meta["returned_models"] == ["fake-2"]


def test_metadata_records_probed_audio_properties_not_the_nominal_ones(tmp_path):
    dataset = tmp_path / "texts.txt"
    dataset.write_text("hello\n")
    provider = TextLengthProvider(actual_rate=22050)
    run_sampling(
        dataset, tmp_path / "out", provider, ["m"], voices=["v"],
        config=SamplingConfig(sample_rate_hz=24000),
    )
    row = json.loads((tmp_path / "out" / "fake-m" / "manifest.jsonl").read_text())
    meta = row["metadata"]
    assert meta["requested_sample_rate_hz"] == 24000
    assert meta["sample_rate_hz"] == 22050
    assert meta["channels"] == 1
    assert meta["duration_sec"] == pytest.approx(0.5, abs=1e-3)
    assert "sent_request" in meta


# --- H5.2 / H5.3: providers send what they record ---------------------------------------


def test_elevenlabs_sends_speed_and_language_only_where_supported():
    audio = _wav(0.1)
    t = FakeTransport([(200, audio), (200, audio)])
    p = ElevenLabsProvider("k", transport=t)
    turbo = SynthesisRequest(
        text="Hi.", voice_id="V", model_id="eleven_turbo_v2_5", language="en-US",
        speaking_rate=1.1,
    )
    result = p.synthesize(turbo)
    body = json.loads(t.calls[0]["body"])
    assert body["voice_settings"] == {"speed": 1.1}
    assert body["language_code"] == "en"
    assert result.request_sent["language_code"] == "en" and "text" not in result.request_sent

    multilingual = SynthesisRequest(
        text="Hi.", voice_id="V", model_id="eleven_multilingual_v2", language="en-US"
    )
    p.synthesize(multilingual)
    body = json.loads(t.calls[1]["body"])
    assert "language_code" not in body and "voice_settings" not in body


def test_elevenlabs_rejects_temperature():
    p = ElevenLabsProvider("k", transport=FakeTransport([]))
    with pytest.raises(ProviderError, match="does not support temperature"):
        p.synthesize(SynthesisRequest(text="x", voice_id="V", model_id="m", temperature=0.5))


def test_hume_pins_version_and_sends_speed():
    audio = _wav(0.1)
    t = FakeTransport([(200, audio), (200, audio)])
    p = HumeProvider("k", transport=t)
    result = p.synthesize(
        SynthesisRequest(text="Hi.", voice_id="Ava", model_id="octave-1", speaking_rate=0.9)
    )
    body = json.loads(t.calls[0]["body"])
    assert body["version"] == "1"
    assert body["utterances"][0]["speed"] == 0.9
    assert result.model_id == "octave-1"
    assert result.request_sent["utterances"][0] == {
        "voice": {"name": "Ava", "provider": "HUME_AI"}, "speed": 0.9,
    }

    p.synthesize(SynthesisRequest(text="Hi.", voice_id="Ava", model_id=""))
    assert json.loads(t.calls[1]["body"])["version"] == "2"  # provider default, explicit


def test_hume_rejects_temperature():
    p = HumeProvider("k", transport=FakeTransport([]))
    with pytest.raises(ProviderError, match="does not support temperature"):
        p.synthesize(SynthesisRequest(text="x", voice_id="V", model_id="octave-2", temperature=1.0))


def test_inworld_reports_served_model_and_sent_request():
    import base64

    payload = {
        "audioContent": base64.b64encode(b"RIFFwav").decode(),
        "usage": {"processedCharactersCount": 3, "modelId": "inworld-tts-2"},
    }
    transport = FakeTransport([(200, json.dumps(payload).encode())])
    provider = InworldProvider("k", transport=transport)
    result = provider.synthesize(
        SynthesisRequest(
            text="Hi", voice_id="Sarah", model_id="inworld-tts-1-max", speaking_rate=1.2
        )
    )
    assert result.model_id == "inworld-tts-2"
    assert result.request_sent["modelId"] == "inworld-tts-1-max"
    assert result.request_sent["audioConfig"]["speakingRate"] == 1.2
    assert "text" not in result.request_sent
