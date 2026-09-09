from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

from tts_assess.config import load_config, write_default_config
from tts_assess.pipeline import run_assessment
from tts_assess.reporting.compare import run_comparison
from tts_assess.reporting.preview import serve_report
from tts_assess.sampling.providers import build_provider
from tts_assess.sampling.sampler import SamplingConfig, run_sampling

app = typer.Typer(help="Standalone offline TTS audio assessment toolkit.")
console = Console()


def _resolve_api_key(api_key_file: Path | None, api_key_env: str) -> str:
    """Return the provider API key from a file (preferred) or an env var."""
    if api_key_file is not None:
        try:
            key = api_key_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise typer.BadParameter(f"cannot read API key file {api_key_file}: {exc}") from exc
        if not key:
            raise typer.BadParameter(f"API key file {api_key_file} is empty")
        return key
    key = os.environ.get(api_key_env)
    if not key:
        raise typer.BadParameter(
            f"no API key: pass --api-key-file or set the {api_key_env} environment variable"
        )
    return key


@app.command()
def run(
    manifest: Path = typer.Argument(..., help="Path to a JSONL or CSV manifest."),
    output_dir: Path = typer.Option(Path("tts_assess_results"), "--output-dir", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    allow_manifest_errors: bool = typer.Option(
        False, help="Continue when manifest paths are missing."
    ),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", help="Measurement cache dir (default: <output_dir>/.measure_cache)."
    ),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the measurement cache."),
) -> None:
    """Assess audio files and write JSON, CSV, and HTML reports."""
    assessment_config = load_config(config)
    rows, summary = run_assessment(
        manifest,
        output_dir,
        assessment_config,
        fail_on_manifest_errors=not allow_manifest_errors,
        use_cache=not no_cache,
        cache_dir=cache_dir,
    )
    console.print(f"[green]Assessed {len(rows)} samples[/green]")
    console.print(f"Results: {output_dir / 'results.jsonl'}")
    console.print(f"Summary: {output_dir / 'summary.json'}")
    if assessment_config.reporting.html:
        console.print(f"Report: {output_dir / 'report.html'}")
    if "cache" in summary:
        console.print(
            f"Cache: {summary['cache']['hits']} hits, {summary['cache']['misses']} computed"
        )
    console.print(f"Pass rate: {summary['pass_rate']:.1%}")


@app.command()
def compare(
    runs: list[Path] = typer.Argument(
        ..., help="Two or more run output dirs (or results.jsonl files) to compare."
    ),
    output_dir: Path = typer.Option(Path("tts_assess_comparison"), "--output-dir", "-o"),
    config: Path | None = typer.Option(None, "--config", "-c"),
    label: list[str] = typer.Option(
        [], "--label", "-l", help="Run label, in run order (repeat once per run)."
    ),
    title: str | None = typer.Option(None, "--title", help="Comparison report title."),
) -> None:
    """Merge several assessment runs into one comparative report."""
    assessment_config = load_config(config)
    run_comparison(
        list(runs),
        output_dir,
        assessment_config,
        labels=list(label) or None,
        title=title,
    )
    console.print(f"[green]Compared {len(runs)} runs[/green]")
    console.print(f"Report: {output_dir / 'comparison.html'}")
    console.print(f"Data: {output_dir / 'comparison.json'}")


@app.command()
def sample(
    dataset: Path = typer.Argument(
        ..., help="Text dataset: .txt (one per line), .jsonl, or .csv with a 'text' column."
    ),
    output_dir: Path = typer.Option(Path("tts_samples"), "--output-dir", "-o"),
    provider: str = typer.Option("inworld", "--provider", "-p"),
    model: list[str] = typer.Option(
        [], "--model", "-m", help="Model id (repeatable). Defaults to the provider default."
    ),
    voice: list[str] = typer.Option([], "--voice", "-v", help="Prebuilt voice id (repeatable)."),
    num_voices: int | None = typer.Option(
        None, "--num-voices", help="Use N voices from the provider catalog."
    ),
    shuffle_voices: bool = typer.Option(
        False, "--shuffle-voices", help="With --num-voices, pick N random (seeded) voices."
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for --shuffle-voices."),
    language: str = typer.Option("en-US", "--language", help="BCP-47 language tag."),
    audio_format: str = typer.Option(
        "WAV", "--format", help="WAV, LINEAR16, MP3, FLAC, OGG_OPUS, PCM, ALAW, or MULAW."
    ),
    sample_rate: int = typer.Option(24000, "--sample-rate"),
    speaking_rate: float | None = typer.Option(None, "--speaking-rate"),
    temperature: float | None = typer.Option(None, "--temperature"),
    concurrency: int = typer.Option(4, "--concurrency"),
    limit: int | None = typer.Option(None, "--limit", help="Only sample the first N texts."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Re-synthesize existing audio."),
    allow_model_mismatch: bool = typer.Option(
        False,
        "--allow-model-mismatch",
        help="Keep clips the provider reports as served by a different model than requested "
        "(recorded as metadata.returned_model). By default they are logged as errors.",
    ),
    api_key_file: Path | None = typer.Option(
        None, "--api-key-file", help="Read the API key from this file instead of the environment."
    ),
    api_key_env: str = typer.Option("INWORLD_API_KEY", "--api-key-env"),
) -> None:
    """Synthesize a text dataset into eval-ready manifests (one run per model)."""
    api_key = _resolve_api_key(api_key_file, api_key_env)
    tts_provider = build_provider(provider, api_key)
    config = SamplingConfig(
        language=language,
        audio_encoding=audio_format.upper(),
        sample_rate_hz=sample_rate,
        speaking_rate=speaking_rate,
        temperature=temperature,
        concurrency=concurrency,
        overwrite=overwrite,
        allow_model_mismatch=allow_model_mismatch,
    )
    with Progress(transient=True, console=console) as progress:
        task = progress.add_task("Synthesizing", total=None)
        summary = run_sampling(
            dataset,
            output_dir,
            tts_provider,
            model or None,
            voices=voice or None,
            num_voices=num_voices,
            shuffle_voices=shuffle_voices,
            seed=seed,
            config=config,
            limit=limit,
            progress_cb=lambda _event: progress.advance(task, 1),
        )
    total_samples = sum(run["samples"] for run in summary["runs"])
    total_errors = sum(run["errors"] for run in summary["runs"])
    console.print(
        f"[green]Sampled {total_samples} clips[/green] across {len(summary['runs'])} model(s) "
        f"and {len(summary['voices'])} voice(s)"
    )
    if total_errors:
        console.print(f"[red]{total_errors} synthesis errors[/red] (see sampling_meta.json)")
    for run in summary["runs"]:
        console.print(f"  {run['model']}: {run['manifest']}")


@app.command("voices")
def voices_command(
    provider: str = typer.Option("inworld", "--provider", "-p"),
    api_key_file: Path | None = typer.Option(
        None, "--api-key-file", help="Read the API key from this file instead of the environment."
    ),
    api_key_env: str = typer.Option("INWORLD_API_KEY", "--api-key-env"),
) -> None:
    """List the provider's prebuilt voices."""
    api_key = _resolve_api_key(api_key_file, api_key_env)
    catalog = build_provider(provider, api_key).list_voices()
    for entry in catalog:
        languages = ", ".join(entry.languages)
        console.print(f"{entry.voice_id}\t{entry.name or ''}\t{languages}")
    console.print(f"[green]{len(catalog)} voices[/green]")


@app.command("init-config")
def init_config(path: Path = typer.Argument(Path("tts-assess.yml"))) -> None:
    """Write a default editable YAML config."""
    write_default_config(path)
    console.print(f"[green]Wrote {path}[/green]")


@app.command()
def preview(
    report: Path = typer.Argument(..., help="Path to report.html"),
    port: int = typer.Option(8042, "--port"),
    open_browser: bool = typer.Option(True, "--open-browser/--no-open-browser"),
) -> None:
    """Serve report.html locally with live reload."""
    serve_report(report, port=port, open_browser=open_browser)
