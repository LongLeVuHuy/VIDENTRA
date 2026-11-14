"""Ingestion job execution, attempt staging, retry policy, and atomic publishing."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any
import uuid

import pyarrow.parquet as pq

from videntra.config.ingestion import IngestionConfig
from videntra.ingestion.probe import probe_media, ProbeResult, ProbeError
from videntra.ingestion.decode import decode_frames, DecodeError, DecodedFrame
from videntra.registry.access import MediaResolver
from videntra.registry.store import sha256_file


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def compute_config_hash(cfg: IngestionConfig | dict[str, Any]) -> str:
    d = cfg.to_dict() if isinstance(cfg, IngestionConfig) else cfg
    return hashlib.sha256(canonical_json(d).encode("utf-8")).hexdigest()


def compute_job_key(asset_hash: str, stage_id: str, config_hash: str) -> str:
    token = f"{asset_hash}:{stage_id}:{config_hash}"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def safe_path_id(name: str) -> str:
    import re
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def verify_artifact_parquet(path: Path | str, expected_min_rows: int = 0) -> int:
    """Verify that a file is a valid non-empty Parquet file and return row count."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        raise ValueError(f"Parquet artifact {p.name} does not exist or is empty")
    pf = pq.ParquetFile(str(p))
    num_rows = pf.metadata.num_rows
    if num_rows < expected_min_rows:
        raise ValueError(f"Parquet artifact {p.name} has {num_rows} rows < {expected_min_rows}")
    return num_rows


def verify_artifact_jsonl(path: Path | str) -> int:
    """Verify that a JSONL file exists and contains valid JSON lines."""
    p = Path(path)
    if not p.is_file() or p.stat().st_size == 0:
        raise ValueError(f"JSONL artifact {p.name} does not exist or is empty")
    count = 0
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                json.loads(line)
                count += 1
    return count


@dataclass(frozen=True)
class JobResult:
    video_id: str
    asset_hash: str
    job_key: str
    status: str  # "committed", "cached", "failed", "blocked_rights"
    attempts: int
    error_message: str | None = None
    frames_count: int = 0
    samples_count: int = 0
    ocr_probes_count: int = 0
    audio_entries_count: int = 0
    timeline_path: str | None = None
    samples_path: str | None = None
    audio_map_path: str | None = None
    ocr_probes_path: str | None = None
    elapsed_seconds: float = 0.0
    config_hash: str | None = None


@dataclass(frozen=True)
class RunReport:
    run_id: str
    started_at: str
    completed_at: str
    total_assets: int
    committed_assets: int
    cached_assets: int
    failed_assets: int
    blocked_assets: int
    total_frames_decoded: int
    total_samples_selected: int
    total_ocr_probes: int
    results: tuple[JobResult, ...]


class IngestionJobManager:
    """Orchestrates resilient, resumable ingestion jobs with atomic publishing."""

    def __init__(
        self,
        config: IngestionConfig,
        output_dir: Path | str,
        run_id: str | None = None,
        resume: bool = True,
    ):
        config.validate()
        self.config = config
        self.config_hash = compute_config_hash(config)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.run_id = run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    def execute_asset(
        self,
        media_record: dict[str, Any],
        resolver: MediaResolver | None = None,
        for_inference: bool = True,
    ) -> JobResult:
        """Run complete ingestion, timeline, and sampling stages for one media asset."""
        start_time = time.monotonic()
        video_id = media_record["video_id"]
        asset_hash = media_record["sha256"]
        timeline_rev = media_record.get("timeline_revision") or "rev_m1_1"
        sampler_rev = "videntra_hybrid_v1"

        job_key = compute_job_key(asset_hash, "ingest_m1", self.config_hash)

        # 1. Resolve local path and enforce access / hash policy (R08)
        try:
            if resolver is not None:
                media_path = resolver.resolve(video_id, for_inference=for_inference)
            elif "local_path" in media_record or "path" in media_record:
                media_path = Path(media_record.get("local_path") or media_record["path"]).resolve()
                if not media_path.is_file():
                    return JobResult(
                        video_id=video_id,
                        asset_hash=asset_hash,
                        job_key=job_key,
                        status="failed",
                        attempts=1,
                        error_message=f"Media file not found: {media_path}",
                        elapsed_seconds=time.monotonic() - start_time,
                        config_hash=self.config_hash,
                    )
                # Verify media bytes match recorded SHA-256
                actual_media_sha = sha256_file(media_path)
                if actual_media_sha != asset_hash:
                    return JobResult(
                        video_id=video_id,
                        asset_hash=asset_hash,
                        job_key=job_key,
                        status="failed",
                        attempts=1,
                        error_message=f"Media hash mismatch: expected {asset_hash}, got {actual_media_sha}",
                        elapsed_seconds=time.monotonic() - start_time,
                        config_hash=self.config_hash,
                    )
            else:
                return JobResult(
                    video_id=video_id,
                    asset_hash=asset_hash,
                    job_key=job_key,
                    status="failed",
                    attempts=1,
                    error_message="Cannot resolve media path (no resolver or local path)",
                    elapsed_seconds=time.monotonic() - start_time,
                    config_hash=self.config_hash,
                )
        except PermissionError as exc:
            return JobResult(
                video_id=video_id,
                asset_hash=asset_hash,
                job_key=job_key,
                status="blocked_rights",
                attempts=1,
                error_message=f"Access denied by rights gate: {exc}",
                elapsed_seconds=time.monotonic() - start_time,
                config_hash=self.config_hash,
            )
        except Exception as exc:
            return JobResult(
                video_id=video_id,
                asset_hash=asset_hash,
                job_key=job_key,
                status="failed",
                attempts=1,
                error_message=str(exc),
                elapsed_seconds=time.monotonic() - start_time,
                config_hash=self.config_hash,
            )

        # Committed target paths
        safe_vid = safe_path_id(video_id)
        timeline_dir = self.output_dir / "timeline" / safe_vid / timeline_rev
        samples_dir = self.output_dir / "samples" / safe_vid / sampler_rev
        target_frames = timeline_dir / "frames.parquet"
        target_audio = timeline_dir / "audio_map.jsonl"
        target_timeline_commit = timeline_dir / "commit.json"

        target_samples = samples_dir / "samples.parquet"
        target_probes = samples_dir / "ocr_probes.parquet"
        target_samples_commit = samples_dir / "commit.json"

        # 2. Check resume cache with recipe identity & artifact verification (R01, R02)
        if self.resume:
            if target_timeline_commit.exists() and target_samples_commit.exists():
                try:
                    with open(target_timeline_commit, "r", encoding="utf-8") as f:
                        tl_commit = json.load(f)
                    with open(target_samples_commit, "r", encoding="utf-8") as f:
                        samp_commit = json.load(f)

                    # Validate commit metadata against current request
                    if (
                        tl_commit.get("asset_hash") == asset_hash
                        and tl_commit.get("config_hash") == self.config_hash
                        and tl_commit.get("job_key") == job_key
                        and samp_commit.get("config_hash") == self.config_hash
                        and samp_commit.get("job_key") == job_key
                    ):
                        # Verify all artifacts physically exist, have rows, and are uncorrupted
                        frames_rows = verify_artifact_parquet(target_frames, expected_min_rows=1)
                        samples_rows = verify_artifact_parquet(target_samples, expected_min_rows=1)
                        probes_rows = verify_artifact_parquet(target_probes, expected_min_rows=1)
                        audio_rows = verify_artifact_jsonl(target_audio) if target_audio.exists() else 0

                        return JobResult(
                            video_id=video_id,
                            asset_hash=asset_hash,
                            job_key=job_key,
                            status="cached",
                            attempts=0,
                            frames_count=frames_rows,
                            samples_count=samples_rows,
                            ocr_probes_count=probes_rows,
                            audio_entries_count=audio_rows,
                            timeline_path=str(target_frames),
                            samples_path=str(target_samples),
                            audio_map_path=str(target_audio) if target_audio.exists() else None,
                            ocr_probes_path=str(target_probes),
                            elapsed_seconds=time.monotonic() - start_time,
                            config_hash=self.config_hash,
                        )
                except Exception:
                    # Incomplete or corrupted cache, proceed to re-ingest
                    pass

        # 3. Execution attempts
        max_retries = self.config.jobs.max_retries
        attempts_dir = self.output_dir / "_attempts" / f"{job_key}_{uuid.uuid4().hex[:8]}"
        attempts_dir.mkdir(parents=True, exist_ok=True)

        last_error = None
        for attempt in range(1, max_retries + 2):
            try:
                from videntra.timeline.ledger import write_timeline_ledger
                from videntra.timeline.maps import PiecewiseSourceMap, build_audio_map
                from videntra.sampling.hybrid import HybridAdaptiveSampler
                from videntra.sampling.probes import select_ocr_probes

                # Probe media
                probe_res = probe_media(
                    media_path,
                    stream_selection=self.config.stream_selection,
                )

                # Build piecewise source map
                source_map = None
                if media_record.get("source_mapping_kind") == "constant_offset":
                    source_map = PiecewiseSourceMap(
                        source_video_id=media_record.get("source_video_id"),
                        mapping_kind="constant_offset",
                        source_offset_us=media_record.get("source_offset_us"),
                    )

                # Initialize hybrid adaptive sampler
                sampler = HybridAdaptiveSampler(
                    config=self.config.sampling,
                    video_id=video_id,
                    timeline_revision=timeline_rev,
                    sampler_revision=sampler_rev,
                )

                # Streaming decode with bounded memory (R04)
                # Feed sampler and write ledger row groups without holding all RGB pixels in memory
                attempt_frames = attempts_dir / "frames.parquet"
                metadata_frames: list[DecodedFrame] = []
                first_origin = Fraction(0)

                def _frame_stream():
                    nonlocal first_origin
                    for f in decode_frames(media_path, probe_res, recipe=self.config.decode, include_pixels=True):
                        if f.ordinal == 0:
                            first_origin = f.origin
                        # Hybrid sampler downscales/processes and keeps pre-burst ring buffer
                        sampler.process_frame(f)

                        # Retain lightweight metadata without pixels for timeline ledger & probes
                        f_meta = DecodedFrame(
                            ordinal=f.ordinal,
                            pts=f.pts,
                            dts=f.dts,
                            time_base_num=f.time_base_num,
                            time_base_den=f.time_base_den,
                            origin=f.origin,
                            pts_occurrence=f.pts_occurrence,
                            epoch_id=f.epoch_id,
                            is_keyframe=f.is_keyframe,
                            pict_type=f.pict_type,
                            width=f.width,
                            height=f.height,
                            pixels=None,  # Free RGB memory!
                            best_effort_timestamp=f.best_effort_timestamp,
                            timestamp_quality=f.timestamp_quality,
                        )
                        metadata_frames.append(f_meta)
                        # Yield frame to ledger writer (it doesn't need pixels)
                        yield f_meta
                        # Release pixel buffer from original decoded frame
                        f.pixels = None

                # Write timeline ledger in streaming fashion
                summary = write_timeline_ledger(
                    attempt_frames,
                    _frame_stream(),
                    video_id=video_id,
                    timeline_revision=timeline_rev,
                    row_group_size=self.config.timeline.row_group_size,
                    source_map=source_map,
                )

                # Finalize hybrid sampling
                sample_manifest = sampler.finalize()
                attempt_samples = attempts_dir / "samples.parquet"
                sample_manifest.to_parquet(attempt_samples)

                # Audio map
                attempt_audio = attempts_dir / "audio_map.jsonl"
                audio_map = build_audio_map(media_path, probe_res, first_origin)
                audio_map.to_jsonl(attempt_audio)

                # Independent OCR probes using shot cuts from hybrid sampling (R06)
                attempt_probes = attempts_dir / "ocr_probes.parquet"
                ocr_manifest = select_ocr_probes(
                    metadata_frames,
                    video_id=video_id,
                    probe_fps=self.config.sampling.ocr_probe_fps,
                    shot_cut_ordinals=sample_manifest.shot_cut_ordinals,
                )
                ocr_manifest.to_parquet(attempt_probes)

                # Verify all attempt files before publication (R02)
                frames_rows = verify_artifact_parquet(attempt_frames, expected_min_rows=summary.total_frames)
                samples_rows = verify_artifact_parquet(attempt_samples, expected_min_rows=1)
                probes_rows = verify_artifact_parquet(attempt_probes, expected_min_rows=1)
                audio_rows = verify_artifact_jsonl(attempt_audio) if probe_res.has_audio else 0

                # 4. Atomic publish with verified commit markers (R05)
                timeline_dir.mkdir(parents=True, exist_ok=True)
                samples_dir.mkdir(parents=True, exist_ok=True)

                # Move artifacts
                shutil.move(str(attempt_frames), str(target_frames))
                shutil.move(str(attempt_audio), str(target_audio))
                shutil.move(str(attempt_samples), str(target_samples))
                shutil.move(str(attempt_probes), str(target_probes))

                # Write commit metadata markers LAST as atomic completion pointers
                now_utc = datetime.now(timezone.utc).isoformat()
                commit_meta = {
                    "schema_version": "1.0",
                    "video_id": video_id,
                    "asset_hash": asset_hash,
                    "job_key": job_key,
                    "config_hash": self.config_hash,
                    "recipe_id": self.config.recipe_id,
                    "timeline_revision": timeline_rev,
                    "sampler_revision": sampler_rev,
                    "committed_at_utc": now_utc,
                    "frames_count": frames_rows,
                    "samples_count": samples_rows,
                    "ocr_probes_count": probes_rows,
                    "audio_entries_count": audio_rows,
                    "artifacts": {
                        "frames": {"sha256": sha256_file(target_frames), "rows": frames_rows},
                        "audio_map": {"sha256": sha256_file(target_audio), "rows": audio_rows},
                        "samples": {"sha256": sha256_file(target_samples), "rows": samples_rows},
                        "ocr_probes": {"sha256": sha256_file(target_probes), "rows": probes_rows},
                    },
                }

                # Atomic commit marker write
                staging_commit_tl = attempts_dir / "commit_tl.json"
                staging_commit_samp = attempts_dir / "commit_samp.json"
                with open(staging_commit_tl, "w", encoding="utf-8") as f:
                    json.dump(commit_meta, f, indent=2, ensure_ascii=False)
                with open(staging_commit_samp, "w", encoding="utf-8") as f:
                    json.dump(commit_meta, f, indent=2, ensure_ascii=False)

                shutil.move(str(staging_commit_tl), str(target_timeline_commit))
                shutil.move(str(staging_commit_samp), str(target_samples_commit))

                # Cleanup attempt dir
                shutil.rmtree(attempts_dir, ignore_errors=True)

                return JobResult(
                    video_id=video_id,
                    asset_hash=asset_hash,
                    job_key=job_key,
                    status="committed",
                    attempts=attempt,
                    frames_count=frames_rows,
                    samples_count=samples_rows,
                    ocr_probes_count=probes_rows,
                    audio_entries_count=audio_rows,
                    timeline_path=str(target_frames),
                    samples_path=str(target_samples),
                    audio_map_path=str(target_audio),
                    ocr_probes_path=str(target_probes),
                    elapsed_seconds=time.monotonic() - start_time,
                    config_hash=self.config_hash,
                )
            except Exception as exc:
                last_error = exc
                time.sleep(0.1)

        # All retries failed
        shutil.rmtree(attempts_dir, ignore_errors=True)
        return JobResult(
            video_id=video_id,
            asset_hash=asset_hash,
            job_key=job_key,
            status="failed",
            attempts=max_retries + 1,
            error_message=f"Failed after {max_retries + 1} attempts: {last_error}",
            elapsed_seconds=time.monotonic() - start_time,
            config_hash=self.config_hash,
        )

    def execute_manifest(
        self,
        manifest_records: list[dict[str, Any]],
        resolver: MediaResolver | None = None,
        for_inference: bool = True,
    ) -> RunReport:
        """Run batch ingestion on a list of media asset records."""
        start_iso = datetime.now(timezone.utc).isoformat()
        results = []

        total_frames = 0
        total_samples = 0
        total_probes = 0
        committed = 0
        cached = 0
        failed = 0
        blocked = 0

        for r in manifest_records:
            res = self.execute_asset(r, resolver=resolver, for_inference=for_inference)
            results.append(res)
            if res.status == "committed":
                committed += 1
                total_frames += res.frames_count
                total_samples += res.samples_count
                total_probes += res.ocr_probes_count
            elif res.status == "cached":
                cached += 1
                total_frames += res.frames_count
                total_samples += res.samples_count
                total_probes += res.ocr_probes_count
            elif res.status == "blocked_rights":
                blocked += 1
            else:
                failed += 1

        end_iso = datetime.now(timezone.utc).isoformat()

        report = RunReport(
            run_id=self.run_id,
            started_at=start_iso,
            completed_at=end_iso,
            total_assets=len(manifest_records),
            committed_assets=committed,
            cached_assets=cached,
            failed_assets=failed,
            blocked_assets=blocked,
            total_frames_decoded=total_frames,
            total_samples_selected=total_samples,
            total_ocr_probes=total_probes,
            results=tuple(results),
        )

        # Write immutable run report in reports directory (R09)
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / f"{self.run_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)

        # Write latest run pointers
        run_manifest_path = self.output_dir / "run_manifest.json"
        with open(run_manifest_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)

        metrics = {
            "run_id": self.run_id,
            "total_assets": len(manifest_records),
            "committed_assets": committed,
            "cached_assets": cached,
            "failed_assets": failed,
            "blocked_assets": blocked,
            "total_frames_decoded": total_frames,
            "total_samples_selected": total_samples,
            "total_ocr_probes": total_probes,
            "config_hash": self.config_hash,
            "report_path": str(report_path),
        }
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        return report
