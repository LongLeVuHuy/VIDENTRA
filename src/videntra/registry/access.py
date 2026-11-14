"""Allowlisted media access. Application boundary, not an OS sandbox."""
from pathlib import Path
from urllib.parse import urlsplit, unquote

from videntra.contracts import validate
from .store import sha256_file


class MediaResolver:
    def __init__(self, roots: dict[str, Path], records: list[dict], denied_roots: tuple[Path, ...] = ()):
        self.roots = {key: value.resolve() for key, value in roots.items()}
        self.denied_roots = tuple(path.resolve() for path in denied_roots)
        self.records = {}
        for record in records:
            validate("media_asset", record)
            if record["video_id"] in self.records:
                raise ValueError("Duplicate video ID")
            self.records[record["video_id"]] = record.copy()

    def resolve(self, video_id: str, *, for_inference: bool = True) -> Path:
        record = self.records[video_id]
        if for_inference and record["rights_status"] != "allowed":
            raise PermissionError("Media rights are not approved for inference")
        uri = urlsplit(record["media_uri"])
        if uri.scheme != "asset" or uri.netloc not in self.roots or uri.query or uri.fragment:
            raise PermissionError("Only registered asset URIs are supported")
        relative = unquote(uri.path).lstrip("/")
        if "\\" in relative or ":" in relative or ".." in Path(relative).parts:
            raise PermissionError("Unsafe media locator")
        root = self.roots[uri.netloc]
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or any(path.is_relative_to(p) for p in self.denied_roots):
            raise PermissionError("Media path escapes allowed root")
        if path.suffix.lower() not in {".mp4", ".mkv", ".mov", ".webm", ".wav", ".flac"}:
            raise PermissionError("Non-media asset denied")
        if sha256_file(path) != record["sha256"]:
            raise ValueError("Media content hash mismatch")
        return path

