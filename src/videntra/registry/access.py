"""Allowlisted media access. Application boundary, not an OS sandbox."""
from pathlib import Path
from urllib.parse import urlsplit, unquote

from videntra.contracts import validate
from .store import sha256_file


ALLOWED_MEDIA_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".wav", ".flac"}


class MediaResolver:
    def __init__(
        self,
        roots: dict[str, Path],
        records: list[dict],
        denied_roots: tuple[Path, ...] = (),
        *,
        allow_unregistered_roots: bool = False,
    ):
        self.roots = {key: value.resolve() for key, value in roots.items()}
        self.denied_roots = tuple(path.resolve() for path in denied_roots)
        self.allow_unregistered_roots = allow_unregistered_roots
        self.records = {}
        self._discovered: dict[str, Path] = {}
        for record in records:
            validate("media_asset", record)
            if record["video_id"] in self.records:
                raise ValueError("Duplicate video ID")
            self.records[record["video_id"]] = record.copy()
        if allow_unregistered_roots:
            self._scan_roots()

    def _is_denied(self, path: Path) -> bool:
        return any(path.is_relative_to(p) for p in self.denied_roots)

    def _scan_roots(self) -> None:
        for root in self.roots.values():
            if not root.exists() or not root.is_dir() or self._is_denied(root):
                continue
            for item in root.iterdir():
                if not item.is_file():
                    continue
                if item.suffix.lower() not in ALLOWED_MEDIA_EXTENSIONS:
                    continue
                resolved = item.resolve()
                if resolved.is_relative_to(root) and not self._is_denied(resolved):
                    self._discovered.setdefault(item.stem, resolved)

    def resolve(self, video_id: str, *, for_inference: bool = True) -> Path:
        if video_id not in self.records:
            if self.allow_unregistered_roots and video_id in self._discovered:
                path = self._discovered[video_id]
                if not any(path.is_relative_to(root) for root in self.roots.values()) or self._is_denied(path):
                    raise PermissionError("Media path escapes allowed root")
                if path.suffix.lower() not in ALLOWED_MEDIA_EXTENSIONS:
                    raise PermissionError("Non-media asset denied")
                if not path.is_file():
                    raise KeyError(video_id)
                return path
            raise KeyError(video_id)

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
        if not path.is_relative_to(root) or self._is_denied(path):
            raise PermissionError("Media path escapes allowed root")
        if path.suffix.lower() not in ALLOWED_MEDIA_EXTENSIONS:
            raise PermissionError("Non-media asset denied")
        if sha256_file(path) != record["sha256"]:
            raise ValueError("Media content hash mismatch")
        return path

