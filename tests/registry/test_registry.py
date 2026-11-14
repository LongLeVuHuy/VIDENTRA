import copy
import sqlite3
import pytest
from videntra.registry import Registry, RegistryConflict


def test_restart_idempotency_immutability_and_alias(tmp_path, media):
    database = tmp_path / "registry.sqlite"
    item = ("media_asset", media["video_id"], "r1", media)
    with Registry(database) as registry:
        assert registry.register_batch([item]) == 1
        assert registry.register_batch([item]) == 0
    with Registry(database) as registry:
        assert registry.get(*item[:3]) == media
        alias = dict(media, video_id="alias")
        registry.register_batch([("media_asset", "alias", "r1", alias)])
        assert registry.aliases(media["sha256"]) == sorted(["alias", media["video_id"]])
        modified = dict(media, exposure="unknown")
        with pytest.raises(RegistryConflict):
            registry.register_batch([(*item[:3], modified)])
        assert registry.register_batch([("media_asset", media["video_id"], "r2", modified)]) == 1
        with pytest.raises(sqlite3.IntegrityError):
            registry.db.execute("DELETE FROM records")
        with pytest.raises(sqlite3.IntegrityError):
            registry.db.execute("UPDATE media_identity SET sha256='changed'")


def test_failed_batch_rolls_back(tmp_path, media):
    item = ("media_asset", media["video_id"], "r1", media)
    with Registry(tmp_path / "registry.sqlite") as registry:
        registry.register_batch([item])
        new = dict(media, video_id="new")
        bad = dict(media, sha256="f"*64)
        with pytest.raises(RegistryConflict):
            registry.register_batch([("media_asset", "new", "r1", new),
                                     ("media_asset", media["video_id"], "r2", bad)])
        with pytest.raises(KeyError):
            registry.get("media_asset", "new", "r1")
        assert registry.aliases(media["sha256"]) == [media["video_id"]]


def test_registry_rejects_unrelated_kind(tmp_path, media):
    with Registry(tmp_path / "registry.sqlite") as registry:
        with pytest.raises(ValueError):
            registry.register_batch([("ground_truth", "x", "r1", media)])

