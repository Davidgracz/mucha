from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sqlite3
import subprocess
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_FILES = {
    "brain_state.npz": PROJECT_ROOT / "state" / "brain_state.npz",
    "language.sqlite3": PROJECT_ROOT / "state" / "language.sqlite3",
    "voice_episodes.sqlite3": PROJECT_ROOT / "state" / "voice_episodes.sqlite3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sqlite_backup(source: Path, destination: Path) -> None:
    source_uri = f"file:{source.as_posix()}?mode=ro"
    source_db = sqlite3.connect(source_uri, uri=True, timeout=15.0)
    destination_db = sqlite3.connect(destination, timeout=15.0)
    try:
        source_db.execute("PRAGMA busy_timeout=15000")
        destination_db.execute("PRAGMA busy_timeout=15000")
        source_db.backup(destination_db)
        destination_db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        destination_db.commit()
    finally:
        destination_db.close()
        source_db.close()


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def build_backup(output: Path) -> dict:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mucha-backup-") as tmp_name:
        tmp = Path(tmp_name)
        staged: dict[str, Path] = {}

        brain = STATE_FILES["brain_state.npz"]
        if brain.is_file():
            staged_brain = tmp / "brain_state.npz"
            shutil.copy2(brain, staged_brain)
            staged["brain_state.npz"] = staged_brain

        for name in ("language.sqlite3", "voice_episodes.sqlite3"):
            source = STATE_FILES[name]
            if source.is_file():
                destination = tmp / name
                sqlite_backup(source, destination)
                staged[name] = destination

        if not staged:
            raise RuntimeError(
                "Nie znaleziono żadnych plików stanu w katalogu state/. "
                "Uruchom Muchę przynajmniej raz przed wykonaniem backupu."
            )

        manifest = {
            "format": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "project_root": str(PROJECT_ROOT),
            "git_commit": git_commit(),
            "files": {},
        }
        for name, path in staged.items():
            manifest["files"][name] = {
                "size": path.stat().st_size,
                "sha256": sha256(path),
            }

        manifest_path = tmp / "backup_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with zipfile.ZipFile(
            output,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            for name, path in staged.items():
                archive.write(path, arcname=f"state/{name}")
            archive.write(manifest_path, arcname="backup_manifest.json")

    manifest["archive"] = str(output)
    manifest["archive_size"] = output.stat().st_size
    manifest["archive_sha256"] = sha256(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tworzy spójny backup wyuczonego stanu Muchy."
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Docelowy plik ZIP.",
    )
    args = parser.parse_args()

    manifest = build_backup(args.output)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
