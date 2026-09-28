#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.toml"
LOCAL_CONFIG_PATH = ROOT / "config.local.toml"


def _load_database_path() -> Path:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Brak {CONFIG_PATH}")
    with CONFIG_PATH.open("rb") as handle:
        base = tomllib.load(handle)
    database = base.get("language", {}).get("database", "state/language.sqlite3")
    if LOCAL_CONFIG_PATH.exists():
        with LOCAL_CONFIG_PATH.open("rb") as handle:
            local = tomllib.load(handle)
        database = local.get("language", {}).get("database", database)
    path = Path(str(database))
    if not path.is_absolute():
        path = ROOT / path
    return path


def _connect() -> sqlite3.Connection:
    path = _load_database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=5.0)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=5000")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS social_user_affinity(
            user_id INTEGER PRIMARY KEY,
            display_name TEXT NOT NULL DEFAULT '',
            affinity REAL NOT NULL DEFAULT 0,
            positive_reactions INTEGER NOT NULL DEFAULT 0,
            negative_reactions INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL DEFAULT 0
        )
        """
    )
    db.commit()
    return db


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _row(db: sqlite3.Connection, user_id: int):
    return db.execute(
        """
        SELECT user_id, display_name, affinity,
               positive_reactions, negative_reactions, updated_at
        FROM social_user_affinity
        WHERE user_id=?
        """,
        (int(user_id),),
    ).fetchone()


def _format_row(row) -> str:
    if row is None:
        return "brak wpisu"
    user_id, name, affinity, pos, neg, updated = row
    name = name or str(user_id)
    return (
        f"{user_id}  {name}  affinity={float(affinity):+.3f}  "
        f"reakcje +{int(pos)}/-{int(neg)}"
    )


def list_rows(db: sqlite3.Connection, limit: int = 100, search: str = "") -> None:
    limit = max(1, min(500, int(limit)))
    if search:
        needle = f"%{search.lower()}%"
        rows = db.execute(
            """
            SELECT user_id, display_name, affinity,
                   positive_reactions, negative_reactions, updated_at
            FROM social_user_affinity
            WHERE lower(display_name) LIKE ? OR CAST(user_id AS TEXT) LIKE ?
            ORDER BY affinity DESC, updated_at DESC
            LIMIT ?
            """,
            (needle, f"%{search}%", limit),
        ).fetchall()
    else:
        rows = db.execute(
            """
            SELECT user_id, display_name, affinity,
                   positive_reactions, negative_reactions, updated_at
            FROM social_user_affinity
            ORDER BY affinity DESC, updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    if not rows:
        print("Brak użytkowników affinity.")
        return

    print(f"{'USER ID':<20} {'NAZWA':<28} {'AFFINITY':>9} {'+':>5} {'-':>5}")
    print("-" * 72)
    for user_id, name, affinity, pos, neg, _ in rows:
        print(
            f"{str(user_id):<20} {str(name or user_id)[:28]:<28} "
            f"{float(affinity):>+9.3f} {int(pos):>5} {int(neg):>5}"
        )


def set_affinity(
    db: sqlite3.Connection,
    user_id: int,
    value: float,
    display_name: str | None = None,
) -> float:
    value = _clamp(value)
    existing = _row(db, user_id)
    name = (
        str(display_name).strip()[:120]
        if display_name
        else (str(existing[1]) if existing and existing[1] else str(user_id))
    )
    db.execute(
        """
        INSERT INTO social_user_affinity(
            user_id, display_name, affinity,
            positive_reactions, negative_reactions, updated_at
        ) VALUES(?,?,?,0,0,strftime('%s','now'))
        ON CONFLICT(user_id) DO UPDATE SET
            display_name=CASE
                WHEN excluded.display_name='' THEN social_user_affinity.display_name
                ELSE excluded.display_name
            END,
            affinity=excluded.affinity,
            updated_at=excluded.updated_at
        """,
        (int(user_id), name, value),
    )
    db.commit()
    return value


def add_affinity(
    db: sqlite3.Connection,
    user_id: int,
    delta: float,
    display_name: str | None = None,
) -> float:
    before = _row(db, user_id)
    current = float(before[2]) if before else 0.0
    return set_affinity(db, user_id, current + float(delta), display_name)


def interactive(db: sqlite3.Connection) -> None:
    print("MUCHA — affinity CLI")
    print("Edytujesz LEGACY affinity z language.sqlite3.")
    print("Przy aktywnej neural social memory effective affinity może być inne.")
    print()
    list_rows(db, 100)
    while True:
        print()
        raw = input("User ID (Enter = koniec, 'list' = lista): ").strip()
        if not raw:
            return
        if raw.lower() == "list":
            list_rows(db, 100)
            continue
        try:
            user_id = int(raw)
        except ValueError:
            print("Nieprawidłowe ID.")
            continue

        print("Aktualnie:", _format_row(_row(db, user_id)))
        mode = input("Operacja [set/add/reset]: ").strip().lower()
        if mode == "reset":
            value = set_affinity(db, user_id, 0.0)
            print(f"Ustawiono {user_id}: {value:+.3f}")
            continue
        if mode not in {"set", "add"}:
            print("Nieznana operacja.")
            continue
        try:
            value = float(input("Wartość (-1.0 .. 1.0): ").strip().replace(",", "."))
        except ValueError:
            print("Nieprawidłowa liczba.")
            continue
        if mode == "set":
            result = set_affinity(db, user_id, value)
        else:
            result = add_affinity(db, user_id, value)
        print("Po zmianie:", _format_row(_row(db, user_id)))
        print(f"legacy affinity = {result:+.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ręczna edycja legacy affinity Muchy."
    )
    sub = parser.add_subparsers(dest="command")

    p_list = sub.add_parser("list", help="Pokaż użytkowników")
    p_list.add_argument("--limit", type=int, default=100)
    p_list.add_argument("--search", default="")

    p_get = sub.add_parser("get", help="Pokaż użytkownika")
    p_get.add_argument("user_id", type=int)

    p_set = sub.add_parser("set", help="Ustaw affinity")
    p_set.add_argument("user_id", type=int)
    p_set.add_argument("value", type=float)
    p_set.add_argument("--name")

    p_add = sub.add_parser("add", help="Dodaj/odejmij affinity")
    p_add.add_argument("user_id", type=int)
    p_add.add_argument("delta", type=float)
    p_add.add_argument("--name")

    p_reset = sub.add_parser("reset", help="Ustaw affinity na 0")
    p_reset.add_argument("user_id", type=int)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    db = _connect()
    try:
        if args.command is None:
            interactive(db)
        elif args.command == "list":
            list_rows(db, args.limit, args.search)
        elif args.command == "get":
            print(_format_row(_row(db, args.user_id)))
        elif args.command == "set":
            set_affinity(db, args.user_id, args.value, args.name)
            print("Po zmianie:", _format_row(_row(db, args.user_id)))
        elif args.command == "add":
            add_affinity(db, args.user_id, args.delta, args.name)
            print("Po zmianie:", _format_row(_row(db, args.user_id)))
        elif args.command == "reset":
            set_affinity(db, args.user_id, 0.0)
            print("Po zmianie:", _format_row(_row(db, args.user_id)))
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
