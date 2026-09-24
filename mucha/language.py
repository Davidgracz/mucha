from __future__ import annotations

import math
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path

MENTION_RE = re.compile(r"<@!?\d+>|<@&\d+>|<#\d+>")
URL_RE = re.compile(r"https?://\S+", re.I)

START_A = "\u0002"
START_B = "\u0003"


class OnlineLanguage:
    """Zero-pretraining character-level online language learner.

    The generator never selects whole learned words from a vocabulary.
    It learns transitions between individual characters and builds output
    character by character. Legacy word-level tables may remain in the same
    SQLite file, but they are ignored by this model.
    """

    def __init__(
        self,
        db_path: str | Path,
        min_chars: int,
        min_unique_chars: int,
        max_chars: int,
        seed: int = 67,
    ):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.rng = random.Random(seed)
        self.min_chars = max(1, int(min_chars))
        self.min_unique_chars = max(1, int(min_unique_chars))
        self.max_chars = max(24, int(max_chars))
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS char_unigram(
                ch TEXT PRIMARY KEY,
                n INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS char_bigram(
                a TEXT NOT NULL,
                b TEXT NOT NULL,
                n INTEGER NOT NULL,
                PRIMARY KEY(a,b)
            );

            CREATE TABLE IF NOT EXISTS char_trigram(
                a TEXT NOT NULL,
                b TEXT NOT NULL,
                c TEXT NOT NULL,
                n INTEGER NOT NULL,
                reward REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(a,b,c)
            );

            CREATE TABLE IF NOT EXISTS char_starts(
                a TEXT NOT NULL,
                b TEXT NOT NULL,
                n INTEGER NOT NULL,
                PRIMARY KEY(a,b)
            );

            CREATE TABLE IF NOT EXISTS char_stats(
                k TEXT PRIMARY KEY,
                v INTEGER NOT NULL
            );
            """
        )
        self.db.commit()

    @staticmethod
    def normalize(text: str) -> str:
        text = MENTION_RE.sub(" @user ", text)
        text = URL_RE.sub(" ", text)
        text = text.replace("@everyone", "everyone").replace("@here", "here")
        text = text.replace("\r", " ").replace("\n", " ")
        text = " ".join(text.split())
        return text[:1800]

    @classmethod
    def characters(cls, text: str) -> list[str]:
        text = cls.normalize(text).lower()
        return list(text)

    def learn(self, text: str) -> int:
        chars = self.characters(text)
        if not chars:
            return 0

        chars = chars[:1800]
        seq = [START_A, START_B] + chars
        cur = self.db.cursor()

        for ch in chars:
            cur.execute(
                "INSERT INTO char_unigram(ch,n) VALUES(?,1) "
                "ON CONFLICT(ch) DO UPDATE SET n=n+1",
                (ch,),
            )

        for a, b in zip(seq, seq[1:]):
            cur.execute(
                "INSERT INTO char_bigram(a,b,n) VALUES(?,?,1) "
                "ON CONFLICT(a,b) DO UPDATE SET n=n+1",
                (a, b),
            )

        for a, b, c in zip(seq, seq[1:], seq[2:]):
            cur.execute(
                "INSERT INTO char_trigram(a,b,c,n,reward) VALUES(?,?,?,1,0) "
                "ON CONFLICT(a,b,c) DO UPDATE SET n=n+1",
                (a, b, c),
            )

        if chars:
            first = chars[0]
            second = chars[1] if len(chars) > 1 else " "
            cur.execute(
                "INSERT INTO char_starts(a,b,n) VALUES(?,?,1) "
                "ON CONFLICT(a,b) DO UPDATE SET n=n+1",
                (first, second),
            )

        cur.execute(
            "INSERT INTO char_stats(k,v) VALUES('chars',?) "
            "ON CONFLICT(k) DO UPDATE SET v=v+excluded.v",
            (len(chars),),
        )
        cur.execute(
            "INSERT INTO char_stats(k,v) VALUES('messages',1) "
            "ON CONFLICT(k) DO UPDATE SET v=v+1"
        )
        self.db.commit()
        return len(chars)

    def stats(self) -> tuple[int, int]:
        row = self.db.execute(
            "SELECT v FROM char_stats WHERE k='chars'"
        ).fetchone()
        total = int(row[0]) if row else 0
        unique = int(
            self.db.execute("SELECT COUNT(*) FROM char_unigram").fetchone()[0]
        )
        return total, unique

    def diagnostics(self) -> dict:
        total, unique = self.stats()
        row = self.db.execute(
            "SELECT v FROM char_stats WHERE k='messages'"
        ).fetchone()
        messages = int(row[0]) if row else 0
        transitions = int(
            self.db.execute("SELECT COUNT(*) FROM char_trigram").fetchone()[0]
        )
        return {
            "mode": "characters",
            "chars": total,
            "unique_chars": unique,
            "messages": messages,
            "transitions": transitions,
            "ready": self.ready(),
        }

    def ready(self) -> bool:
        total, unique = self.stats()
        return total >= self.min_chars and unique >= self.min_unique_chars

    def _weighted_choice(self, rows: list[tuple[str, float]]) -> str | None:
        if not rows:
            return None
        vals = [(token, max(0.0001, float(weight))) for token, weight in rows]
        total = sum(w for _, w in vals)
        r = self.rng.random() * total
        upto = 0.0
        for token, weight in vals:
            upto += weight
            if upto >= r:
                return token
        return vals[-1][0]

    def _pick_start(self, context: str) -> tuple[str, str] | None:
        ctx = self.characters(context)

        if len(ctx) >= 2:
            a, b = ctx[-2], ctx[-1]
            exists = self.db.execute(
                "SELECT 1 FROM char_trigram WHERE a=? AND b=? LIMIT 1",
                (a, b),
            ).fetchone()
            if exists:
                return a, b

        if ctx:
            b = ctx[-1]
            rows = self.db.execute(
                "SELECT a,b,n FROM char_starts "
                "WHERE a=? OR b=? ORDER BY n DESC LIMIT 120",
                (b, b),
            ).fetchall()
            if rows:
                packed = self._weighted_choice(
                    [(a + "\u0000" + bb, float(n) ** 0.72) for a, bb, n in rows]
                )
                if packed:
                    return tuple(packed.split("\u0000", 1))  # type: ignore[return-value]

        rows = self.db.execute(
            "SELECT a,b,n FROM char_starts ORDER BY n DESC LIMIT 250"
        ).fetchall()
        if not rows:
            return None

        packed = self._weighted_choice(
            [(a + "\u0000" + b, float(n) ** 0.72) for a, b, n in rows]
        )
        if packed is None:
            return None
        a, b = packed.split("\u0000", 1)
        return a, b

    def _next_char(
        self,
        a: str,
        b: str,
        out: list[str],
        arousal: float,
    ) -> str | None:
        rows = self.db.execute(
            "SELECT c,n,reward FROM char_trigram "
            "WHERE a=? AND b=? ORDER BY n DESC LIMIT 180",
            (a, b),
        ).fetchall()

        if rows:
            # Higher arousal flattens the distribution, increasing invention.
            exponent = max(0.38, 0.78 - 0.30 * float(arousal))
            weighted: list[tuple[str, float]] = []
            for c, n, reward in rows:
                repeat_penalty = 1.0
                if len(out) >= 3 and c == out[-1] == out[-2] == out[-3]:
                    repeat_penalty = 0.03
                elif len(out) >= 2 and c == out[-1] == out[-2]:
                    repeat_penalty = 0.18
                score = (
                    (float(n) ** exponent)
                    * math.exp(max(-2.0, min(2.0, float(reward))))
                    * repeat_penalty
                )
                weighted.append((c, score))

            if weighted:
                return self._weighted_choice(weighted)

        rows2 = self.db.execute(
            "SELECT b,n FROM char_bigram WHERE a=? ORDER BY n DESC LIMIT 160",
            (b,),
        ).fetchall()
        if rows2:
            exponent = max(0.35, 0.68 - 0.25 * float(arousal))
            return self._weighted_choice(
                [(ch, float(n) ** exponent) for ch, n in rows2]
            )

        rows3 = self.db.execute(
            "SELECT ch,n FROM char_unigram ORDER BY n DESC LIMIT 220"
        ).fetchall()
        if rows3:
            return self._weighted_choice(
                [(ch, float(n) ** 0.50) for ch, n in rows3]
            )
        return None

    def generate(
        self,
        context: str = "",
        arousal: float = 0.5,
    ) -> tuple[str | None, list[tuple[str, str, str]]]:
        if not self.ready():
            return None, []

        start = self._pick_start(context)
        if start is None:
            return None, []

        a, b = start
        out = [a, b]
        used_trigrams: list[tuple[str, str, str]] = []

        target = max(
            24,
            min(
                self.max_chars,
                int(45 + float(arousal) * max(0, self.max_chars - 45)),
            ),
        )

        for _ in range(max(0, target - 2)):
            c = self._next_char(a, b, out, arousal)
            if c is None:
                break

            used_trigrams.append((a, b, c))
            out.append(c)
            a, b = b, c

            text_so_far = "".join(out)
            if (
                c in ".!?"
                and len(text_so_far.strip()) >= 20
                and self.rng.random() < 0.72
            ):
                break

        text = "".join(out)
        text = re.sub(r"\s+", " ", text).strip()
        text = text.replace("@everyone", "＠everyone").replace("@here", "＠here")
        text = re.sub(r"<@!?\d+>", "@user", text)

        if len(text) < 2:
            return None, used_trigrams

        # Avoid sending a dangling half-word too often. This does not consult
        # any dictionary; it only trims to a learned word boundary.
        if len(text) >= self.max_chars:
            last_space = text.rfind(" ")
            if last_space > len(text) * 0.55:
                text = text[:last_space]

        return text[:700], used_trigrams

    def reinforce(
        self,
        trigrams: list[tuple[str, str, str]],
        amount: float,
    ) -> None:
        if not trigrams:
            return

        amount = max(-1.0, min(1.0, float(amount)))
        cur = self.db.cursor()
        counts = Counter(trigrams)

        for (a, b, c), mult in counts.items():
            cur.execute(
                "UPDATE char_trigram "
                "SET reward=MAX(-2.0, MIN(2.0, reward + ?)) "
                "WHERE a=? AND b=? AND c=?",
                (0.06 * amount * min(mult, 4), a, b, c),
            )

        self.db.commit()

    def close(self) -> None:
        self.db.close()
