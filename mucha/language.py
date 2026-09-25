from __future__ import annotations

import math
import random
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Callable

MENTION_RE = re.compile(r"<@!?\d+>|<@&\d+>|<#\d+>")
URL_RE = re.compile(r"https?://\S+", re.I)
WORD_RE = re.compile(
    r"@user|[^\W_]+(?:['’][^\W_]+)?|[.!?,;:]",
    re.UNICODE,
)

START_A = "\u0002"
START_B = "\u0003"
WORD_START_A = "\u0002W"
WORD_START_B = "\u0003W"
WORD_END = "\u0004W"


class OnlineLanguage:
    """Zero-pretraining hybrid online language learner.

    Character transitions keep spelling flexible while an online word
    unigram/bigram/trigram model learns sentence structure much faster.
    Both layers are learned only from Discord text, STT transcripts and
    feedback stored in the local SQLite database.
    """

    def __init__(
        self,
        db_path: str | Path,
        min_chars: int,
        min_unique_chars: int,
        max_chars: int,
        seed: int = 67,
        hybrid_word_enabled: bool = True,
        word_model_probability: float = 0.90,
        word_max_tokens: int = 14,
        word_recent_window_seconds: int = 3600,
        word_recent_boost: float = 2.00,
        word_frequency_exponent: float = 0.95,
        word_arousal_flatten: float = 0.12,
        char_frequency_exponent: float = 0.90,
        char_arousal_flatten: float = 0.15,
        word_reward_scale: float = 0.12,
        connectome_word_control_enabled: bool = True,
        connectome_word_control_min_vocab: int = 1500,
        connectome_word_control_strength: float = 0.35,
        connectome_word_control_candidates: int = 24,
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
        self.hybrid_word_enabled = bool(hybrid_word_enabled)
        self.word_model_probability = max(
            0.0,
            min(1.0, float(word_model_probability)),
        )
        self.word_max_tokens = max(3, int(word_max_tokens))
        self.word_recent_window_seconds = max(
            1,
            int(word_recent_window_seconds),
        )
        self.word_recent_boost = max(1.0, float(word_recent_boost))
        self.word_frequency_exponent = max(
            0.1,
            float(word_frequency_exponent),
        )
        self.word_arousal_flatten = max(
            0.0,
            float(word_arousal_flatten),
        )
        self.char_frequency_exponent = max(
            0.1,
            float(char_frequency_exponent),
        )
        self.char_arousal_flatten = max(
            0.0,
            float(char_arousal_flatten),
        )
        self.word_reward_scale = max(
            0.0,
            min(1.0, float(word_reward_scale)),
        )
        self.connectome_word_control_enabled = bool(
            connectome_word_control_enabled
        )
        self.connectome_word_control_min_vocab = max(
            8,
            int(connectome_word_control_min_vocab),
        )
        self.connectome_word_control_strength = max(
            0.0,
            min(2.0, float(connectome_word_control_strength)),
        )
        self.connectome_word_control_candidates = max(
            4,
            min(96, int(connectome_word_control_candidates)),
        )
        self._brain_word_control_last: dict = {
            "active": False,
            "ready": False,
            "vocab": 0,
            "min_vocab": self.connectome_word_control_min_vocab,
            "evaluated": 0,
            "mean_score": 0.5,
        }
        self._last_generator = "none"
        self._init_schema()
        self._bootstrap_from_legacy_words()
        self._bootstrap_word_model_from_legacy()

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

            CREATE TABLE IF NOT EXISTS word_unigram(
                token TEXT PRIMARY KEY,
                n INTEGER NOT NULL,
                reward REAL NOT NULL DEFAULT 0,
                last_seen REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS word_bigram(
                a TEXT NOT NULL,
                b TEXT NOT NULL,
                n INTEGER NOT NULL,
                reward REAL NOT NULL DEFAULT 0,
                last_seen REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(a,b)
            );

            CREATE TABLE IF NOT EXISTS word_trigram(
                a TEXT NOT NULL,
                b TEXT NOT NULL,
                c TEXT NOT NULL,
                n INTEGER NOT NULL,
                reward REAL NOT NULL DEFAULT 0,
                last_seen REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(a,b,c)
            );

            CREATE TABLE IF NOT EXISTS word_starts(
                a TEXT NOT NULL,
                b TEXT NOT NULL,
                n INTEGER NOT NULL,
                last_seen REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(a,b)
            );

            CREATE TABLE IF NOT EXISTS social_user_affinity(
                user_id INTEGER PRIMARY KEY,
                display_name TEXT NOT NULL DEFAULT '',
                affinity REAL NOT NULL DEFAULT 0,
                positive_reactions INTEGER NOT NULL DEFAULT 0,
                negative_reactions INTEGER NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS social_word_feedback(
                word TEXT PRIMARY KEY,
                confirmations INTEGER NOT NULL DEFAULT 0,
                reward REAL NOT NULL DEFAULT 0,
                updated_at REAL NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS social_word_user(
                word TEXT NOT NULL,
                user_id INTEGER NOT NULL,
                PRIMARY KEY(word,user_id)
            );
            """
        )
        self.db.commit()

    def _table_exists(self, name: str) -> bool:
        row = self.db.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    def _bootstrap_from_legacy_words(self) -> None:
        """One-time conversion of the old word model into character transitions.

        Legacy words are never used directly during generation. Their observed
        frequencies only seed char-level statistics, so subsequent output is
        still assembled one character at a time.
        """
        done = self.db.execute(
            "SELECT v FROM char_stats WHERE k='legacy_bootstrap_v1'"
        ).fetchone()
        if done:
            return

        required = ("unigram", "bigram")
        if not all(self._table_exists(name) for name in required):
            self.db.execute(
                "INSERT INTO char_stats(k,v) VALUES('legacy_bootstrap_v1',0) "
                "ON CONFLICT(k) DO NOTHING"
            )
            self.db.commit()
            return

        char_counts: Counter[str] = Counter()
        bigram_counts: Counter[tuple[str, str]] = Counter()
        trigram_counts: Counter[tuple[str, str, str]] = Counter()
        start_counts: Counter[tuple[str, str]] = Counter()
        imported_chars = 0
        imported_items = 0

        def add_sequence(text: str, weight: int, as_start: bool = False) -> None:
            nonlocal imported_chars, imported_items
            normalized = self.normalize(text).lower()
            if not normalized:
                return
            chars = list(normalized[:180])
            if not chars:
                return
            weight = max(1, min(8, int(weight)))
            seq = [START_A, START_B] + chars if as_start else chars

            for ch in chars:
                char_counts[ch] += weight
            for a, b in zip(seq, seq[1:]):
                bigram_counts[(a, b)] += weight
            for a, b, cc in zip(seq, seq[1:], seq[2:]):
                trigram_counts[(a, b, cc)] += weight

            if as_start:
                first = chars[0]
                second = chars[1] if len(chars) > 1 else " "
                start_counts[(first, second)] += weight

            imported_chars += len(chars) * weight
            imported_items += 1

        # Old individual words teach spelling structure. Frequency is compressed
        # logarithmically so very common words do not completely dominate.
        for token, n in self.db.execute(
            "SELECT token,n FROM unigram ORDER BY n DESC LIMIT 6000"
        ).fetchall():
            if not token:
                continue
            weight = max(1, min(6, int(math.log2(max(1, int(n))) + 1)))
            add_sequence(str(token), weight, as_start=False)

        # Old word pairs are especially useful because they teach spaces and
        # transitions across word boundaries without making whole words atomic.
        for a, b, n in self.db.execute(
            "SELECT a,b,n FROM bigram ORDER BY n DESC LIMIT 12000"
        ).fetchall():
            if not a or not b:
                continue
            weight = max(1, min(4, int(math.log2(max(1, int(n))) + 1)))
            add_sequence(f"{a} {b}", weight, as_start=False)

        # Preserve some sentence-start statistics when the legacy table exists.
        if self._table_exists("starts"):
            for a, b, n in self.db.execute(
                "SELECT a,b,n FROM starts ORDER BY n DESC LIMIT 3000"
            ).fetchall():
                if not a:
                    continue
                text = f"{a} {b}" if b else str(a)
                weight = max(1, min(5, int(math.log2(max(1, int(n))) + 1)))
                add_sequence(text, weight, as_start=True)

        cur = self.db.cursor()
        for ch, n in char_counts.items():
            cur.execute(
                "INSERT INTO char_unigram(ch,n) VALUES(?,?) "
                "ON CONFLICT(ch) DO UPDATE SET n=n+excluded.n",
                (ch, int(n)),
            )
        for (a, b), n in bigram_counts.items():
            cur.execute(
                "INSERT INTO char_bigram(a,b,n) VALUES(?,?,?) "
                "ON CONFLICT(a,b) DO UPDATE SET n=n+excluded.n",
                (a, b, int(n)),
            )
        for (a, b, cc), n in trigram_counts.items():
            cur.execute(
                "INSERT INTO char_trigram(a,b,c,n,reward) VALUES(?,?,?,?,0) "
                "ON CONFLICT(a,b,c) DO UPDATE SET n=n+excluded.n",
                (a, b, cc, int(n)),
            )
        for (a, b), n in start_counts.items():
            cur.execute(
                "INSERT INTO char_starts(a,b,n) VALUES(?,?,?) "
                "ON CONFLICT(a,b) DO UPDATE SET n=n+excluded.n",
                (a, b, int(n)),
            )

        if imported_chars:
            cur.execute(
                "INSERT INTO char_stats(k,v) VALUES('chars',?) "
                "ON CONFLICT(k) DO UPDATE SET v=v+excluded.v",
                (int(imported_chars),),
            )
        cur.execute(
            "INSERT INTO char_stats(k,v) VALUES('legacy_bootstrap_chars',?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (int(imported_chars),),
        )
        cur.execute(
            "INSERT INTO char_stats(k,v) VALUES('legacy_bootstrap_items',?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (int(imported_items),),
        )
        cur.execute(
            "INSERT INTO char_stats(k,v) VALUES('legacy_bootstrap_v1',1) "
            "ON CONFLICT(k) DO UPDATE SET v=1"
        )
        self.db.commit()

    def _bootstrap_word_model_from_legacy(self) -> None:
        done = self.db.execute(
            "SELECT v FROM char_stats WHERE k='hybrid_word_bootstrap_v1'"
        ).fetchone()
        if done:
            return

        legacy_seen = 0.0
        cur = self.db.cursor()
        imported = 0

        if self._table_exists("unigram"):
            for token, n in self.db.execute(
                "SELECT token,n FROM unigram ORDER BY n DESC LIMIT 10000"
            ).fetchall():
                token = str(token or "").strip().lower()
                if not token:
                    continue
                count = max(1, min(50, int(n)))
                cur.execute(
                    "INSERT INTO word_unigram(token,n,reward,last_seen) "
                    "VALUES(?,?,0,?) "
                    "ON CONFLICT(token) DO UPDATE SET "
                    "n=word_unigram.n+excluded.n, "
                    "last_seen=MAX(word_unigram.last_seen,excluded.last_seen)",
                    (token, count, legacy_seen),
                )
                imported += 1

        if self._table_exists("bigram"):
            for a, b, n in self.db.execute(
                "SELECT a,b,n FROM bigram ORDER BY n DESC LIMIT 20000"
            ).fetchall():
                a = str(a or "").strip().lower()
                b = str(b or "").strip().lower()
                if not a or not b:
                    continue
                count = max(1, min(30, int(n)))
                cur.execute(
                    "INSERT INTO word_bigram(a,b,n,reward,last_seen) "
                    "VALUES(?,?,?,0,?) "
                    "ON CONFLICT(a,b) DO UPDATE SET "
                    "n=word_bigram.n+excluded.n, "
                    "last_seen=MAX(word_bigram.last_seen,excluded.last_seen)",
                    (a, b, count, legacy_seen),
                )
                imported += 1

        if self._table_exists("trigram"):
            try:
                rows = self.db.execute(
                    "SELECT a,b,c,n FROM trigram "
                    "ORDER BY n DESC LIMIT 30000"
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            for a, b, cc, n in rows:
                a = str(a or "").strip().lower()
                b = str(b or "").strip().lower()
                cc = str(cc or "").strip().lower()
                if not a or not b or not cc:
                    continue
                count = max(1, min(20, int(n)))
                cur.execute(
                    "INSERT INTO word_trigram(a,b,c,n,reward,last_seen) "
                    "VALUES(?,?,?,?,0,?) "
                    "ON CONFLICT(a,b,c) DO UPDATE SET "
                    "n=word_trigram.n+excluded.n, "
                    "last_seen=MAX(word_trigram.last_seen,excluded.last_seen)",
                    (a, b, cc, count, legacy_seen),
                )
                imported += 1

        if self._table_exists("starts"):
            try:
                rows = self.db.execute(
                    "SELECT a,b,n FROM starts ORDER BY n DESC LIMIT 5000"
                ).fetchall()
            except sqlite3.OperationalError:
                rows = []
            for a, b, n in rows:
                a = str(a or "").strip().lower()
                b = str(b or "").strip().lower()
                if not a:
                    continue
                if not b:
                    b = WORD_END
                count = max(1, min(20, int(n)))
                cur.execute(
                    "INSERT INTO word_starts(a,b,n,last_seen) "
                    "VALUES(?,?,?,?) "
                    "ON CONFLICT(a,b) DO UPDATE SET "
                    "n=word_starts.n+excluded.n, "
                    "last_seen=MAX(word_starts.last_seen,excluded.last_seen)",
                    (a, b, count, legacy_seen),
                )

        cur.execute(
            "INSERT INTO char_stats(k,v) "
            "VALUES('hybrid_word_bootstrap_v1',?) "
            "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (max(1, imported),),
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

    @classmethod
    def words(cls, text: str) -> list[str]:
        normalized = cls.normalize(text).lower()
        return WORD_RE.findall(normalized)[:120]

    def _learn_words(
        self,
        text: str,
        cur: sqlite3.Cursor,
        now: float,
    ) -> int:
        tokens = self.words(text)
        if not tokens:
            return 0

        for token in tokens:
            cur.execute(
                "INSERT INTO word_unigram(token,n,reward,last_seen) "
                "VALUES(?,1,0,?) "
                "ON CONFLICT(token) DO UPDATE SET "
                "n=word_unigram.n+1,last_seen=excluded.last_seen",
                (token, now),
            )

        transition_tokens = tokens + [WORD_END]
        for a, b in zip(transition_tokens, transition_tokens[1:]):
            cur.execute(
                "INSERT INTO word_bigram(a,b,n,reward,last_seen) "
                "VALUES(?,?,1,0,?) "
                "ON CONFLICT(a,b) DO UPDATE SET "
                "n=word_bigram.n+1,last_seen=excluded.last_seen",
                (a, b, now),
            )

        for a, b, cc in zip(
            transition_tokens,
            transition_tokens[1:],
            transition_tokens[2:],
        ):
            cur.execute(
                "INSERT INTO word_trigram(a,b,c,n,reward,last_seen) "
                "VALUES(?,?,?,1,0,?) "
                "ON CONFLICT(a,b,c) DO UPDATE SET "
                "n=word_trigram.n+1,last_seen=excluded.last_seen",
                (a, b, cc, now),
            )

        first = tokens[0]
        second = tokens[1] if len(tokens) > 1 else WORD_END
        cur.execute(
            "INSERT INTO word_starts(a,b,n,last_seen) VALUES(?,?,1,?) "
            "ON CONFLICT(a,b) DO UPDATE SET "
            "n=word_starts.n+1,last_seen=excluded.last_seen",
            (first, second, now),
        )
        return len(tokens)

    def learn(self, text: str) -> int:
        chars = self.characters(text)
        if not chars:
            return 0

        chars = chars[:1800]
        seq = [START_A, START_B] + chars
        cur = self.db.cursor()
        now = time.time()
        word_count = self._learn_words(text, cur, now)

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
        cur.execute(
            "INSERT INTO char_stats(k,v) VALUES('word_tokens',?) "
            "ON CONFLICT(k) DO UPDATE SET v=v+excluded.v",
            (int(word_count),),
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
        word_tokens_row = self.db.execute(
            "SELECT v FROM char_stats WHERE k='word_tokens'"
        ).fetchone()
        word_vocab = int(
            self.db.execute(
                "SELECT COUNT(*) FROM word_unigram"
            ).fetchone()[0]
        )
        word_bigrams = int(
            self.db.execute(
                "SELECT COUNT(*) FROM word_bigram"
            ).fetchone()[0]
        )
        word_trigrams = int(
            self.db.execute(
                "SELECT COUNT(*) FROM word_trigram"
            ).fetchone()[0]
        )
        legacy_chars_row = self.db.execute(
            "SELECT v FROM char_stats WHERE k='legacy_bootstrap_chars'"
        ).fetchone()
        legacy_items_row = self.db.execute(
            "SELECT v FROM char_stats WHERE k='legacy_bootstrap_items'"
        ).fetchone()
        return {
            "mode": (
                "hybrid-word+characters"
                if self.hybrid_word_enabled
                else "characters"
            ),
            "chars": total,
            "unique_chars": unique,
            "messages": messages,
            "transitions": transitions,
            "word_tokens": (
                int(word_tokens_row[0])
                if word_tokens_row
                else 0
            ),
            "word_vocab": word_vocab,
            "word_bigrams": word_bigrams,
            "word_trigrams": word_trigrams,
            "last_generator": self._last_generator,
            "word_model_probability": self.word_model_probability,
            "word_recent_boost": self.word_recent_boost,
            "connectome_word_control_enabled": (
                self.connectome_word_control_enabled
            ),
            "connectome_word_control_ready": bool(
                self.connectome_word_control_enabled
                and word_vocab >= self.connectome_word_control_min_vocab
            ),
            "connectome_word_control_min_vocab": (
                self.connectome_word_control_min_vocab
            ),
            "connectome_word_control_strength": (
                self.connectome_word_control_strength
            ),
            "connectome_word_control_candidates": (
                self.connectome_word_control_candidates
            ),
            "connectome_word_control_last": dict(
                self._brain_word_control_last
            ),
            "legacy_bootstrap_chars": int(legacy_chars_row[0]) if legacy_chars_row else 0,
            "legacy_bootstrap_items": int(legacy_items_row[0]) if legacy_items_row else 0,
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
            exponent = max(
                0.55,
                self.char_frequency_exponent
                - self.char_arousal_flatten * float(arousal),
            )
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
            exponent = max(
                0.50,
                (self.char_frequency_exponent - 0.10)
                - self.char_arousal_flatten * 0.8 * float(arousal),
            )
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

    def _recent_multiplier(self, last_seen: float) -> float:
        if last_seen <= 0.0:
            return 1.0
        age = max(0.0, time.time() - float(last_seen))
        window = max(1.0, float(self.word_recent_window_seconds))
        if age >= window:
            return 1.0
        freshness = 1.0 - age / window
        effective_boost = 1.0 + (self.word_recent_boost - 1.0) * 0.45
        return 1.0 + (effective_boost - 1.0) * freshness

    def _word_weight(
        self,
        n: int,
        reward: float,
        last_seen: float,
        arousal: float,
        repeat_penalty: float = 1.0,
    ) -> float:
        exponent = max(
            0.55,
            self.word_frequency_exponent
            - self.word_arousal_flatten * float(arousal),
        )
        return (
            (max(1.0, float(n)) ** exponent)
            * math.exp(max(-2.0, min(2.0, float(reward))))
            * self._recent_multiplier(float(last_seen))
            * max(0.01, float(repeat_penalty))
        )

    def _weighted_word_row(
        self,
        rows: list[tuple],
        arousal: float,
        token_index: int,
        out: list[str],
    ) -> str | None:
        weighted: list[tuple[str, float]] = []
        for row in rows:
            token = str(row[token_index])
            n = int(row[token_index + 1])
            reward = float(row[token_index + 2])
            last_seen = float(row[token_index + 3])
            repeat_penalty = 1.0
            if out and token == out[-1]:
                repeat_penalty *= 0.10
            if len(out) >= 2 and token == out[-2]:
                repeat_penalty *= 0.35
            if (
                token in {".", "!", "?", ",", ";", ":"}
                and sum(
                    1
                    for item in out
                    if item not in {".", "!", "?", ",", ";", ":"}
                ) < 2
            ):
                repeat_penalty *= 0.05
            weighted.append(
                (
                    token,
                    self._word_weight(
                        n,
                        reward,
                        last_seen,
                        arousal,
                        repeat_penalty,
                    ),
                )
            )
        return self._weighted_choice(weighted)

    def _format_word_tokens(self, tokens: list[str]) -> str:
        text = ""
        punctuation = {".", "!", "?", ",", ";", ":"}
        for token in tokens:
            if token == WORD_END:
                break
            if token in punctuation:
                text = text.rstrip() + token
            else:
                if text and not text.endswith(" "):
                    text += " "
                text += token
        text = re.sub(r"\s+", " ", text).strip(" ,;:")
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        return text

    def _char_trigrams_for_text(
        self,
        text: str,
    ) -> list[tuple[str, str, str]]:
        chars = self.characters(text)
        seq = [START_A, START_B] + chars
        return [
            (a, b, cc)
            for a, b, cc in zip(seq, seq[1:], seq[2:])
        ]

    def _generate_words(
        self,
        context: str,
        arousal: float,
        brain_word_score: Callable[[str], float] | None = None,
    ) -> str | None:
        if not self.hybrid_word_enabled:
            return None

        vocab = int(
            self.db.execute(
                "SELECT COUNT(*) FROM word_unigram"
            ).fetchone()[0]
        )
        if vocab < 8:
            return None

        brain_ready = bool(
            self.connectome_word_control_enabled
            and vocab >= self.connectome_word_control_min_vocab
        )
        brain_active = bool(
            brain_ready
            and brain_word_score is not None
            and self.connectome_word_control_strength > 0.0
        )
        brain_score_cache: dict[str, float] = {}
        brain_scores_used: list[float] = []
        self._brain_word_control_last = {
            "active": brain_active,
            "ready": brain_ready,
            "vocab": vocab,
            "min_vocab": self.connectome_word_control_min_vocab,
            "evaluated": 0,
            "mean_score": 0.5,
        }

        def get_brain_score(token: str) -> float:
            if not brain_active or brain_word_score is None:
                return 0.5
            cached = brain_score_cache.get(token)
            if cached is not None:
                return cached
            try:
                score = max(
                    0.0,
                    min(1.0, float(brain_word_score(token))),
                )
            except Exception:
                score = 0.5
            brain_score_cache[token] = score
            brain_scores_used.append(score)
            return score

        arousal = max(0.0, min(1.0, float(arousal)))
        ctx = self.words(context)
        context_bigrams = set(zip(ctx, ctx[1:]))
        context_trigrams = set(zip(ctx, ctx[1:], ctx[2:]))
        out: list[str] = []
        state: tuple[str, str] | None = None
        punctuation = {".", "!", "?", ",", ";", ":"}

        def choose_mixed(
            sources: list[tuple[float, list[tuple]]],
            *,
            allow_end: bool = True,
        ) -> str | None:
            combined: dict[str, float] = {}
            lexical_count = sum(
                1 for item in out if item not in punctuation
            )

            for source_weight, rows in sources:
                if source_weight <= 0.0 or not rows:
                    continue

                local: list[tuple[str, float]] = []
                for row in rows:
                    token = str(row[0])
                    if not allow_end and token == WORD_END:
                        continue

                    n = int(row[1])
                    reward = float(row[2])
                    last_seen = float(row[3])
                    repeat_penalty = 1.0

                    if out and token == out[-1]:
                        repeat_penalty *= 0.08
                    if len(out) >= 2 and token == out[-2]:
                        repeat_penalty *= 0.30
                    if (
                        token in punctuation
                        and lexical_count < 2
                    ):
                        repeat_penalty *= 0.04

                    # Do not simply mirror the current prompt. Learned
                    # transitions still matter, but exact prompt n-grams are
                    # deliberately less attractive than new combinations.
                    if out and (out[-1], token) in context_bigrams:
                        repeat_penalty *= 0.55
                    if (
                        len(out) >= 2
                        and (out[-2], out[-1], token)
                        in context_trigrams
                    ):
                        repeat_penalty *= 0.38

                    local.append(
                        (
                            token,
                            self._word_weight(
                                n,
                                reward,
                                last_seen,
                                arousal,
                                repeat_penalty,
                            ),
                        )
                    )

                local_total = sum(weight for _, weight in local)
                if local_total <= 0.0:
                    continue

                # Normalize every Markov order independently first. This makes
                # the interpolation weights meaningful even when one table has
                # many more candidates or much larger raw counts.
                for token, weight in local:
                    combined[token] = combined.get(token, 0.0) + (
                        source_weight * weight / local_total
                    )

            if not combined:
                return None

            if brain_active:
                ranked = sorted(
                    combined.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
                adjusted = dict(combined)
                checked = 0
                for token, base_weight in ranked:
                    if token in punctuation or token == WORD_END:
                        continue
                    score = get_brain_score(token)
                    centered = (score - 0.5) * 2.0
                    multiplier = math.exp(
                        self.connectome_word_control_strength
                        * centered
                    )
                    adjusted[token] = base_weight * multiplier
                    checked += 1
                    if checked >= self.connectome_word_control_candidates:
                        break
                combined = adjusted

            return self._weighted_choice(list(combined.items()))

        def source_weights(
            trigram_rows: list[tuple],
            bigram_rows: list[tuple],
            generated_tokens: int,
        ) -> tuple[float, float, float]:
            # Calm state follows grammar more closely; exploration deliberately
            # backs off to shorter history so separate learned phrases can cross.
            tri = 0.55 - 0.25 * arousal
            bi = 0.30 + 0.05 * arousal
            uni = 1.0 - tri - bi

            # A single known continuation is effectively memorized text. Reduce
            # its authority so the model gets a real chance to invent a branch.
            if len(trigram_rows) <= 1:
                moved = tri * 0.55
                tri -= moved
                bi += moved * 0.55
                uni += moved * 0.45
            if len(bigram_rows) <= 1:
                moved = bi * 0.25
                bi -= moved
                uni += moved

            # The longer one exact path survives, the more strongly we invite a
            # backoff. This prevents long verbatim runs through training text.
            if generated_tokens >= 3:
                moved = min(tri * 0.45, 0.04 * generated_tokens)
                tri -= moved
                bi += moved * 0.45
                uni += moved * 0.55

            total = tri + bi + uni
            if total <= 0.0:
                return 0.0, 0.0, 1.0
            return tri / total, bi / total, uni / total

        unigram_rows = self.db.execute(
            "SELECT token,n,reward,last_seen FROM word_unigram "
            "ORDER BY n DESC LIMIT 260"
        ).fetchall()

        # First word: use the current context as a hint, not as a hard path.
        if ctx:
            trigram_rows: list[tuple] = []
            if len(ctx) >= 2:
                trigram_rows = self.db.execute(
                    "SELECT c,n,reward,last_seen FROM word_trigram "
                    "WHERE a=? AND b=? ORDER BY n DESC LIMIT 120",
                    (ctx[-2], ctx[-1]),
                ).fetchall()

            bigram_rows = self.db.execute(
                "SELECT b,n,reward,last_seen FROM word_bigram "
                "WHERE a=? ORDER BY n DESC LIMIT 160",
                (ctx[-1],),
            ).fetchall()

            tri_w, bi_w, uni_w = source_weights(
                trigram_rows,
                bigram_rows,
                0,
            )
            token = choose_mixed(
                [
                    (tri_w, trigram_rows),
                    (bi_w, bigram_rows),
                    (uni_w, unigram_rows),
                ],
                allow_end=False,
            )
            if token is not None:
                out.append(token)
                state = (ctx[-1], token)

        # No useful contextual start: choose a learned sentence opening, but
        # only as a seed. Later tokens still use interpolated backoff.
        if state is None:
            rows = self.db.execute(
                "SELECT a,b,n,last_seen FROM word_starts "
                "ORDER BY n DESC LIMIT 250"
            ).fetchall()
            weighted: list[tuple[str, float]] = []
            for a, b, n, last_seen in rows:
                if a in punctuation:
                    continue
                packed = str(a) + "\u0000" + str(b)
                weighted.append(
                    (
                        packed,
                        (max(1.0, float(n)) ** 0.78)
                        * self._recent_multiplier(float(last_seen)),
                    )
                )
            packed = self._weighted_choice(weighted)
            if packed:
                a, b = packed.split("\u0000", 1)
                if b == WORD_END:
                    out.append(a)
                    state = (WORD_START_B, a)
                else:
                    out.extend([a, b])
                    state = (a, b)

        if state is None:
            return None

        max_tokens = max(4, int(self.word_max_tokens))
        base_tokens = min(7, max_tokens)
        target_tokens = min(
            max_tokens,
            max(
                4,
                int(
                    base_tokens
                    + arousal * max(0, max_tokens - base_tokens)
                ),
            ),
        )

        while len(out) < target_tokens:
            a, b = state
            trigram_rows = self.db.execute(
                "SELECT c,n,reward,last_seen FROM word_trigram "
                "WHERE a=? AND b=? ORDER BY n DESC LIMIT 140",
                (a, b),
            ).fetchall()
            bigram_rows = self.db.execute(
                "SELECT b,n,reward,last_seen FROM word_bigram "
                "WHERE a=? ORDER BY n DESC LIMIT 170",
                (b,),
            ).fetchall()

            tri_w, bi_w, uni_w = source_weights(
                trigram_rows,
                bigram_rows,
                len(out),
            )
            token = choose_mixed(
                [
                    (tri_w, trigram_rows),
                    (bi_w, bigram_rows),
                    (uni_w, unigram_rows),
                ]
            )

            if token is None or token == WORD_END:
                break

            out.append(token)
            state = (b, token)

            lexical = sum(
                1
                for item in out
                if item not in punctuation
            )
            if (
                token in {".", "!", "?"}
                and lexical >= 4
                and self.rng.random() < 0.90
            ):
                break

        text = self._format_word_tokens(out)
        if len(text) < 3:
            return None

        if len(text) > self.max_chars:
            text = text[: self.max_chars]
            last_space = text.rfind(" ")
            if last_space > self.max_chars * 0.55:
                text = text[:last_space]
            text = text.rstrip(" ,;:")

        if brain_active:
            self._brain_word_control_last["evaluated"] = len(
                brain_score_cache
            )
            self._brain_word_control_last["mean_score"] = (
                sum(brain_scores_used) / len(brain_scores_used)
                if brain_scores_used
                else 0.5
            )

        return text

    def _word_output_too_close_to_context(
        self,
        generated: str,
        context: str,
    ) -> bool:
        generated_words = [
            token
            for token in self.words(generated)
            if token not in {".", "!", "?", ",", ";", ":"}
        ]
        context_words = [
            token
            for token in self.words(context)
            if token not in {".", "!", "?", ",", ";", ":"}
        ]
        if not generated_words or not context_words:
            return False

        if generated_words == context_words:
            return True

        if len(generated_words) >= 2 and len(context_words) >= 2:
            context_bigrams = set(
                zip(context_words, context_words[1:])
            )
            generated_bigrams = list(
                zip(generated_words, generated_words[1:])
            )
            shared = sum(
                pair in context_bigrams
                for pair in generated_bigrams
            )
            if shared / max(1, len(generated_bigrams)) >= 0.75:
                return True

        return False

    def generate(
        self,
        context: str = "",
        arousal: float = 0.5,
        brain_word_score: Callable[[str], float] | None = None,
    ) -> tuple[str | None, list[tuple[str, str, str]]]:
        if not self.ready():
            self._last_generator = "not-ready"
            return None, []

        if (
            self.hybrid_word_enabled
            and self.rng.random() < self.word_model_probability
        ):
            word_text = None
            for attempt in range(4):
                candidate = self._generate_words(
                    context,
                    arousal,
                    brain_word_score=brain_word_score,
                )
                if not candidate:
                    continue
                word_text = candidate
                if (
                    attempt >= 3
                    or not self._word_output_too_close_to_context(
                        candidate,
                        context,
                    )
                ):
                    break
            if word_text:
                self._last_generator = "words"
                return (
                    word_text,
                    self._char_trigrams_for_text(word_text),
                )

        self._last_generator = "characters"
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
                int(35 + float(arousal) * max(0, self.max_chars - 35)),
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

    def _reinforce_words(self, text: str, amount: float) -> None:
        tokens = self.words(text)
        if not tokens:
            return

        amount = max(-1.0, min(1.0, float(amount)))
        scale = self.word_reward_scale * amount
        cur = self.db.cursor()

        unigram_counts = Counter(tokens)
        for token, mult in unigram_counts.items():
            cur.execute(
                "UPDATE word_unigram "
                "SET reward=MAX(-2.0,MIN(2.0,reward+?)) "
                "WHERE token=?",
                (scale * min(mult, 4), token),
            )

        transition_tokens = tokens + [WORD_END]
        bigram_counts = Counter(
            zip(transition_tokens, transition_tokens[1:])
        )
        for (a, b), mult in bigram_counts.items():
            cur.execute(
                "UPDATE word_bigram "
                "SET reward=MAX(-2.0,MIN(2.0,reward+?)) "
                "WHERE a=? AND b=?",
                (scale * min(mult, 4), a, b),
            )

        trigram_counts = Counter(
            zip(
                transition_tokens,
                transition_tokens[1:],
                transition_tokens[2:],
            )
        )
        for (a, b, cc), mult in trigram_counts.items():
            cur.execute(
                "UPDATE word_trigram "
                "SET reward=MAX(-2.0,MIN(2.0,reward+?)) "
                "WHERE a=? AND b=? AND c=?",
                (scale * min(mult, 4), a, b, cc),
            )

        self.db.commit()

    def reinforce_text(self, text: str, amount: float) -> None:
        chars = self.characters(text)
        if chars:
            seq = [START_A, START_B] + chars
            trigrams = [
                (a, b, cc)
                for a, b, cc in zip(seq, seq[1:], seq[2:])
            ]
            self.reinforce(trigrams, amount)
        self._reinforce_words(text, amount)

    def get_user_affinity(self, user_id: int) -> float:
        row = self.db.execute(
            "SELECT affinity FROM social_user_affinity WHERE user_id=?",
            (int(user_id),),
        ).fetchone()
        return float(row[0]) if row else 0.0

    def adjust_user_affinity(
        self,
        user_id: int,
        display_name: str,
        delta: float,
        reaction_kind: str | None = None,
    ) -> float:
        user_id = int(user_id)
        delta = max(-1.0, min(1.0, float(delta)))
        positive = 1 if reaction_kind == "positive" else 0
        negative = 1 if reaction_kind == "negative" else 0
        self.db.execute(
            """
            INSERT INTO social_user_affinity(
                user_id, display_name, affinity,
                positive_reactions, negative_reactions, updated_at
            ) VALUES(?,?,?,?,?,strftime('%s','now'))
            ON CONFLICT(user_id) DO UPDATE SET
                display_name=excluded.display_name,
                affinity=MAX(-1.0, MIN(1.0, social_user_affinity.affinity + excluded.affinity)),
                positive_reactions=social_user_affinity.positive_reactions + excluded.positive_reactions,
                negative_reactions=social_user_affinity.negative_reactions + excluded.negative_reactions,
                updated_at=excluded.updated_at
            """,
            (
                user_id,
                str(display_name)[:120],
                delta,
                positive,
                negative,
            ),
        )
        self.db.commit()
        return self.get_user_affinity(user_id)

    def user_affinities(self, limit: int = 50) -> list[dict]:
        rows = self.db.execute(
            """
            SELECT user_id, display_name, affinity,
                   positive_reactions, negative_reactions, updated_at
            FROM social_user_affinity
            ORDER BY affinity DESC, updated_at DESC
            LIMIT ?
            """,
            (max(1, min(200, int(limit))),),
        ).fetchall()
        return [
            {
                "user_id": int(user_id),
                "display_name": str(display_name or user_id),
                "affinity": float(affinity),
                "positive_reactions": int(positive),
                "negative_reactions": int(negative),
                "updated_at": float(updated_at),
            }
            for (
                user_id,
                display_name,
                affinity,
                positive,
                negative,
                updated_at,
            ) in rows
        ]

    def record_word_feedback(
        self,
        word: str,
        user_id: int,
        amount: float,
    ) -> dict:
        normalized = self.normalize(word).lower().strip()
        if not normalized or " " in normalized:
            return {}
        amount = max(-1.0, min(1.0, float(amount)))
        self.db.execute(
            """
            INSERT INTO social_word_feedback(word, confirmations, reward, updated_at)
            VALUES(?,1,?,strftime('%s','now'))
            ON CONFLICT(word) DO UPDATE SET
                confirmations=social_word_feedback.confirmations+1,
                reward=MAX(-2.0, MIN(2.0, social_word_feedback.reward+excluded.reward)),
                updated_at=excluded.updated_at
            """,
            (normalized, amount),
        )
        self.db.execute(
            "INSERT OR IGNORE INTO social_word_user(word,user_id) VALUES(?,?)",
            (normalized, int(user_id)),
        )
        self.db.commit()
        row = self.db.execute(
            """
            SELECT f.confirmations, f.reward, f.updated_at,
                   COUNT(u.user_id)
            FROM social_word_feedback f
            LEFT JOIN social_word_user u ON u.word=f.word
            WHERE f.word=?
            GROUP BY f.word
            """,
            (normalized,),
        ).fetchone()
        if not row:
            return {}
        return {
            "word": normalized,
            "confirmations": int(row[0]),
            "reward": float(row[1]),
            "unique_users": int(row[3]),
            "updated_at": float(row[2]),
        }

    def top_word_feedback(self, limit: int = 20) -> list[dict]:
        rows = self.db.execute(
            """
            SELECT f.word, f.confirmations, f.reward, f.updated_at,
                   COUNT(u.user_id) AS unique_users
            FROM social_word_feedback f
            LEFT JOIN social_word_user u ON u.word=f.word
            GROUP BY f.word
            ORDER BY unique_users DESC, f.confirmations DESC, f.reward DESC
            LIMIT ?
            """,
            (max(1, min(100, int(limit))),),
        ).fetchall()
        return [
            {
                "word": str(word),
                "confirmations": int(confirmations),
                "reward": float(reward),
                "unique_users": int(unique_users),
                "updated_at": float(updated_at),
            }
            for word, confirmations, reward, updated_at, unique_users in rows
        ]

    def close(self) -> None:
        self.db.close()
