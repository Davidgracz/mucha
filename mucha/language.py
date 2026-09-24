from __future__ import annotations

import math
import random
import re
import sqlite3
from collections import Counter
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+(?:['’\-][A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż0-9]+)*|[^\w\s]", re.UNICODE)
MENTION_RE = re.compile(r"<@!?\d+>|<@&\d+>|<#\d+>")
URL_RE = re.compile(r"https?://\S+", re.I)


class OnlineLanguage:
    """A zero-pretraining online trigram learner.

    It stores counts, not raw Discord messages. Output vocabulary and phrase
    structure therefore emerge only from messages seen on the server.
    """

    def __init__(self, db_path: str | Path, min_tokens: int, min_unique: int, max_tokens: int, seed: int = 67):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.rng = random.Random(seed)
        self.min_tokens = min_tokens
        self.min_unique = min_unique
        self.max_tokens = max_tokens
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS unigram(token TEXT PRIMARY KEY, n INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS bigram(a TEXT NOT NULL, b TEXT NOT NULL, n INTEGER NOT NULL,
                PRIMARY KEY(a,b));
            CREATE TABLE IF NOT EXISTS trigram(a TEXT NOT NULL, b TEXT NOT NULL, c TEXT NOT NULL, n INTEGER NOT NULL,
                reward REAL NOT NULL DEFAULT 0, PRIMARY KEY(a,b,c));
            CREATE TABLE IF NOT EXISTS starts(a TEXT NOT NULL, b TEXT NOT NULL, n INTEGER NOT NULL,
                PRIMARY KEY(a,b));
            CREATE TABLE IF NOT EXISTS stats(k TEXT PRIMARY KEY, v INTEGER NOT NULL);
            """
        )
        self.db.commit()

    @staticmethod
    def normalize(text: str) -> str:
        text = MENTION_RE.sub(" @user ", text)
        text = URL_RE.sub(" <url> ", text)
        text = text.replace("@everyone", "everyone").replace("@here", "here")
        return " ".join(text.split())[:1800]

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        return TOKEN_RE.findall(cls.normalize(text).lower())

    def learn(self, text: str) -> int:
        toks = self.tokenize(text)
        if len(toks) < 1:
            return 0
        toks = toks[:180]
        cur = self.db.cursor()
        for t in toks:
            cur.execute(
                "INSERT INTO unigram(token,n) VALUES(?,1) ON CONFLICT(token) DO UPDATE SET n=n+1", (t,)
            )
        for a, b in zip(toks, toks[1:]):
            cur.execute(
                "INSERT INTO bigram(a,b,n) VALUES(?,?,1) ON CONFLICT(a,b) DO UPDATE SET n=n+1", (a,b)
            )
        for a, b, c in zip(toks, toks[1:], toks[2:]):
            cur.execute(
                "INSERT INTO trigram(a,b,c,n,reward) VALUES(?,?,?,1,0) "
                "ON CONFLICT(a,b,c) DO UPDATE SET n=n+1", (a,b,c)
            )
        if len(toks) >= 2:
            cur.execute(
                "INSERT INTO starts(a,b,n) VALUES(?,?,1) ON CONFLICT(a,b) DO UPDATE SET n=n+1", (toks[0], toks[1])
            )
        cur.execute(
            "INSERT INTO stats(k,v) VALUES('tokens',?) ON CONFLICT(k) DO UPDATE SET v=v+excluded.v", (len(toks),)
        )
        self.db.commit()
        return len(toks)

    def stats(self) -> tuple[int, int]:
        row = self.db.execute("SELECT v FROM stats WHERE k='tokens'").fetchone()
        total = int(row[0]) if row else 0
        unique = int(self.db.execute("SELECT COUNT(*) FROM unigram").fetchone()[0])
        return total, unique

    def ready(self) -> bool:
        total, unique = self.stats()
        return total >= self.min_tokens and unique >= self.min_unique

    def _weighted_choice(self, rows: list[tuple[str, float]]) -> str | None:
        if not rows:
            return None
        vals = [(token, max(0.0001, float(weight))) for token, weight in rows]
        total = sum(w for _, w in vals)
        r = self.rng.random() * total
        upto = 0.0
        for token, w in vals:
            upto += w
            if upto >= r:
                return token
        return vals[-1][0]

    def _pick_start(self, context_tokens: list[str]) -> tuple[str, str] | None:
        if len(context_tokens) >= 2:
            a, b = context_tokens[-2], context_tokens[-1]
            exists = self.db.execute("SELECT 1 FROM trigram WHERE a=? AND b=? LIMIT 1", (a,b)).fetchone()
            if exists:
                return a, b
        if context_tokens:
            b = context_tokens[-1]
            rows = self.db.execute("SELECT a,b,n FROM starts WHERE a=? OR b=? ORDER BY n DESC LIMIT 80", (b,b)).fetchall()
            if rows:
                options = [((a, bb), n) for a, bb, n in rows]
                total = sum(n for _, n in options)
                r = self.rng.random() * total
                s = 0
                for pair, n in options:
                    s += n
                    if s >= r:
                        return pair
        rows = self.db.execute("SELECT a,b,n FROM starts ORDER BY n DESC LIMIT 150").fetchall()
        if not rows:
            return None
        pair = self._weighted_choice([(f"{a}\0{b}", n) for a,b,n in rows])
        if pair is None:
            return None
        a, b = pair.split("\0", 1)
        return a, b

    def generate(self, context: str = "", arousal: float = 0.5) -> tuple[str | None, list[tuple[str,str,str]]]:
        if not self.ready():
            return None, []
        ctx = self.tokenize(context)
        start = self._pick_start(ctx)
        if start is None:
            return None, []
        a, b = start
        out = [a, b]
        used_trigrams: list[tuple[str,str,str]] = []
        target = max(4, min(self.max_tokens, int(7 + arousal * (self.max_tokens - 7))))
        for _ in range(target - 2):
            rows = self.db.execute(
                "SELECT c,n,reward FROM trigram WHERE a=? AND b=? ORDER BY n DESC LIMIT 120", (a,b)
            ).fetchall()
            weighted = []
            for c, n, reward in rows:
                rep = 0.25 if len(out) >= 2 and c == out[-1] == out[-2] else 1.0
                weighted.append((c, (float(n) ** 0.72) * math.exp(max(-2.0, min(2.0, reward))) * rep))
            c = self._weighted_choice(weighted)
            if c is None:
                rows2 = self.db.execute("SELECT b,n FROM bigram WHERE a=? ORDER BY n DESC LIMIT 100", (b,)).fetchall()
                c = self._weighted_choice([(tok, n ** 0.72) for tok, n in rows2])
            if c is None:
                break
            used_trigrams.append((a,b,c))
            out.append(c)
            a, b = b, c
            if c in (".", "!", "?") and len(out) >= 5 and self.rng.random() < 0.75:
                break

        text = self.detokenize(out)
        if not text or len(text) < 2:
            return None, used_trigrams
        return text[:700], used_trigrams

    @staticmethod
    def detokenize(tokens: list[str]) -> str:
        if not tokens:
            return ""
        no_space_before = set(".,!?;:%)]}")
        no_space_after = set("([{")
        s = tokens[0]
        for t in tokens[1:]:
            if t in no_space_before or (s and s[-1] in no_space_after):
                s += t
            else:
                s += " " + t
        s = s.replace("@everyone", "＠everyone").replace("@here", "＠here")
        s = re.sub(r"<@!?\d+>", "@user", s)
        return s.strip()

    def reinforce(self, trigrams: list[tuple[str,str,str]], amount: float) -> None:
        if not trigrams:
            return
        amount = max(-1.0, min(1.0, float(amount)))
        cur = self.db.cursor()
        counts = Counter(trigrams)
        for (a,b,c), mult in counts.items():
            cur.execute(
                "UPDATE trigram SET reward=MAX(-2.0, MIN(2.0, reward + ?)) WHERE a=? AND b=? AND c=?",
                (0.08 * amount * min(mult, 3), a,b,c),
            )
        self.db.commit()

    def close(self) -> None:
        self.db.close()
