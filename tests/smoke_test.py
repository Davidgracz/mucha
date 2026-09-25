from pathlib import Path
import shutil
import tempfile
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mucha.config import BrainConfig
from mucha.connectome import Connectome
from mucha.brain import FlyBrain
from mucha.language import OnlineLanguage


def main():
    td = Path(tempfile.mkdtemp(prefix="mucha-test-"))
    try:
        subprocess.check_call([sys.executable, "tools/make_demo_connectome.py", "--output", str(td / "c"), "--neurons", "800", "--connections", "9000"])
        c = Connectome.load(td / "c")
        cfg = BrainConfig(connectome_dir=td / "c", state_file=td / "brain.npz")
        b = FlyBrain(c, cfg)
        b.inject_text("siema mucha co tam?", 123, True)
        b.step(5)
        scores = b.action_scores()
        assert 0 <= scores["speak"] <= 1
        assert 0 <= b.language_word_score("siema") <= 1
        b.reward(1)
        b.save()

        lang = OnlineLanguage(
            td / "lang.sqlite3",
            5,
            3,
            80,
            word_model_probability=1.0,
            word_recent_boost=2.0,
            connectome_word_control_min_vocab=8,
            connectome_word_control_strength=0.35,
        )
        for s in [
            "siema co tam",
            "co tam u ciebie",
            "siema no co",
            "u mnie git",
        ]:
            lang.learn(s)
        assert lang.ready()
        diag = lang.diagnostics()
        assert diag["word_vocab"] >= 6
        assert diag["word_bigrams"] >= 4
        text, tri = lang.generate(
            "siema",
            brain_word_score=b.language_word_score,
        )
        assert text
        assert lang.diagnostics()["last_generator"] == "words"

        # The word generator should recombine learned transitions instead of
        # walking one memorized sentence path every time.
        variants = set()
        for _ in range(24):
            generated, _ = lang.generate(
                "siema co tam",
                arousal=0.80,
                brain_word_score=b.language_word_score,
            )
            if generated:
                variants.add(generated.lower())
        assert len(variants) >= 3
        assert "siema co tam" not in variants
        brain_diag = lang.diagnostics()
        assert brain_diag["connectome_word_control_ready"]
        assert brain_diag["connectome_word_control_last"]["active"]
        assert brain_diag["connectome_word_control_last"]["evaluated"] > 0

        lang.reinforce_text(text, 1)
        lang.close()
        print("SMOKE TEST OK", text)
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
