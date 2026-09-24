from pathlib import Path
import shutil
import tempfile
import subprocess
import sys

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
        b.reward(1)
        b.save()

        lang = OnlineLanguage(td / "lang.sqlite3", 5, 3, 20)
        for s in ["siema co tam", "co tam u ciebie", "siema no co", "u mnie git"]:
            lang.learn(s)
        assert lang.ready()
        text, tri = lang.generate("siema")
        assert text
        lang.reinforce(tri, 1)
        lang.close()
        print("SMOKE TEST OK", text)
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
