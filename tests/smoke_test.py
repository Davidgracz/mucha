from pathlib import Path
import shutil
import tempfile
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mucha.config import BrainConfig
from mucha.connectome import Connectome
from mucha.brain import FlyBrain
from mucha.language import OnlineLanguage
from mucha.web_ui import ASSOCIATIONS_HTML, CONFIG_HTML, CONNECTOME_HTML, NEUROMAP_HTML, PUBLIC_OVERVIEW_HTML


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
        pools = b.action_pool_diagnostics()
        assert set(pools) == set(b.ACTIONS)
        assert pools["speak"]["seed_count"] > 0
        assert pools["speak"]["mode"] == "adaptive-learned-connectome"
        assert pools["speak"]["seed_kind"] == "sensory-cues"
        assert pools["speak"]["seed_types"]
        assert pools["speak"]["top_seed_activity"]
        assert pools["speak"]["pool_size"] > 0
        assert pools["voice_move"]["seed_count"] > 0
        assert pools["voice_leave"]["seed_count"] > 0
        assert pools["stay"]["seed_count"] > 0
        assert not np.array_equal(
            b._action_output_pools["speak"],
            b._action_output_pools["voice_leave"],
        )
        assert 0 <= b.language_word_score("siema") <= 1

        nm = b.neuromodulator_diagnostics()
        assert nm["enabled"]
        assert nm["dopamine"]["neurons"] > 0
        assert nm["serotonin"]["neurons"] > 0
        assert nm["octopamine"]["neurons"] > 0
        typed_modulators = np.unique(
            np.concatenate([
                b._modulator_pools["dopamine"],
                b._modulator_pools["serotonin"],
                b._modulator_pools["octopamine"],
            ])
        )
        tested_residual = False
        for mod_idx in typed_modulators:
            base_mass = float(
                np.abs(c.matrix[:, int(mod_idx)]).sum()
            )
            if base_mass <= 1e-9:
                continue
            runtime_mass = float(
                np.abs(
                    b._runtime_matrix_cpu[:, int(mod_idx)]
                ).sum()
            )
            assert np.isclose(
                runtime_mass,
                base_mass
                * cfg.neuromodulatory_direct_residual,
                rtol=1e-4,
                atol=1e-7,
            )
            tested_residual = True
            break
        assert tested_residual
        for name in ("dopamine", "serotonin", "octopamine"):
            pool = b._modulator_pools[name]
            idx = b._backend_indices(pool[:1])
            b.state[idx] = np.float32(0.9)
        b.step(2)
        nm = b.neuromodulator_diagnostics()
        assert nm["dopamine"]["level"] > 0
        assert nm["serotonin"]["level"] > 0
        assert nm["octopamine"]["level"] > 0
        assert nm["plasticity_gain"] > 1.0
        assert nm["effective_gain"] > cfg.propagation_gain
        assert nm["effective_leak"] > cfg.leak

        b.mark_language_output("siema mucha dobry tekst")
        output_trace = b.capture_learning_trace(256)
        assert len(output_trace[0]) > 0
        assert "x" in c.neuron_meta
        assert "cell_class" in c.neuron_meta
        assert "primary_neuropil" in c.neuron_meta
        neuro = b.neuro_map_snapshot(80, "xy")
        assert neuro["coordinate_mode"] == "synthetic"
        assert neuro["region_source"] == "neuropil"
        assert neuro["neuropil_labels"] == 3
        assert len(neuro["nodes"]) >= 40
        assert len(neuro["reference"]) > 0
        assert neuro["regions"]
        assert neuro["regions"][0]["history"]
        for _ in range(9):
            b.step(1)
            neuro = b.neuro_map_snapshot(80, "xy")
        assert neuro["history_samples"] >= 10
        assert neuro["regions"][0]["correlation_samples"] >= 10
        assert "Fly Brain Neuro-map" in NEUROMAP_HTML
        assert "/api/neuromap" in NEUROMAP_HTML
        assert "Runtime correlation" in NEUROMAP_HTML
        assert "FOLLOW ACTIVITY" in CONNECTOME_HTML
        assert "Action Circuits • biology + learned" in CONNECTOME_HTML
        assert "biological-circuits" in CONNECTOME_HTML
        assert "connected-outputs" in CONNECTOME_HTML
        assert "isolated-nodes" in CONNECTOME_HTML
        assert "Direction arrow" in CONNECTOME_HTML
        assert "Neuromodulation v2" in CONNECTOME_HTML
        assert "neuromod-da" in CONNECTOME_HTML
        assert "Zapisz i zrestartuj Muchę" in CONFIG_HTML
        assert "Neuromodulatory v2" in CONFIG_HTML
        assert "dopamine_plasticity_gain" in CONFIG_HTML
        assert 'const out={brain:{},language:{},behavior:{},voice:{}};' in CONFIG_HTML
        bot_source = (ROOT / "mucha" / "discord_bot.py").read_text(
            encoding="utf-8"
        )
        for hard_blocked_id in (
            "344519890083774475",
            "506193122460434443",
            "784590857633267713",
        ):
            assert hard_blocked_id in bot_source
        assert "HARD_BLOCKED_TEXT_CHANNEL_IDS" in bot_source
        assert "NA SZTYWNO" in CONFIG_HTML
        assert "\\n  [\"connectome_word_control_enabled\"" not in CONFIG_HTML
        assert "Mucha — publiczny podgląd" in PUBLIC_OVERVIEW_HTML
        assert "/api/public/state" in PUBLIC_OVERVIEW_HTML
        assert "Mapa skojarzeń Muchy" in ASSOCIATIONS_HTML
        assert "/api/associations" in ASSOCIATIONS_HTML

        assoc = b.word_association_snapshot(
            ["siema", "mucha", "ciebie", "spacer"],
            max_nodes=8,
            edge_limit=12,
        )
        assert assoc["node_count"] == 4
        assert assoc["method"].startswith("word sensory pools")
        assert all(0.0 <= edge["weight"] <= 1.0 for edge in assoc["edges"])

        visual = b.connectome_visual_snapshot(24, 60)
        assert visual["selected_neurons"] >= 12
        assert len(visual["nodes"]) == visual["selected_neurons"]
        assert visual["total_neurons"] == c.n_neurons
        assert visual["total_connections"] == int(c.matrix.nnz)
        assert visual["selected_edges"] > 0
        assert visual["connected_neurons"] > 0
        assert visual["connected_output_neurons"] > 0
        assert visual["isolated_neurons"] < visual["selected_neurons"]
        assert visual["mode"] == "stable"
        stable_ids = [node["id"] for node in visual["nodes"]]
        b.step(1)
        stable_again = b.connectome_visual_snapshot(24, 60)
        assert [node["id"] for node in stable_again["nodes"]] == stable_ids
        followed = b.connectome_visual_snapshot(
            24,
            60,
            follow_activity=True,
        )
        assert followed["mode"] == "follow_activity"

        # Reward should now change both neuron bias and real existing
        # connectome edges through the sparse learned-synapse overlay.
        coo = c.matrix.tocoo()
        edge_pos = next(
            (
                i
                for i, (row, col) in enumerate(zip(coo.row, coo.col))
                if int(row) != int(col)
            ),
            0,
        )
        edge_trace = (
            np.asarray(
                [coo.row[edge_pos], coo.col[edge_pos]],
                dtype=np.int32,
            ),
            np.asarray([1.0, 1.0], dtype=np.float32),
        )
        learning = b.reward(
            1,
            action="speak",
            trace=edge_trace,
        )
        assert learning["changed_synapses"] >= 1
        assert learning["learned_synapses"] >= 1
        b.save()
        reloaded = FlyBrain(c, cfg)
        assert reloaded.diagnostics()["learned_synapses"] >= 1

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
        feedback_words = []
        ticks_before_language = b.tick_count

        def brain_word_feedback(token, previous):
            feedback_words.append((token, previous))
            b.advance_language_word(
                token,
                previous,
                magnitude=0.12,
                steps=1,
            )

        text, tri = lang.generate(
            "siema",
            brain_word_score=b.language_word_score,
            brain_word_feedback=brain_word_feedback,
        )
        assert text
        assert feedback_words
        assert b.tick_count > ticks_before_language
        assert lang.diagnostics()["last_generator"] == "words"
        assert lang.diagnostics()["connectome_word_control_last"][
            "recurrent_feedback"
        ]
        assert lang.diagnostics()["connectome_word_control_last"][
            "feedback_words"
        ] > 0

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
        association_words = lang.association_words(8)
        assert association_words
        assert all(len(item["word"]) >= 3 for item in association_words)

        lang.reinforce_text(text, 1)
        lang.close()
        print("SMOKE TEST OK", text)
    finally:
        shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    main()
