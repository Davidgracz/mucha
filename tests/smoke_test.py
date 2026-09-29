from pathlib import Path
import shutil
import tempfile
import time
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mucha.config import BrainConfig
from mucha.connectome import Connectome
from mucha.brain import FlyBrain
from mucha.episodic import VoiceEpisodicMemory
from mucha.language import OnlineLanguage
from mucha.web_ui import AFFINITY_HTML, ASSOCIATIONS_HTML, CONFIG_HTML, CONNECTOME_HTML, NEUROMAP_HTML, HTML, OVERVIEW_HTML, PUBLIC_OVERVIEW_HTML


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

        internal = b.internal_state_diagnostics()
        assert internal["enabled"] is True
        assert set(internal["states"]) == {
            "social_need",
            "curiosity",
            "stress",
            "satiety",
            "arousal",
        }
        assert internal["recurrent_edges"] > 0
        for name, row in internal["states"].items():
            assert row["neurons"] > 0
            assert row["entry_neurons"] > 0
            assert row["target_actions"]
            assert row["mode"] == "FAFB-recurrent-attractor"

        attractor_edges = b._internal_attractor_matrix_cpu.tocoo()
        assert attractor_edges.nnz > 0
        for row, col, value in zip(
            attractor_edges.row[:64],
            attractor_edges.col[:64],
            attractor_edges.data[:64],
        ):
            assert value > 0.0
            assert b._runtime_matrix_cpu[int(row), int(col)] > 0.0

        curiosity_entry = b._internal_state_entry_pools["curiosity"]
        assert len(curiosity_entry) > 0
        assert set(curiosity_entry).issubset(set(c.sensory.tolist()))
        cue_diag = b.inject_internal_state_cue(
            "curiosity",
            1.0,
            key="smoke:internal:curiosity",
        )
        assert cue_diag["entry_neurons"] > 0
        assert cue_diag["attractor_neurons"] > 0
        b.step(4)
        curiosity_after = b.internal_state_diagnostics()
        assert curiosity_after["states"]["curiosity"]["mean_abs"] > 0.0

        b.inject("signal-flow-smoke", 1.0, 64)
        b.step(2)
        signal_flow = b.signal_flow_snapshot()
        assert signal_flow["latest"] is not None
        assert signal_flow["latest"]["edges"]
        assert signal_flow["latest"]["winner"] in b.ACTIONS
        assert any(
            cue.get("key") == "signal-flow-smoke"
            for cue in signal_flow["latest"]["cues"]
        )
        first_flow_edge = signal_flow["latest"]["edges"][0]
        assert "source" in first_flow_edge
        assert "target" in first_flow_edge
        assert "contribution" in first_flow_edge
        assert "effective_weight" in first_flow_edge
        assert "attractor_delta" in first_flow_edge
        assert "source_position" in first_flow_edge
        assert "target_position" in first_flow_edge
        assert signal_flow["latest"]["output_points"]
        assert "live edge contribution" in signal_flow["method"]

        episodes = VoiceEpisodicMemory(
            max_events=32,
            learning_rate=0.5,
        )
        assert episodes.predict("ctx", "voice_join") == 0.0
        ep = episodes.observe(
            guild_id=1,
            context="ctx",
            action="voice_join",
            actual_reward=0.6,
            predicted_reward=0.0,
            source="smoke",
        )
        assert abs(ep["prediction_error"] - 0.6) < 1e-9
        assert abs(
            episodes.predict("ctx", "voice_join") - 0.3
        ) < 1e-9
        assert episodes.size() == 1

        persistent_path = td / "voice-episodes.sqlite3"
        persistent = VoiceEpisodicMemory(
            max_events=32,
            learning_rate=0.5,
            database=persistent_path,
            max_persisted_events=100,
        )
        scene_key = persistent.make_scene_key(
            555,
            [22, 11],
        )
        persisted = persistent.observe(
            guild_id=7,
            channel_id=555,
            channel_name="ASG",
            user_ids=[22, 11],
            user_names=["Stivi", "Dawid"],
            context="in|need=0|fatigue=1",
            scene_key=scene_key,
            action="voice_move",
            actual_reward=0.8,
            source="persistent smoke",
        )
        assert persisted["scene_observations"] == 1
        assert persistent.size() == 1
        expected_before = persistent.predict(
            "in|need=0|fatigue=1",
            "voice_move",
            scene_key=scene_key,
        )
        assert expected_before > 0.0
        persistent.close()

        restored = VoiceEpisodicMemory(
            max_events=32,
            learning_rate=0.5,
            database=persistent_path,
            max_persisted_events=100,
        )
        assert restored.size() == 1
        expected_after = restored.predict(
            "in|need=0|fatigue=1",
            "voice_move",
            scene_key=scene_key,
        )
        assert abs(expected_after - expected_before) < 1e-9
        restored_recent = restored.recent(1)[0]
        assert restored_recent["channel_id"] == 555
        assert restored_recent["channel_name"] == "ASG"
        assert restored_recent["user_ids"] == [11, 22]
        assert set(restored_recent["user_names"]) == {
            "Dawid",
            "Stivi",
        }
        assert restored.diagnostics()["persistent"] is True
        replay_candidates = restored.replay_candidates(
            limit=4,
            max_age_seconds=86400,
        )
        assert replay_candidates
        assert replay_candidates[0]["action"] == "voice_move"
        assert replay_candidates[0]["channel_id"] == 555
        assert replay_candidates[0]["replay_score"] > 0.0
        assert replay_candidates[0]["user_ids"] == [11, 22]
        assert replay_candidates[0]["user_names"] == [
            "Dawid",
            "Stivi",
        ]
        strength_before_replay = float(
            replay_candidates[0]["memory_strength"]
        )
        consolidated_memory = restored.consolidate_replay(
            replay_candidates[0],
            0.12,
        )
        assert consolidated_memory["replay_count"] >= 1
        assert (
            consolidated_memory["strength"]
            > strength_before_replay
        )
        top_memories = restored.consolidation_summary(4)
        assert top_memories
        assert top_memories[0]["scene_key"] == scene_key
        strength_before_forgetting = float(
            top_memories[0]["strength"]
        )
        forgetting_diag = restored.apply_forgetting(
            now=time.time() + 14 * 86400,
            force=True,
        )
        assert forgetting_diag["ran"] is True
        assert forgetting_diag["factor"] < 1.0
        assert (
            restored.consolidation_summary(1)[0]["strength"]
            < strength_before_forgetting
        )
        restored.close()

        # Voice behavior is now selected by competition between connectome
        # action readouts. The adapter may mask physically impossible actions,
        # but does not apply join/move/leave thresholds in neural mode.
        state_before_social_drive = b.compute.to_cpu(
            b.state
        ).copy()
        b.inject_voice_decision_context(
            999,
            None,
            connected=False,
            dwell_progress=0.0,
            overstay_level=0.0,
            human_count=0,
            alternatives=3,
            outside_seconds=420.0,
            available_humans=4,
            social_drive_level=1.0,
            social_drive_magnitude=1.4,
        )
        state_after_social_drive = b.compute.to_cpu(
            b.state
        )
        assert np.max(
            np.abs(
                state_after_social_drive
                - state_before_social_drive
            )
        ) > 0.25

        state_before_opportunity = b.compute.to_cpu(
            b.state
        ).copy()
        opportunity_diag = b.inject_voice_reward_opportunity(
            999,
            55555,
            human_count=3,
            strength=1.1,
        )
        assert opportunity_diag["action"] == "voice_join"
        assert opportunity_diag["neurons"] > 0
        assert opportunity_diag["mode"].startswith(
            "connectome-guided"
        )
        assert opportunity_diag["reach_max"] > 0.0
        assert opportunity_diag["mode"] in {
            "connectome-guided-excitatory-sensory",
            "connectome-guided-structural-sensory",
        }
        state_after_opportunity = b.compute.to_cpu(
            b.state
        )
        assert np.max(
            np.abs(
                state_after_opportunity
                - state_before_opportunity
            )
        ) > 0.20
        b.step(2)
        voice_out = b.voice_action_decision(
            connected=False,
            can_join=True,
        )
        assert voice_out["source"] == "connectome-readout-competition"
        assert voice_out["action"] in {"voice_join", "stay"}
        assert set(voice_out["candidates"]) == {"voice_join", "stay"}
        assert voice_out["margin"] >= 0.0
        assert "tie_break" in voice_out
        assert "tie_evidence" in voice_out
        assert "tie_evidence_margin" in voice_out
        assert "raw_winner_margin" in voice_out

        tie_cfg = BrainConfig(
            connectome_dir=td / "c",
            state_file=td / "tie-brain.npz",
        )
        tie_brain = FlyBrain(c, tie_cfg)
        tie_brain.state[...] = 0
        tie_brain.plastic_bias[...] = 0
        tie_brain.eligibility[...] = 0
        tie_brain.reward_trace = 0.0
        flat_voice = tie_brain.voice_action_decision(
            connected=False,
            can_join=True,
        )
        assert flat_voice["tie_break"] == (
            "unbiased-flat-neural-fallback"
        )
        assert set(flat_voice["tie_evidence"]) == {
            "stay",
            "voice_join",
        }
        assert flat_voice["margin"] == 0.0
        assert flat_voice["action"] in {
            "stay",
            "voice_join",
        }

        state_before_homeostasis = b.compute.to_cpu(
            b.state
        ).copy()
        homeostasis_diag = b.inject_voice_decision_context(
            999,
            12345,
            connected=True,
            dwell_progress=1.2,
            overstay_level=0.7,
            human_count=3,
            disliked_strength=0.6,
            alternatives=2,
            social_fatigue_level=0.8,
            social_fatigue_magnitude=1.1,
            habituation_level=0.7,
            habituation_change_magnitude=0.9,
            exploration_drive_level=0.9,
            exploration_drive_magnitude=1.0,
        )
        assert homeostasis_diag["social_fatigue_level"] == 0.8
        assert homeostasis_diag["habituation_level"] == 0.7
        assert homeostasis_diag["exploration_drive_level"] == 0.9
        assert homeostasis_diag["social_fatigue_move"] is not None
        assert homeostasis_diag["social_fatigue_leave"] is not None
        assert homeostasis_diag["habituation"] is not None
        assert homeostasis_diag["exploration_drive"] is not None
        assert "internal_state_cues" in homeostasis_diag
        assert "satiety" in homeostasis_diag["internal_state_cues"]
        assert "curiosity" in homeostasis_diag["internal_state_cues"]
        assert "stress" in homeostasis_diag["internal_state_cues"]
        state_after_homeostasis = b.compute.to_cpu(
            b.state
        )
        assert np.max(
            np.abs(
                state_after_homeostasis
                - state_before_homeostasis
            )
        ) > 0.10
        b.step(2)
        voice_in = b.voice_action_decision(
            connected=True,
            can_move=True,
            can_leave=True,
        )
        assert voice_in["action"] in {
            "voice_move",
            "voice_leave",
            "stay",
        }
        assert set(voice_in["candidates"]) == {
            "voice_move",
            "voice_leave",
            "stay",
        }
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

        # Each Discord user now owns a stable sensory identity plus a
        # connectivity-derived memory assembly. Social reward/punish must
        # change persistent neural memory rather than only the SQL affinity.
        social_uid = 424242
        identity_a = b.user_identity_pool(social_uid).copy()
        identity_b = b.user_identity_pool(social_uid).copy()
        memory_a = b.user_memory_pool(social_uid).copy()
        memory_b = b.user_memory_pool(social_uid).copy()
        assert np.array_equal(identity_a, identity_b)
        assert np.array_equal(memory_a, memory_b)
        assert len(identity_a) > 0
        assert len(memory_a) > 0
        b.activate_user_memory(social_uid, 0.7)
        b.step(2)
        social_positive = b.reinforce_user_memory(
            social_uid,
            0.12,
        )
        assert social_positive["neural_affinity"] > 0.0
        assert social_positive["maturity"] > 0.0
        assert social_positive["learned_synapses"] > 0
        assert social_positive["top_edges"]

        disliked_uid = 434343
        social_negative = b.reinforce_user_memory(
            disliked_uid,
            -0.18,
        )
        assert social_negative["neural_affinity"] < 0.0
        assert social_negative["learned_synapses"] > 0

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
        assert "Path Inspector" in CONNECTOME_HTML
        assert "voice-social-drive" in CONNECTOME_HTML
        assert "voice-reward-opportunity" in CONNECTOME_HTML
        assert "path-list" in CONNECTOME_HTML
        assert "selectAction" in CONNECTOME_HTML
        assert "&action=" in CONNECTOME_HTML
        assert "encodeURIComponent(selectedAction)" in CONNECTOME_HTML
        assert "visualWithSelectedPath" in CONNECTOME_HTML
        assert "Zapisz i zrestartuj Muchę" in CONFIG_HTML
        assert "Neuromodulatory v2" in CONFIG_HTML
        assert "dopamine_plasticity_gain" in CONFIG_HTML
        assert "neural_social_memory_enabled" in CONFIG_HTML
        assert "neural_affinity_weight" in CONFIG_HTML
        assert "connectome_voice_control_enabled" in CONFIG_HTML
        assert "social_drive_enabled" in CONFIG_HTML
        assert "social_drive_max_magnitude" in CONFIG_HTML
        assert "social_drive_stay_punish" in CONFIG_HTML
        assert "social_join_reward" in CONFIG_HTML
        assert "reward_opportunity_enabled" in CONFIG_HTML
        assert "reward_opportunity_success_chance" in CONFIG_HTML
        assert "reward_opportunity_reward" in CONFIG_HTML
        assert "reward_opportunity_stay_punish" in CONFIG_HTML
        assert "reward_opportunity_stay_punish_interval_seconds" in CONFIG_HTML
        assert "motivation_propagation_steps" in CONFIG_HTML
        assert "homeostasis_enabled" in CONFIG_HTML
        assert "social_fatigue_start_seconds" in CONFIG_HTML
        assert "habituation_half_life_seconds" in CONFIG_HTML
        assert "habituation_max_suppression" in CONFIG_HTML
        assert "exploration_drive_enabled" in CONFIG_HTML
        assert "exploration_drive_max_magnitude" in CONFIG_HTML
        assert "episodic_prediction_enabled" in CONFIG_HTML
        assert "episodic_database" in CONFIG_HTML
        assert "episodic_max_persisted_events" in CONFIG_HTML
        assert "episodic_recall_magnitude" in CONFIG_HTML
        assert "prediction_learning_rate" in CONFIG_HTML
        assert "prediction_error_scale" in CONFIG_HTML
        assert "prediction_error_max_correction" in CONFIG_HTML
        assert "prediction_credit_queue_size" in CONFIG_HTML
        assert "prediction_credit_decay_seconds" in CONFIG_HTML
        assert "memory_replay_enabled" in CONFIG_HTML
        assert "memory_replay_idle_seconds" in CONFIG_HTML
        assert "memory_replay_interval_seconds" in CONFIG_HTML
        assert "memory_replay_batch_size" in CONFIG_HTML
        assert "memory_replay_magnitude" in CONFIG_HTML
        assert "memory_replay_reward_scale" in CONFIG_HTML
        assert "memory_replay_steps" in CONFIG_HTML
        assert "memory_replay_max_age_days" in CONFIG_HTML
        assert "episodic_consolidation_gain" in CONFIG_HTML
        assert "episodic_forgetting_half_life_days" in CONFIG_HTML
        assert "consolidation_enabled" in CONFIG_HTML
        assert "synaptic_consolidation_gain" in CONFIG_HTML
        assert "synaptic_consolidation_protection" in CONFIG_HTML
        assert "internal_states_enabled" in CONFIG_HTML
        assert "internal_state_pool_size" in CONFIG_HTML
        assert "internal_state_recurrent_gain" in CONFIG_HTML
        assert "internal_state_arousal_gain" in CONFIG_HTML
        assert "Internal states / attractors" in CONFIG_HTML
        assert "Attention / Working Memory" in CONFIG_HTML
        assert "Learned Action Policy" in CONFIG_HTML
        assert "action_policy_enabled" in CONFIG_HTML
        assert "action_policy_lr" in CONFIG_HTML
        assert "action_policy_max_bias" in CONFIG_HTML
        assert "action_policy_decay" in CONFIG_HTML
        assert "attention_enabled" in CONFIG_HTML
        assert "attention_half_life_seconds" in CONFIG_HTML
        assert "working_memory_seconds" in CONFIG_HTML
        assert "attention_reinject_magnitude" in CONFIG_HTML
        assert "Attention / Working Memory" in HTML
        assert "Learned Action Policy" in HTML
        assert "renderActionPolicy" in HTML
        assert "renderAttention" in HTML
        assert "Jak Mucha doszła do tego, co robi teraz?" in HTML
        assert "data-help-key=\"decision-flow\"" in HTML
        assert "const HELP=" in HTML
        assert "function enhanceHelp()" in HTML
        assert "mode-simple" in HTML
        assert "mode-full" in HTML
        assert "flow-attention" in HTML
        assert "prediction-error" in HTML
        assert "ATTRACTORS" in NEUROMAP_HTML
        assert "renderInternalStates" in NEUROMAP_HTML
        assert "drawAttractors" in NEUROMAP_HTML
        assert "SOCIAL NEED" in NEUROMAP_HTML
        assert "LEARNED SYNAPSES" in NEUROMAP_HTML
        assert "drawLearnedSynapses" in NEUROMAP_HTML
        assert "CONSOLIDATED" in NEUROMAP_HTML
        assert "FADING" in NEUROMAP_HTML
        assert "MEMORY REPLAY" in HTML
        assert "Credit queue" in HTML
        assert "SIGNAL FLOW" in NEUROMAP_HTML
        assert "FOLLOW DECISION" in NEUROMAP_HTML
        assert "Live signal flow" in NEUROMAP_HTML
        assert "renderSignalFlow" in NEUROMAP_HTML
        assert "flow-live-btn" in NEUROMAP_HTML
        assert "Historia — kliknij, aby odtworzyć przepływ" in NEUROMAP_HTML
        assert "flowReplayTick" in NEUROMAP_HTML
        assert "live flow in / out" in NEUROMAP_HTML
        assert "neural tie-break" in HTML
        assert "overstay_punish_amount" in CONFIG_HTML
        assert "overstay_punish_interval_seconds" in CONFIG_HTML
        assert "threat_ramp_seconds" in CONFIG_HTML
        assert "threat_magnitude" in CONFIG_HTML
        assert "Social Neural Memory" in AFFINITY_HTML
        assert "neural-users" in AFFINITY_HTML
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
        assert "_user_affinity_components" in bot_source
        assert "_write_neural_social_memory" in bot_source
        assert "voice_action_decision" in bot_source
        assert "tie_evidence" in bot_source
        assert "tie_break" in bot_source
        assert "action_policy_gate" in bot_source
        assert '"action_policy": action_policy_debug' in bot_source
        assert "inject_voice_decision_context" in bot_source
        assert "_last_social_drive_punish" in bot_source
        assert "neural social drive outside voice" in bot_source
        assert "neural social drive join" in bot_source
        assert "_voice_reward_opportunity_for" in bot_source
        assert "random voice reward opportunity" in bot_source
        assert "ignored neural reward opportunity" in bot_source
        assert "_last_reward_opportunity_stay_punish" in bot_source
        assert "motivation_propagation_steps" in bot_source
        assert "_voice_homeostasis_levels" in bot_source
        assert "habituation_suppression" in bot_source
        assert "social_fatigue_level" in bot_source
        assert "exploration_drive_level" in bot_source
        assert "VoiceEpisodicMemory" in bot_source
        assert "_voice_prediction_context" in bot_source
        assert "_update_pending_voice_scene" in bot_source
        assert "episodic_recall_magnitude" in bot_source
        assert "prediction_scene_key" in bot_source
        assert "prediction_error_max_correction" in bot_source
        assert "_queue_voice_prediction" in bot_source
        assert "prediction_credit_decay_seconds" in bot_source
        assert "_maybe_memory_replay" in bot_source
        assert "memory_replay_reward_scale" in bot_source
        assert "replay_candidates" in bot_source
        assert "_attention_observe_text" in bot_source
        assert "_inject_attention_context" in bot_source
        assert "_attention_language_context" in bot_source
        assert '"attention": attention_debug' in bot_source
        brain_source = (ROOT / "mucha" / "brain.py").read_text(
            encoding="utf-8"
        )
        assert "_capture_signal_flow_tick" in brain_source
        assert "attention_score" in brain_source
        assert "action_policy_score" in brain_source
        assert "action_policy_gate" in brain_source
        assert "action_policy_diagnostics" in brain_source
        assert "connectome-readout+learned-policy" in brain_source
        assert "signal_flow_snapshot" in brain_source
        assert "effective_weight" in brain_source
        assert "top_synapses" in brain_source
        assert "consolidate_and_forget" in brain_source
        assert "learned_synapses_snapshot" in brain_source
        assert "_synaptic_consolidation_map" in brain_source
        assert "_build_internal_state_attractors" in brain_source
        assert "inject_internal_state_cue" in brain_source
        assert "internal_state_diagnostics" in brain_source
        assert "_internal_attractor_edge_map" in brain_source
        assert "internal-state:stress:chaser" in bot_source
        assert "neural_arousal" in bot_source
        assert "effective_arousal" in bot_source
        assert "consolidate_replay" in bot_source
        assert "apply_forgetting" in bot_source
        assert "preferred_channel_id" in bot_source
        assert "NA SZTYWNO" in CONFIG_HTML
        assert "\\n  [\"connectome_word_control_enabled\"" not in CONFIG_HTML
        assert "Mucha — publiczny podgląd" in PUBLIC_OVERVIEW_HTML
        assert "/api/public/state" in PUBLIC_OVERVIEW_HTML
        assert "Mowa / Language Brain" in ASSOCIATIONS_HTML
        assert "Live trace ostatniej generacji" in ASSOCIATIONS_HTML
        assert "Connectome word control" in ASSOCIATIONS_HTML
        assert "brain_score" in ASSOCIATIONS_HTML
        assert "selection_roll" in ASSOCIATIONS_HTML
        assert "Recurrent feedback" in ASSOCIATIONS_HTML
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

        path = b.action_path_snapshot(
            "voice_move",
            max_depth=5,
            max_paths=4,
        )
        assert path["action"] == "voice_move"
        assert path["pool_size"] > 0
        assert path["paths"]
        assert path["method"].startswith("live backward beam trace")
        first_path = path["paths"][0]
        assert first_path["nodes"]
        if first_path["edges"]:
            edge = first_path["edges"][0]
            assert "base_weight" in edge
            assert "learned_delta" in edge
            assert "effective_weight" in edge

        neuro = b.neuro_map_snapshot(
            count=220,
            projection="xy",
        )
        assert "signal_flow" in neuro
        assert neuro["signal_flow"]["latest"] is not None
        assert neuro["nodes"]
        assert "incoming_edges" in neuro["nodes"][0]
        assert "outgoing_edges" in neuro["nodes"][0]
        assert "live_flow_in" in neuro["nodes"][0]
        assert "live_flow_out" in neuro["nodes"][0]
        assert "learned_synapses" in neuro
        assert "internal_states" in neuro
        assert neuro["internal_states"]["states"]
        assert any(
            node.get("internal_states")
            for node in neuro["nodes"]
        )

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
        assert learning["top_synapses"]
        assert "source" in learning["top_synapses"][0]
        assert "target" in learning["top_synapses"][0]
        assert "change" in learning["top_synapses"][0]
        assert "consolidation" in learning["top_synapses"][0]

        learned_before = b.learned_synapses_snapshot(20)
        assert learned_before["total"] >= 1
        first_strength = max(
            edge["consolidation"]
            for edge in learned_before["edges"]
        )
        for _ in range(8):
            b.reward(
                1,
                action="speak",
                trace=edge_trace,
            )
        learned_repeated = b.learned_synapses_snapshot(20)
        repeated_strength = max(
            edge["consolidation"]
            for edge in learned_repeated["edges"]
        )
        assert repeated_strength > first_strength

        strongest_before = max(
            abs(edge["learned_delta"])
            for edge in learned_repeated["edges"]
        )
        forgetting = b.consolidate_and_forget(
            elapsed_seconds=7 * 86400,
            force=True,
        )
        assert forgetting["ran"] is True
        assert forgetting["bias_factor"] < 1.0
        learned_after = b.learned_synapses_snapshot(20)
        assert learned_after["total"] >= 1
        strongest_after = max(
            abs(edge["learned_delta"])
            for edge in learned_after["edges"]
        )
        assert strongest_after < strongest_before

        policy_before = b.action_policy_diagnostics()
        react_bias_before = policy_before["actions"]["react"]["bias"]
        raw_react = b.action_scores()["react"]
        gate_before = b.action_policy_gate(
            "react",
            raw_react,
            0.62,
        )
        b.reward(
            1.0,
            action="react",
            trace=edge_trace,
        )
        policy_after = b.action_policy_diagnostics()
        react_bias_after = policy_after["actions"]["react"]["bias"]
        assert react_bias_after > react_bias_before
        assert (
            policy_after["actions"]["react"]["effective_score"]
            >= policy_after["actions"]["react"]["raw_score"]
        )
        gate_after = b.action_policy_gate(
            "react",
            b.action_scores()["react"],
            0.62,
        )
        assert (
            gate_after["learned_raw_threshold"]
            < gate_before["learned_raw_threshold"]
        )
        assert policy_after["last_update"]["action"] == "react"

        b.save()
        reloaded = FlyBrain(c, cfg)
        assert reloaded.diagnostics()["learned_synapses"] >= 1
        reloaded_learned = reloaded.learned_synapses_snapshot(20)
        assert reloaded_learned["edges"]
        assert max(
            edge["consolidation"]
            for edge in reloaded_learned["edges"]
        ) > 0.0
        reloaded_policy = reloaded.action_policy_diagnostics()
        assert reloaded_policy["actions"]["speak"]["updates"] > 0
        assert reloaded_policy["actions"]["speak"]["bias"] > 0.0

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
        generation_trace = lang.generation_trace()
        assert generation_trace["generator"] == "words"
        assert generation_trace["result"]
        assert generation_trace["attempts"]
        accepted_attempt = next(
            (
                item
                for item in generation_trace["attempts"]
                if item.get("accepted")
            ),
            generation_trace["attempts"][-1],
        )
        assert accepted_attempt["steps"]
        traced_step = next(
            (
                item
                for item in accepted_attempt["steps"]
                if item.get("phase") == "word-choice"
            ),
            accepted_attempt["steps"][0],
        )
        assert traced_step["candidates"]
        assert traced_step["selected"]
        if traced_step.get("phase") == "word-choice":
            assert 0.0 <= traced_step["selection_roll"] <= 1.0
            selected_rows = [
                item
                for item in traced_step["candidates"]
                if str(item.get("token"))
                == str(traced_step["selected"])
            ]
            assert selected_rows
            selected_row = selected_rows[0]
            assert "base_weight" in selected_row
            assert "final_weight" in selected_row
            assert "choice_share" in selected_row
            assert "brain_multiplier" in selected_row

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
