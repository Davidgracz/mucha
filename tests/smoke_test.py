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
from mucha.voice_sensory import VoiceSensoryBus
from mucha.web_ui import AFFINITY_HTML, ASSOCIATIONS_HTML, AUTONOMY_HTML, CONFIG_HTML, CONNECTOME_HTML, NEUROMAP_HTML, HTML, OVERVIEW_HTML, PUBLIC_OVERVIEW_HTML, WebDashboard


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

        speak_vs_stay = b.action_competition(
            ("speak", "stay")
        )
        assert speak_vs_stay["action"] in {"speak", "stay"}
        assert set(speak_vs_stay["candidates"]) == {
            "speak",
            "stay",
        }
        assert "raw_candidates" in speak_vs_stay
        assert "policy_scores" in speak_vs_stay
        react_vs_stay = b.action_competition(
            ("react", "stay")
        )
        assert react_vs_stay["action"] in {"react", "stay"}
        assert set(react_vs_stay["candidates"]) == {
            "react",
            "stay",
        }

        target_ctx_a = b.inject_voice_target_context(
            1,
            111,
            novelty=0.9,
            recent=0.1,
            uncertainty=0.7,
            reward_opportunity_strength=0.8,
            is_current=False,
        )
        target_ctx_b = b.inject_voice_target_context(
            1,
            222,
            novelty=0.2,
            recent=0.8,
            uncertainty=0.1,
            reward_opportunity_strength=0.0,
            is_current=False,
        )
        assert target_ctx_a["direct_target_bonus"] is False
        assert target_ctx_a["cue_count"] > 0
        assert target_ctx_b["cue_count"] > 0
        b.inject_voice_snapshot(1, 111, [11, 22])
        b.inject_voice_snapshot(1, 222, [33])
        b.step(2)
        target_choice = b.voice_channel_target_decision(
            1,
            [111, 222],
        )
        assert target_choice["channel_id"] in {111, 222}
        assert set(target_choice["candidates"]) == {
            "111",
            "222",
        }
        assert target_choice["source"] == (
            "neural-channel-readout"
        )
        assert target_choice["direct_target_bonus"] is False

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

        drives_before = b.internal_drive_diagnostics()
        assert drives_before["enabled"] is True
        assert set(drives_before["drives"]) == {
            "social_need",
            "curiosity",
            "exploration",
            "caution",
            "boredom",
        }
        social_before = float(
            drives_before["drives"]["social_need"]["value"]
        )
        boredom_before = float(
            drives_before["drives"]["boredom"]["value"]
        )
        drives_idle = b.tick_internal_drives(
            60.0,
            external_stimulation=False,
            social_contact=False,
        )
        assert (
            drives_idle["drives"]["social_need"]["value"]
            > social_before
        )
        assert (
            drives_idle["drives"]["boredom"]["value"]
            > boredom_before
        )
        contact_before = float(
            drives_idle["drives"]["social_need"]["value"]
        )
        contact_event = b.register_internal_drive_event(
            "social_contact",
            1.0,
            inject=False,
        )
        assert (
            contact_event["values"]["social_need"]
            < contact_before
        )
        caution_before = float(
            contact_event["values"]["caution"]
        )
        threat_event = b.register_internal_drive_event(
            "threat",
            1.0,
            inject=False,
        )
        assert threat_event["values"]["caution"] > caution_before
        assert (
            b.internal_state_diagnostics()["homeostatic_drives"][
                "dominant"
            ]
            in b.INTERNAL_DRIVE_NAMES
        )
        social_map = dict(
            b.INTERNAL_DRIVE_STATE_MAP["social_need"]
        )
        boredom_map = dict(
            b.INTERNAL_DRIVE_STATE_MAP["boredom"]
        )
        assert social_map.get("arousal", 0.0) > 0.0
        assert boredom_map.get("arousal", 0.0) > 0.0
        assert "speak" in b.INTERNAL_STATE_TARGET_ACTIONS["arousal"]

        one_brain_set = b.one_brain_candidate_set(
            {
                "stay": True,
                "speak": True,
                "react": False,
            },
            technical_reasons={
                "stay": "always-available-noop",
                "speak": "text-target-and-language-ready",
                "react": "reaction-cooldown",
            },
        )
        assert set(one_brain_set["candidate_actions"]) == {
            "stay",
            "speak",
        }
        assert one_brain_set["rows"]["react"]["feasible"] is False
        assert one_brain_set["rows"]["react"][
            "technical_reason"
        ] == "reaction-cooldown"
        one_brain_decision = b.one_brain_action_decision(
            one_brain_set,
            decision_context="smoke:text-event",
            predicted_reward_gain=0.85,
            propagation_steps=2,
        )
        assert one_brain_decision["action"] in {
            "stay",
            "speak",
        }
        assert one_brain_decision["decision_context"] == (
            "smoke:text-event"
        )
        assert one_brain_decision["propagation_steps"] == 2
        assert one_brain_decision["executed"] is False
        assert one_brain_decision["source"].startswith(
            "one-brain predicted-reward sensory guidance"
        )

        unobserved_reward = b.action_reward_prediction("speak")
        assert unobserved_reward["predicted_reward"] == 0.0
        assert unobserved_reward["confidence"] == 0.0
        assert (
            unobserved_reward["source"]
            == "unobserved-neutral-prior"
        )

        autonomous_outside = b.autonomous_action_candidates(
            can_speak=True,
            connected_voice=False,
            voice_target_count=2,
            can_explore=True,
        )
        assert autonomous_outside["executed"] is False
        assert set(autonomous_outside["candidate_actions"]) == {
            "stay",
            "speak",
            "voice_join",
            "explore",
        }
        assert "noop" in autonomous_outside["display_candidates"]
        assert "voice_move" not in autonomous_outside[
            "candidate_actions"
        ]
        assert autonomous_outside["winner_preview"] in (
            autonomous_outside["candidate_actions"]
        )
        assert autonomous_outside["rows"]["speak"]["feasible"] is True
        assert (
            autonomous_outside["rows"]["speak"]["drive_support"]
            >= 0.0
        )
        assert "predicted_reward_order" in autonomous_outside
        assert (
            autonomous_outside["predicted_reward_winner"]
            in autonomous_outside["candidate_actions"]
        )
        assert (
            autonomous_outside["rows"]["speak"][
                "prediction_source"
            ]
            == "unobserved-neutral-prior"
        )

        autonomous_rewarded = b.autonomous_action_candidates(
            can_speak=True,
            connected_voice=False,
            voice_target_count=2,
            can_explore=True,
            contextual_reward_predictions={
                "voice_join": {
                    "expected_reward": 0.75,
                    "observations": 8,
                    "confidence": 0.85,
                },
            },
        )
        assert autonomous_rewarded["rows"]["voice_join"][
            "predicted_reward"
        ] > 0.0
        assert autonomous_rewarded["rows"]["voice_join"][
            "prediction_confidence"
        ] > 0.0
        assert autonomous_rewarded["rows"]["voice_join"][
            "prediction_source"
        ] == "episodic-context"
        assert (
            autonomous_rewarded["predicted_reward_winner"]
            == "voice_join"
        )
        assert autonomous_rewarded["prediction_executed"] is False
        autonomous_decision = b.autonomous_action_decision(
            autonomous_rewarded,
            predicted_reward_gain=0.85,
            propagation_steps=2,
        )
        assert autonomous_decision["action"] in (
            autonomous_rewarded["candidate_actions"]
        )
        assert autonomous_decision["executed"] is False
        assert autonomous_decision["propagation_steps"] == 2
        assert "voice_join" in autonomous_decision["prediction_cues"]
        assert (
            autonomous_decision["source"]
            == "one-brain predicted-reward sensory guidance -> "
            "FAFB propagation -> connectome action competition"
        )
        assert autonomous_decision["decision_context"] == "autonomous-idle"

        autonomous_inside = b.autonomous_action_candidates(
            can_speak=False,
            connected_voice=True,
            voice_target_count=1,
            can_explore=True,
        )
        assert set(autonomous_inside["candidate_actions"]) == {
            "stay",
            "voice_move",
            "explore",
        }
        assert "voice_join" not in autonomous_inside[
            "candidate_actions"
        ]
        assert "speak" not in autonomous_inside["candidate_actions"]
        assert autonomous_inside["rows"]["stay"]["feasible"] is True
        assert autonomous_inside["rows"]["speak"]["feasible"] is False

        autonomous_refractory = b.autonomous_action_candidates(
            can_speak=False,
            connected_voice=True,
            voice_target_count=1,
            can_explore=False,
            can_voice_move=False,
        )
        assert set(autonomous_refractory["candidate_actions"]) == {
            "stay",
        }
        assert autonomous_refractory["rows"]["voice_move"][
            "technical_reason"
        ] == "motor-refractory"

        autonomous_no_voice = b.autonomous_action_candidates(
            can_speak=True,
            connected_voice=False,
            voice_target_count=0,
            can_explore=True,
        )
        assert set(autonomous_no_voice["candidate_actions"]) == {
            "stay",
            "speak",
            "explore",
        }
        assert autonomous_no_voice["rows"]["voice_join"][
            "technical_reason"
        ] == "no-voice-target"

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
        uncertainty_before = episodes.semantic_uncertainty(
            "ctx",
            ["voice_join"],
        )
        assert uncertainty_before["uncertainty"] == 1.0
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
        prediction_detail = episodes.prediction_details(
            "ctx",
            "voice_join",
        )
        assert abs(
            prediction_detail["expected_reward"] - 0.3
        ) < 1e-9
        assert prediction_detail["observations"] == 1
        assert prediction_detail["confidence"] > 0.0
        detailed_predictions = episodes.predictions_detailed(
            "ctx",
            ["voice_join", "stay"],
        )
        assert detailed_predictions["voice_join"][
            "observations"
        ] == 1
        assert detailed_predictions["stay"]["observations"] == 0
        uncertainty_after = episodes.semantic_uncertainty(
            "ctx",
            ["voice_join"],
        )
        assert uncertainty_after["uncertainty"] < uncertainty_before["uncertainty"]
        assert ep["semantic_uncertainty_before"] > ep["semantic_uncertainty_after"]
        assert ep["information_gain"] > 0.0
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
        semantic_before = restored.semantic_recall(
            "in|need=0|fatigue=1",
            ["voice_move"],
            channel_id=555,
            user_ids=[11, 22],
            min_observations=2,
        )
        assert semantic_before == {}
        restored.observe(
            guild_id=7,
            channel_id=555,
            channel_name="ASG",
            user_ids=[11, 22],
            user_names=["Dawid", "Stivi"],
            context="in|need=0|fatigue=1",
            scene_key=scene_key,
            action="voice_move",
            actual_reward=0.6,
            source="semantic smoke",
        )
        semantic = restored.semantic_recall(
            "in|need=0|fatigue=1",
            ["voice_move", "stay"],
            channel_id=555,
            user_ids=[11, 22],
            min_observations=2,
        )
        assert "voice_move" in semantic
        assert "stay" not in semantic
        assert semantic["voice_move"]["expected_reward"] > 0.0
        assert semantic["voice_move"]["confidence"] > 0.0
        assert semantic["voice_move"]["signal"] > 0.0
        assert semantic["voice_move"]["contributors"]
        semantic_top = restored.semantic_summary(8)
        assert semantic_top
        assert any(
            row["action"] == "voice_move"
            and row["observations"] >= 2
            for row in semantic_top
        )
        assert restored.diagnostics()["semantic_entries"] > 0

        restored.observe_person_contact(11, "text", now=time.time())
        restored.observe_person_contact(11, "text", now=time.time())
        restored.observe_person_contact(
            11,
            "voice_speech",
            now=time.time(),
        )
        restored.observe_person_social_event(
            11,
            "DIRECT_REPLY",
            0.35,
            now=time.time(),
        )
        restored.observe_person_social_event(
            11,
            "VOICE_REJECTION",
            -0.10,
            now=time.time(),
        )
        person = restored.person_profile(11)
        assert person["user_id"] == 11
        assert person["display_name"] == "Dawid"
        assert person["observations"] >= 7
        assert person["voice_observations"] >= 2
        assert person["contact_observations"] == 3
        assert person["social_observations"] == 2
        assert person["familiarity"] > 0.0
        assert person["confidence"] > 0.0
        assert person["preferred_action"]["action"] == "voice_move"
        assert any(
            row["source"] == "text"
            and row["observations"] == 2
            for row in person["contact_sources"]
        )
        assert any(
            row["event"] == "DIRECT_REPLY"
            and row["signal"] > 0.0
            for row in person["social_events"]
        )
        assert person["recent_episodes"]
        assert any(
            row["user_id"] == 11
            for row in restored.person_profiles(8)
        )

        restored.observe_channel_visit(
            555,
            "ASG",
            [11, 22],
            source="voice_visit",
            now=time.time(),
        )
        restored.observe_channel_visit(
            555,
            "ASG",
            [11],
            source="voice_visit",
            now=time.time(),
        )
        restored.observe_channel_dynamics(
            555,
            "ASG",
            conversation_mode="DIALOGUE",
            intensity=0.72,
            speech_ratio=0.64,
            human_count=2,
            speaker_user_id=11,
            now=time.time(),
        )
        restored.observe_channel_dynamics(
            555,
            "ASG",
            conversation_mode="DIALOGUE",
            intensity=0.58,
            speech_ratio=0.50,
            human_count=3,
            speaker_user_id=22,
            now=time.time(),
        )
        place = restored.channel_profile(555)
        assert place["channel_id"] == 555
        assert place["channel_name"] == "ASG"
        assert place["observations"] >= 4
        assert place["visit_observations"] >= 2
        assert place["dynamics_observations"] == 2
        assert place["familiarity"] > 0.0
        assert place["dominant_mode"] == "DIALOGUE"
        assert place["mean_intensity"] > 0.0
        assert place["mean_speech_ratio"] > 0.0
        assert place["mean_human_density"] > 0.0
        assert any(
            row["user_id"] == 11
            for row in place["people"]
        )
        assert place["recent_episodes"]
        assert any(
            row["channel_id"] == 555
            for row in restored.channel_profiles(8)
        )

        social_scene_key = restored.make_social_scene_key(
            channel_id=555,
            user_ids=[11, 22],
            context="in|need=0|fatigue=1|hab=0|explore=0|humans=2|alts=1",
            conversation_mode="DIALOGUE",
            intensity=0.72,
            speech_ratio=0.64,
            human_count=2,
            dominant_state="social_need",
            dominant_state_level=0.55,
        )
        restored.observe_social_scene_contact(
            social_scene_key,
            now=time.time(),
        )
        restored.observe_social_scene_contact(
            social_scene_key,
            now=time.time(),
        )
        restored.observe_social_scene_outcome(
            social_scene_key,
            "stay",
            0.65,
            now=time.time(),
        )
        restored.observe_social_scene_outcome(
            social_scene_key,
            "voice_leave",
            -0.45,
            now=time.time(),
        )
        social_scene = restored.social_scene_profile(
            social_scene_key
        )
        assert social_scene["channel_id"] == 555
        assert social_scene["user_ids"] == [11, 22]
        assert social_scene["conversation_mode"] == "DIALOGUE"
        assert social_scene["human_count"] == 2
        assert social_scene["dominant_state"] == "social_need"
        assert social_scene["seen_observations"] == 2
        assert social_scene["outcome_observations"] == 2
        assert social_scene["familiarity"] > 0.0
        assert social_scene["preferred_action"]["action"] == "stay"
        assert social_scene["avoided_action"]["action"] == "voice_leave"
        assert any(
            row["scene_key"] == social_scene_key
            for row in restored.social_scene_profiles(8)
        )

        dynamics_snapshot = {
            "conversation_mode": "DIALOGUE",
            "conversation_intensity": 0.72,
            "speech_ratio_60s": 0.64,
            "speaker_switches_60s": 5,
            "overlap_events_60s": 1,
            "mean_handoff_seconds": 0.62,
            "mean_turn_seconds": 3.2,
            "speaker_dominance": 0.58,
            "silence_seconds": 0.4,
            "last_transcript": {
                "words_per_second": 2.1,
            },
        }
        dynamics_key = restored.make_voice_dynamics_key(
            dynamics_snapshot
        )
        assert "mode=DIALOGUE" in dynamics_key
        restored.observe_voice_dynamics_contact(
            dynamics_key,
            now=time.time(),
        )
        restored.observe_voice_dynamics_contact(
            dynamics_key,
            now=time.time(),
        )
        restored.observe_voice_dynamics_outcome(
            dynamics_key,
            "stay",
            0.55,
            now=time.time(),
        )
        restored.observe_voice_dynamics_outcome(
            dynamics_key,
            "voice_move",
            -0.40,
            now=time.time(),
        )
        restored.observe_voice_dynamics_outcome(
            dynamics_key,
            "speak",
            0.30,
            now=time.time(),
        )
        dynamics_profile = restored.voice_dynamics_profile(
            dynamics_key
        )
        assert dynamics_profile["conversation_mode"] == "DIALOGUE"
        assert dynamics_profile["seen_observations"] == 2
        assert dynamics_profile["outcome_observations"] == 3
        assert dynamics_profile["switch_bucket"] == 2
        assert dynamics_profile["overlap_bucket"] == 1
        assert dynamics_profile["handoff_bucket"] == "normal"
        assert dynamics_profile["turn_bucket"] == "medium"
        assert dynamics_profile["silence_bucket"] == "active"
        assert dynamics_profile["speech_rate_bucket"] == "normal"
        assert dynamics_profile["preferred_action"]["action"] == "stay"
        assert dynamics_profile["avoided_action"]["action"] == "voice_move"
        assert any(
            row["action"] == "speak"
            and row["signal"] > 0.0
            for row in dynamics_profile["actions"]
        )
        assert any(
            row["dynamics_key"] == dynamics_key
            for row in restored.voice_dynamics_profiles(8)
        )
        dynamics_brain = b.inject_voice_dynamics_profile(
            dynamics_profile,
            magnitude=0.38,
        )
        assert dynamics_brain["enabled"] is True
        assert dynamics_brain["cue_count"] > 0
        assert dynamics_brain["direct_action_bias"] is False
        assert any(
            row["key"].startswith(
                "voice:dynamics-history:action:"
            )
            for row in dynamics_brain["cues"]
        )

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
        semantic_before_sleep = restored.semantic_recall(
            "in|need=0|fatigue=1",
            ["voice_move"],
            channel_id=555,
            user_ids=[11, 22],
            min_observations=2,
        )["voice_move"]
        observations_before_sleep = int(
            semantic_before_sleep["observations"]
        )
        rehearsed_semantics = restored.rehearse_semantic_replay(
            replay_candidates[0],
        )
        assert rehearsed_semantics
        assert any(
            abs(float(row["delta"])) > 0.0
            for row in rehearsed_semantics
        )
        semantic_after_sleep = restored.semantic_recall(
            "in|need=0|fatigue=1",
            ["voice_move"],
            channel_id=555,
            user_ids=[11, 22],
            min_observations=2,
        )["voice_move"]
        assert int(
            semantic_after_sleep["observations"]
        ) == observations_before_sleep
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
        # Live Voice Sensory Bus measures the Discord scene without
        # producing hand-authored action scores.
        sensory_bus = VoiceSensoryBus(
            speaker_timeout_seconds=0.5,
            recent_window_seconds=60.0,
            reply_window_seconds=10.0,
        )
        sensory_bus.note_pcm(
            7,
            555,
            11,
            "Dawid",
            now=100.00,
        )
        sensory_bus.note_pcm(
            7,
            555,
            11,
            "Dawid",
            now=100.10,
        )
        first_sensory = sensory_bus.snapshot(
            7,
            connected=True,
            channel_id=555,
            channel_name="ASG",
            current_members=[
                {"id": 11, "name": "Dawid", "affinity": 0.42},
                {"id": 22, "name": "Stivi", "affinity": 0.18},
            ],
            other_members=[
                {"id": 33, "name": "Mamba", "affinity": 0.30},
            ],
            familiar_threshold=0.10,
            avoid_threshold=-0.35,
            now=100.20,
        )
        assert first_sensory["speaker_count"] == 1
        assert first_sensory["speakers"][0]["id"] == 11
        assert first_sensory["other_voice_humans"] == 1
        assert first_sensory["other_familiar_humans"] == 1
        assert first_sensory["unique_speakers_60s"] == 1
        assert first_sensory["speech_ratio_60s"] > 0.0
        assert first_sensory["conversation_mode"] in {
            "CONVERSATION",
            "MONOLOGUE",
        }

        sensory_bus.note_tts(
            7,
            555,
            "testowa odpowiedź",
            now=100.21,
        )
        pending_tts = sensory_bus.snapshot(
            7,
            connected=True,
            channel_id=555,
            channel_name="ASG",
            current_members=[
                {"id": 11, "name": "Dawid", "affinity": 0.42},
                {"id": 22, "name": "Stivi", "affinity": 0.18},
            ],
            other_members=[],
            now=100.22,
        )
        assert pending_tts["tts_pending_reply"] is True
        assert pending_tts["tts_age_seconds"] is not None

        sensory_bus.note_pcm(
            7,
            555,
            22,
            "Stivi",
            now=100.25,
        )
        sensory_bus.note_pcm(
            7,
            555,
            22,
            "Stivi",
            now=100.30,
        )
        sensory_bus.note_reply_after_tts(
            7,
            555,
            11,
            "Dawid",
            tts_age_seconds=2.2,
            now=100.30,
        )
        overlap_sensory = sensory_bus.snapshot(
            7,
            connected=True,
            channel_id=555,
            channel_name="ASG",
            current_members=[
                {"id": 11, "name": "Dawid", "affinity": 0.42},
                {"id": 22, "name": "Stivi", "affinity": 0.18},
            ],
            other_members=[],
            familiar_threshold=0.10,
            avoid_threshold=-0.35,
            now=100.35,
        )
        assert overlap_sensory["speaker_count"] == 2
        assert overlap_sensory["overlap_count"] == 1
        assert overlap_sensory["reply_after_tts"] is True
        assert overlap_sensory["reply_user_id"] == 11
        assert overlap_sensory["tts_pending_reply"] is False
        assert overlap_sensory["unique_speakers_60s"] == 2
        assert overlap_sensory["speaker_switches_60s"] >= 1
        assert overlap_sensory["overlap_events_60s"] >= 1
        assert overlap_sensory["conversation_intensity"] > 0.0
        assert 0.0 <= overlap_sensory["speaker_dominance"] <= 1.0

        state_before_voice_sensory = b.compute.to_cpu(
            b.state
        ).copy()
        voice_sensory_diag = b.inject_voice_sensory_bus(
            7,
            overlap_sensory,
            base_magnitude=0.55,
        )
        assert voice_sensory_diag["mode"] == "raw-sensory-only"
        assert voice_sensory_diag["action_guided"] is False
        assert voice_sensory_diag["direct_action_bias"] is False
        assert voice_sensory_diag["cue_count"] > 0
        assert all(
            row["key"].startswith("voice:sensory:")
            for row in voice_sensory_diag["cues"]
        )
        assert any(
            row["key"].startswith(
                "voice:sensory:conversation-mode:"
            )
            for row in voice_sensory_diag["cues"]
        )
        assert voice_sensory_diag["unique_speakers_60s"] == 2
        assert voice_sensory_diag["speaker_switches_60s"] >= 1
        state_after_voice_sensory = b.compute.to_cpu(
            b.state
        )
        assert np.max(
            np.abs(
                state_after_voice_sensory
                - state_before_voice_sensory
            )
        ) > 0.01

        quiet_sensory = sensory_bus.snapshot(
            7,
            connected=True,
            channel_id=555,
            channel_name="ASG",
            current_members=[
                {"id": 11, "name": "Dawid", "affinity": 0.42},
                {"id": 22, "name": "Stivi", "affinity": 0.18},
            ],
            other_members=[],
            familiar_threshold=0.10,
            avoid_threshold=-0.35,
            now=101.10,
        )
        assert quiet_sensory["speaker_count"] == 0
        assert quiet_sensory["overlap_count"] == 0
        assert quiet_sensory["silence_seconds"] > 0.5
        assert quiet_sensory["speech_seconds_60s"] > 0.0
        assert quiet_sensory["mean_turn_seconds"] > 0.0

        sensory_bus.note_pcm(
            7,
            555,
            11,
            "Dawid",
            now=102.00,
        )
        handoff_sensory = sensory_bus.snapshot(
            7,
            connected=True,
            channel_id=555,
            channel_name="ASG",
            current_members=[
                {"id": 11, "name": "Dawid", "affinity": 0.42},
                {"id": 22, "name": "Stivi", "affinity": 0.18},
            ],
            other_members=[],
            now=102.10,
        )
        assert handoff_sensory["handoff_count_60s"] >= 1
        assert handoff_sensory["mean_handoff_seconds"] is not None
        assert handoff_sensory["mean_handoff_seconds"] > 0.0

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

        state_before_semantic = b.compute.to_cpu(
            b.state
        ).copy()
        semantic_diag = b.inject_action_guided_signed_sensory(
            "voice_move",
            "semantic-smoke-negative",
            -0.55,
            width=96,
            hops=3,
        )
        assert semantic_diag["action"] == "voice_move"
        assert semantic_diag["signed"] is True
        assert semantic_diag["valence"] == "negative"
        assert semantic_diag["magnitude"] < 0.0
        assert semantic_diag["neurons"] > 0
        state_after_semantic = b.compute.to_cpu(
            b.state
        )
        assert np.max(
            np.abs(
                state_after_semantic
                - state_before_semantic
            )
        ) > 0.05

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
        assert voice_out["source"] == "connectome-readout+learned-policy"
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
        assert "person_model_enabled" in CONFIG_HTML
        assert "person_model_min_observations" in CONFIG_HTML
        assert "person_model_sensory_magnitude" in CONFIG_HTML
        assert "channel_model_enabled" in CONFIG_HTML
        assert "channel_model_min_observations" in CONFIG_HTML
        assert "channel_model_sensory_magnitude" in CONFIG_HTML
        assert "social_scene_model_enabled" in CONFIG_HTML
        assert "social_scene_min_observations" in CONFIG_HTML
        assert "social_scene_sensory_magnitude" in CONFIG_HTML
        assert "connectome_behavior_competition_enabled" in CONFIG_HTML
        assert "LEGACY: próg mówienia" in CONFIG_HTML
        assert "LEGACY: próg reakcji emoji" in CONFIG_HTML
        assert "voice_dynamics_learning_enabled" in CONFIG_HTML
        assert "voice_dynamics_min_observations" in CONFIG_HTML
        assert "voice_dynamics_sensory_magnitude" in CONFIG_HTML
        assert "voice_dynamics_seen_cooldown_seconds" in CONFIG_HTML
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
        assert "sleep_enabled" in CONFIG_HTML
        assert "sleep_idle_seconds" in CONFIG_HTML
        assert "sleep_cycle_interval_seconds" in CONFIG_HTML
        assert "sleep_max_cycles" in CONFIG_HTML
        assert "sleep_replay_batch_size" in CONFIG_HTML
        assert "sleep_replay_magnitude_multiplier" in CONFIG_HTML
        assert "sleep_reward_scale_multiplier" in CONFIG_HTML
        assert "sleep_steps_multiplier" in CONFIG_HTML
        assert "episodic_consolidation_gain" in CONFIG_HTML
        assert "episodic_forgetting_half_life_days" in CONFIG_HTML
        assert "semantic_memory_enabled" in CONFIG_HTML
        assert "semantic_recall_min_observations" in CONFIG_HTML
        assert "semantic_recall_magnitude" in CONFIG_HTML
        assert "semantic_recall_steps" in CONFIG_HTML
        assert "uncertainty_exploration_enabled" in CONFIG_HTML
        assert "uncertainty_curiosity_magnitude" in CONFIG_HTML
        assert "uncertainty_curiosity_steps" in CONFIG_HTML
        assert "uncertainty_target_weight" in CONFIG_HTML
        assert "information_gain_reward_scale" in CONFIG_HTML
        assert "information_gain_reward_max" in CONFIG_HTML
        assert "information_gain_min_delta" in CONFIG_HTML
        assert "Pamięć semantyczna" in HTML
        assert "Long-term People Memory" in HTML
        assert "renderPersonProfiles" in HTML
        assert 'id="people-memory-grid"' in HTML
        assert "person-memory" in HTML
        assert "Long-term Channel / Place Memory" in HTML
        assert "renderChannelProfiles" in HTML
        assert 'id="channel-memory-grid"' in HTML
        assert "channel-memory" in HTML
        assert "Long-term Social Situations" in HTML
        assert "renderSocialScenes" in HTML
        assert 'id="social-scene-grid"' in HTML
        assert "social-scene-memory" in HTML
        assert "Reward-learned Conversation Dynamics" in HTML
        assert "renderVoiceDynamicsProfiles" in HTML
        assert 'id="voice-dynamics-grid"' in HTML
        assert "voice-dynamics-memory" in HTML
        assert "Target selection" in HTML
        assert "Learning timing" in HTML
        assert "Target score" in HTML
        assert "Curiosity / Uncertainty" in HTML
        assert "Tryb rozmowy" in HTML
        assert "Speech ratio 60s" in HTML
        assert "Zmiany mówcy 60s" in HTML
        assert "Dynamika 60s" in HTML
        assert "CPU system" in HTML
        assert "GPU / VRAM" in HTML
        assert "/api/system" in HTML
        assert "const LIVE_REFRESH_MS=250;" in OVERVIEW_HTML
        assert "CPU system" in OVERVIEW_HTML
        assert "CPU Mucha" in OVERVIEW_HTML
        assert "RAM Mucha" in OVERVIEW_HTML
        assert "Temperatura GPU" in OVERVIEW_HTML
        assert "/api/system" in OVERVIEW_HTML
        assert "Information gain" in HTML
        assert "uncertainty_curiosity_cue" in HTML
        assert "semantic-memory" in HTML
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
        assert "Dlaczego zrobiła X?" in HTML
        assert "renderDecisionTrace" in HTML
        assert "Historia Decision Trace" in HTML
        assert "renderDecisionTraceHistory" in HTML
        assert "selectDecisionTrace" in HTML
        assert "setDecisionTraceFilter" in HTML
        assert 'id="trace-live-btn"' in HTML
        assert 'id="decision-trace-history"' in HTML
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
        assert "Sleep / Offline Consolidation" in HTML
        assert "renderSleep" in HTML
        assert 'id="sleep-progress"' in HTML
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
        assert "297724966571474944" in bot_source
        assert "HARD_BLOCKED_VOICE_USER_IDS" in bot_source
        assert "_voice_channel_has_hard_blocked_user" in bot_source
        assert (
            bot_source.count("_voice_channel_has_hard_blocked_user")
            >= 6
        )
        assert "_user_affinity_components" in bot_source
        assert "_write_neural_social_memory" in bot_source
        assert "voice_action_decision" in bot_source
        brain_source = (ROOT / "mucha" / "brain.py").read_text(
            encoding="utf-8"
        )
        config_source = (ROOT / "mucha" / "config.py").read_text(
            encoding="utf-8"
        )
        config_toml_source = (ROOT / "config.toml").read_text(
            encoding="utf-8"
        )
        for key in (
            "one_brain_enabled",
            "one_brain_predicted_reward_gain",
            "one_brain_prediction_steps",
            "autonomous_loop_enabled",
            "autonomous_predicted_reward_gain",
            "autonomous_prediction_steps",
            "autonomous_explore_cooldown_seconds",
        ):
            assert key in config_source
            assert key in config_toml_source
        assert "def action_competition" in brain_source
        assert "def tick_internal_drives" in brain_source
        assert "def register_internal_drive_event" in brain_source
        assert "def internal_drive_diagnostics" in brain_source
        assert "def autonomous_action_candidates" in brain_source
        assert "def action_reward_prediction" in brain_source
        assert "def one_brain_candidate_set" in brain_source
        assert "def one_brain_action_decision" in brain_source
        assert "one-brain generic feasibility + learned reward" in brain_source
        assert "def autonomous_action_decision" in brain_source
        assert "one-brain-predicted-reward:" in brain_source
        assert "predicted_reward_order" in brain_source
        assert "internal_drive_values" in brain_source
        assert "def voice_channel_target_decision" in brain_source
        assert "def inject_voice_target_context" in brain_source
        assert "tie_evidence" in bot_source
        assert "tie_break" in bot_source
        assert "action_policy_gate" in bot_source
        assert '"action_policy": action_policy_debug' in bot_source
        assert '"decision_trace": decision_trace' in bot_source
        assert '"decision_trace_history": decision_trace_history' in bot_source
        assert "_decision_trace_snapshot" in bot_source
        assert "_remember_decision_trace" in bot_source
        assert "_decision_trace_history_snapshot" in bot_source
        assert "deque(maxlen=48)" in bot_source
        assert "_text_decision_debug" in bot_source
        assert "_autonomous_candidate_contexts" in bot_source
        assert "_autonomous_candidate_debug" in bot_source
        assert "_autonomous_history" in bot_source
        assert "_one_brain_history" in bot_source
        assert "_one_brain_debug" in bot_source
        assert "_remember_one_brain_cycle" in bot_source
        assert '"one_brain": deepcopy(self._one_brain_debug)' in bot_source
        assert '"one_brain_history": deepcopy(' in bot_source
        assert "one-brain-deferred" in bot_source
        assert "defer_to_one_brain=bool(" in bot_source
        assert 'decision_context="autonomous-idle"' in brain_source
        assert "voice-tts:" in bot_source
        assert "_remember_autonomous_execution" in bot_source
        assert '"autonomous_history": deepcopy(' in bot_source
        assert "_execute_autonomous_action" in bot_source
        assert "_autonomous_voice_target" in bot_source
        assert "autonomous_loop_enabled" in bot_source
        assert '("behavior", "autonomous_loop_enabled")' in bot_source
        assert '("behavior", "autonomous_predicted_reward_gain")' in bot_source
        assert '("behavior", "autonomous_prediction_steps")' in bot_source
        assert (
            '("behavior", "autonomous_explore_cooldown_seconds")'
            in bot_source
        )
        assert "24D AUTONOMOUS LOOP" in bot_source
        assert "contextual_reward_predictions" in bot_source
        assert "predictions_detailed" in bot_source
        assert '"autonomous_candidates": deepcopy(' in bot_source
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
        assert "_voice_sensory.note_tts" in bot_source
        assert "_voice_prediction_context" in bot_source
        assert "_update_pending_voice_scene" in bot_source
        assert "episodic_recall_magnitude" in bot_source
        assert "semantic_recall" in bot_source
        assert "semantic_recall_magnitude" in bot_source
        assert "semantic_guided" in bot_source
        assert "semantic_uncertainty" in bot_source
        assert "uncertainty_curiosity_magnitude" in bot_source
        assert "internal-state:curiosity:" in bot_source
        assert "information_gain" in bot_source
        assert "uncertainty_target_weight" in bot_source
        assert "prediction_scene_key" in bot_source
        assert "prediction_error_max_correction" in bot_source
        assert "_queue_voice_prediction" in bot_source
        assert "prediction_credit_decay_seconds" in bot_source
        assert "_maybe_memory_replay" in bot_source
        assert "_sleep_tick" in bot_source
        assert "_note_external_activity" in bot_source
        assert "sleep-offline-consolidation" in bot_source
        assert '"sleep": dict(self._sleep_debug)' in bot_source
        assert "memory_replay_reward_scale" in bot_source
        assert "sleep_reward_scale_multiplier" in bot_source
        assert "replay_candidates" in bot_source
        episodic_source = (ROOT / "mucha" / "episodic.py").read_text(
            encoding="utf-8"
        )
        assert "rehearse_semantic_replay" in episodic_source
        assert "must not increase the observation count" in episodic_source
        assert "observe_person_contact" in episodic_source
        assert "observe_person_social_event" in episodic_source
        assert '"person_contact"' in episodic_source
        assert '"person_social"' in episodic_source
        assert "observe_channel_visit" in episodic_source
        assert "observe_channel_dynamics" in episodic_source
        assert "channel_profile" in episodic_source
        assert "make_social_scene_key" in episodic_source
        assert "observe_social_scene_contact" in episodic_source
        assert "observe_social_scene_outcome" in episodic_source
        assert "social_scene_profile" in episodic_source
        assert "make_voice_dynamics_key" in episodic_source
        assert "observe_voice_dynamics_contact" in episodic_source
        assert "observe_voice_dynamics_outcome" in episodic_source
        assert "voice_dynamics_profile" in episodic_source
        assert '"voice_dynamics_seen"' in episodic_source
        assert '"voice_dynamics"' in episodic_source
        assert '"channel_visit"' in episodic_source
        assert '"channel_people"' in episodic_source
        assert '"channel_mode"' in episodic_source
        assert "inject_person_profile" in bot_source
        assert "inject_channel_profile" in bot_source
        assert "inject_social_scene_profile" in bot_source
        assert "inject_voice_dynamics_profile" in bot_source
        assert "voice_dynamics_key" in bot_source
        assert "_voice_dynamics_model_debug" in bot_source
        assert "action_competition" in bot_source
        assert "def _behavior_gate" in bot_source
        assert "connectome_behavior_competition_enabled" in bot_source
        assert '"connectome-competition"' in bot_source
        assert "voice_channel_target_decision" in bot_source
        assert "inject_voice_target_context" in bot_source
        assert "learning_updates_do_not_override_current_decision" in bot_source
        assert '"neural-channel-readout"' in bot_source
        tts_loop_source = bot_source.split(
            "async def tts_loop",
            1,
        )[1].split(
            "@tts_loop.before_loop",
            1,
        )[0]
        assert 'action_competition(' in tts_loop_source
        assert (
            'scores["speak"] < self.cfg.behavior.speak_threshold'
            not in tts_loop_source
        )
        on_message_source = bot_source.split(
            "async def on_message",
            1,
        )[1].split(
            "async def on_raw_reaction_add",
            1,
        )[0]
        assert '_behavior_gate(' in on_message_source
        assert 'action_policy_gate(' not in on_message_source
        idle_source = bot_source.split(
            "async def idle_loop",
            1,
        )[1].split(
            "@idle_loop.before_loop",
            1,
        )[0]
        assert '_behavior_gate(' in idle_source
        assert 'spontaneous_gate = self.brain.action_policy_gate' not in idle_source
        choose_target_source = bot_source.split(
            "def _choose_voice_target",
            1,
        )[1].split(
            "def _reaction_candidates",
            1,
        )[0]
        assert "voice_channel_target_decision" in choose_target_source
        assert "legacy-affinity-exploration" in choose_target_source
        assert "observe_voice_dynamics_contact" in bot_source
        assert "observe_voice_dynamics_outcome" in bot_source
        assert "social_scene_key" in bot_source
        assert "observe_social_scene_contact" in bot_source
        assert "observe_social_scene_outcome" in bot_source
        assert "observe_person_contact" in bot_source
        assert "observe_person_social_event" in bot_source
        assert "observe_channel_visit" in bot_source
        assert "observe_channel_dynamics" in bot_source
        assert "_channel_model_debug" in bot_source
        assert "_attention_observe_text" in bot_source
        assert "_inject_attention_context" in bot_source
        assert "_attention_language_context" in bot_source
        assert '"attention": attention_debug' in bot_source
        web_ui_source = (ROOT / "mucha" / "web_ui.py").read_text(
            encoding="utf-8"
        )
        assert "_system_status" in web_ui_source
        assert "_safe_public_state" in web_ui_source
        safe_public_block = web_ui_source.split(
            "def _safe_public_state",
            1,
        )[1].split(
            "async def _public_state",
            1,
        )[0]
        assert "decision_trace_history" not in safe_public_block
        assert "decision_trace" not in safe_public_block
        assert "_gpu_monitor_loop" in web_ui_source
        assert "_query_gpu_status" in web_ui_source
        assert "nvidia-smi" in web_ui_source
        assert 'app.router.add_get("/api/system"' in web_ui_source
        requirements_source = (ROOT / "requirements.txt").read_text(
            encoding="utf-8"
        )
        assert "psutil" in requirements_source

        async def _empty_snapshot():
            return {}

        dashboard = WebDashboard(
            snapshot_provider=_empty_snapshot,
            host="127.0.0.1",
            port=8765,
            auto_open=False,
            refresh_ms=250,
        )
        system_diag = dashboard._system_status()
        assert system_diag["cpu_count"] >= 1
        assert 0.0 <= system_diag["cpu_percent"]
        assert system_diag["mem_total"] > 0
        assert system_diag["mem_used"] >= 0
        assert system_diag["disk_total"] > 0
        assert system_diag["disk_free"] >= 0
        assert system_diag["process"]["pid"] > 0
        assert system_diag["process"]["memory_bytes"] > 0
        assert "gpu" in system_diag

        brain_source = (ROOT / "mucha" / "brain.py").read_text(
            encoding="utf-8"
        )
        assert "_capture_signal_flow_tick" in brain_source
        assert "attention_score" in brain_source
        assert "inject_action_guided_signed_sensory" in brain_source
        assert "semantic-memory-sensory" in brain_source
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
        assert "Autonomia 24E" in AUTONOMY_HTML
        assert "One Brain 25" in AUTONOMY_HTML
        assert "One Brain timeline" in AUTONOMY_HTML
        assert "one_brain_history" in AUTONOMY_HTML
        assert "/api/state" in AUTONOMY_HTML
        assert "predicted reward" in AUTONOMY_HTML.lower()
        assert "prediction_cues" in AUTONOMY_HTML
        assert "autonomous_history" in AUTONOMY_HTML
        assert "NIE WYBRANO W TYM TICKU" in AUTONOMY_HTML
        assert 'href="/autonomy"' in OVERVIEW_HTML
        assert 'app.router.add_get("/autonomy"' in web_ui_source
        assert "def _autonomy_page" in web_ui_source
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
