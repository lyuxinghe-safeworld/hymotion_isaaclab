#!/usr/bin/env python3
"""Run ProtoMotions tracker on a converted .motion file in Isaac Lab.

Builds the env/agent inline (not via subprocess) so we can capture
viewport frames each step and compile them into an MP4 video.

Usage:
    python scripts/run_tracking.py \
        --motion-file output/00000001_000.motion \
        --num-envs 1

    # Record video (headless + frame capture → MP4):
    python scripts/run_tracking.py \
        --motion-file output/00000001_000.motion \
        --record --headless
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s: %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = (
    Path.home() / "code/ProtoMotions/data/pretrained_models/motion_tracker/smpl/last.ckpt"
)


# ---------------------------------------------------------------------------
# CLI  (parsed early — before heavy imports)
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ProtoMotions tracker on a .motion file in Isaac Lab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
        help="Path to tracker checkpoint",
    )
    parser.add_argument(
        "--motion-file", type=str, required=True,
        help="Path to .motion or .pt file from convert_npz.py",
    )
    parser.add_argument(
        "--num-envs", type=int, default=1,
        help="Number of environments",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without viewer",
    )
    parser.add_argument(
        "--simulator", type=str, default="isaaclab",
        choices=["isaacgym", "isaaclab", "newton"],
        help="Simulator backend",
    )
    # Recording options
    parser.add_argument(
        "--record", action="store_true",
        help="Capture viewport frames and compile to MP4",
    )
    parser.add_argument(
        "--video-output", type=str, default="",
        help="Output MP4 path (default: derived from motion-file name)",
    )
    parser.add_argument(
        "--clip-length", type=float, default=0,
        help="Max video clip length in seconds (0 = full motion duration)",
    )
    return parser.parse_args()


args = parse_args()

# Simulator import must happen before torch
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

import torch  # noqa: E402


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import atexit
    from dataclasses import asdict

    from lightning.fabric import Fabric

    from protomotions.utils.component_builder import build_all_components
    from protomotions.utils.fabric_config import FabricConfig
    from protomotions.utils.hydra_replacement import get_class
    from protomotions.utils.inference_utils import apply_backward_compatibility_fixes

    # ------------------------------------------------------------------
    # Validate paths
    # ------------------------------------------------------------------
    checkpoint = Path(args.checkpoint)
    motion_file = Path(args.motion_file).resolve()
    if not checkpoint.exists():
        log.error("Checkpoint not found: %s", checkpoint)
        sys.exit(1)
    if not motion_file.exists():
        log.error("Motion file not found: %s", motion_file)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Determine duration / max steps
    # ------------------------------------------------------------------
    motion_data = torch.load(str(motion_file), map_location="cpu", weights_only=False)
    motion_fps = motion_data.get("fps", 30)
    motion_frames = motion_data["rigid_body_pos"].shape[0]
    motion_duration_s = motion_frames / motion_fps

    sim_fps = 30
    clip_s = args.clip_length if args.clip_length > 0 else motion_duration_s
    clip_s = min(clip_s, motion_duration_s)
    max_steps = int(clip_s * sim_fps)

    log.info("=" * 60)
    log.info("Running tracker%s", " with recording" if args.record else "")
    log.info("  motion_file     : %s", motion_file)
    log.info("  motion_duration : %.1f s  (%d frames @ %d fps)",
             motion_duration_s, motion_frames, motion_fps)
    log.info("  clip_length     : %.1f s  (%d sim steps @ %d fps)",
             clip_s, max_steps, sim_fps)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # Video output paths (only when recording)
    # ------------------------------------------------------------------
    frames_dir = None
    video_path = None
    _video_compiled = False

    if args.record:
        if args.video_output:
            video_path = Path(args.video_output)
        else:
            video_path = motion_file.with_suffix(".mp4")
        video_path.parent.mkdir(parents=True, exist_ok=True)

        frames_dir = video_path.with_suffix("") / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        log.info("  frames_dir      : %s", frames_dir)
        log.info("  video_output    : %s", video_path)

    def compile_video():
        nonlocal _video_compiled
        if _video_compiled or not args.record or frames_dir is None:
            return
        _video_compiled = True
        pngs = sorted(frames_dir.glob("*.png"))
        if not pngs:
            log.warning("No frames captured — skipping video compilation.")
            return
        log.info("Compiling %d frames → %s ...", len(pngs), video_path)
        try:
            from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            clip = ImageSequenceClip([str(p) for p in pngs], fps=sim_fps)
            clip.write_videofile(
                str(video_path),
                codec="libx264",
                audio=False,
                threads=4,
                preset="veryfast",
                ffmpeg_params=[
                    "-profile:v", "main",
                    "-level", "4.0",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-crf", "23",
                ],
            )
            log.info("Video saved: %s", video_path)
            print(f"\nVideo saved to: {video_path.resolve()}\n")
        except Exception as e:
            log.error("Failed to compile video: %s", e)

    atexit.register(compile_video)

    # ------------------------------------------------------------------
    # Load tracker configs
    # ------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint)
    resolved_path = ckpt_path.parent / "resolved_configs_inference.pt"
    if not resolved_path.exists():
        log.error("resolved_configs_inference.pt not found at %s", resolved_path)
        sys.exit(1)

    configs = torch.load(str(resolved_path), map_location="cpu", weights_only=False)

    robot_config = configs["robot"]
    simulator_config = configs["simulator"]
    terrain_config = configs.get("terrain")
    scene_lib_config = configs["scene_lib"]
    motion_lib_config = configs["motion_lib"]
    env_config = configs["env"]
    agent_config = configs["agent"]

    # Make robot asset path absolute
    asset_root = getattr(robot_config.asset, "asset_root", "")
    if asset_root and not os.path.isabs(asset_root):
        import protomotions
        proto_root = Path(protomotions.__file__).parent.parent
        robot_config.asset.asset_root = str(proto_root / asset_root)

    # Switch simulator if needed
    current_sim = simulator_config._target_.split(".")[-3]
    if args.simulator != current_sim:
        from protomotions.simulator.factory import update_simulator_config_for_test
        simulator_config = update_simulator_config_for_test(
            current_simulator_config=simulator_config,
            new_simulator=args.simulator,
            robot_config=robot_config,
        )

    apply_backward_compatibility_fixes(robot_config, simulator_config, env_config)

    simulator_config.num_envs = args.num_envs
    simulator_config.headless = args.headless
    motion_lib_config.motion_file = str(motion_file)

    if hasattr(env_config, "max_episode_length"):
        env_config.max_episode_length = max(env_config.max_episode_length, max_steps + 100)

    # ------------------------------------------------------------------
    # Build components
    # ------------------------------------------------------------------
    fabric_config = FabricConfig(devices=1, num_nodes=1, loggers=[], callbacks=[])
    fabric = Fabric(**asdict(fabric_config))
    fabric.launch()
    device = fabric.device

    simulator_extra_params = {}
    if args.simulator == "isaaclab":
        app_launcher_flags = {"headless": args.headless, "device": str(device)}
        app_launcher = AppLauncher(app_launcher_flags)
        simulator_extra_params["simulation_app"] = app_launcher.app

    from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator
    terrain_config, simulator_config = convert_friction_for_simulator(
        terrain_config, simulator_config
    )

    components = build_all_components(
        terrain_config=terrain_config,
        scene_lib_config=scene_lib_config,
        motion_lib_config=motion_lib_config,
        simulator_config=simulator_config,
        robot_config=robot_config,
        device=device,
        **simulator_extra_params,
    )

    EnvClass = get_class(env_config._target_)
    env = EnvClass(
        config=env_config,
        robot_config=robot_config,
        device=device,
        terrain=components["terrain"],
        scene_lib=components["scene_lib"],
        motion_lib=components["motion_lib"],
        simulator=components["simulator"],
    )

    AgentClass = get_class(agent_config._target_)
    agent = AgentClass(
        config=agent_config, env=env, fabric=fabric,
        root_dir=ckpt_path.parent,
    )
    agent.setup()
    agent.load(str(args.checkpoint), load_env=False)

    simulator = components["simulator"]

    # ------------------------------------------------------------------
    # Simulation loop with optional frame capture
    # ------------------------------------------------------------------
    agent.eval()
    done_indices = None
    log.info("Starting simulation loop (%d steps) ...", max_steps)

    try:
        for step in range(max_steps):
            obs, _ = env.reset(done_indices)
            obs = agent.add_agent_info_to_obs(obs)
            obs_td = agent.obs_dict_to_tensordict(obs)

            model_outs = agent.model(obs_td)
            actions = model_outs.get("mean_action", model_outs.get("action"))

            obs, rewards, dones, terminated, extras = env.step(actions)
            obs = agent.add_agent_info_to_obs(obs)

            # Capture frame when recording
            if args.record and frames_dir is not None:
                frame_path = str(frames_dir / f"{step:06d}.png")
                simulator._write_viewport_to_file(frame_path)

            done_indices = dones.nonzero(as_tuple=False).squeeze(-1)

            if step % 100 == 0:
                log.info("  step %d / %d", step, max_steps)

    except KeyboardInterrupt:
        log.info("Interrupted at step %d", step)

    compile_video()
    log.info("Done.")


if __name__ == "__main__":
    main()
