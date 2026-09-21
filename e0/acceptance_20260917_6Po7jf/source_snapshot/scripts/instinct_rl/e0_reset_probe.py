"""Bounded, real-simulation E0 reset probe; keeps the registered Flat configuration intact.

Only diagnostic state is changed: explicit partial resets, one episode counter near
timeout, and one robot placed sideways near the floor to exercise illegal contact.
"""

import argparse
import json
import math
import sys
import traceback
from pathlib import Path
from types import MethodType

from isaaclab.app import AppLauncher


def serializable(value):
    if hasattr(value, "detach"):
        return serializable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def probe(args, report):
    import gymnasium as gym
    import torch

    from isaaclab_tasks.utils import parse_env_cfg

    import instinctlab.tasks  # noqa: F401
    from instinctlab.utils.wrappers import InstinctRlVecEnvWrapper

    def check(name, passed, **details):
        report["checks"][name] = {"status": "PASS" if passed else "FAIL", **serializable(details)}
        print(f"[E0 RESET] {name}: {report['checks'][name]['status']}", flush=True)

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    cfg.seed = args.seed
    env = gym.make(args.task, cfg=cfg)
    raw = env.unwrapped
    original_reset, original_step = raw._reset_idx, raw.step
    try:
        wrapper = InstinctRlVecEnvWrapper(env)
        robot = raw.scene["robot"]
        term = raw.action_manager.get_term("joint_pos")
        chosen = torch.tensor([0, 2], device=raw.device)
        others = torch.tensor([1, 3], device=raw.device)
        stage, latest, finite_steps, merge_results = "warmup", {}, [], []
        report["reset_events"] = []

        def state():
            return {name: tensor.clone() for name, tensor in {
                "root_state": robot.data.root_state_w,
                "joint_pos": robot.data.joint_pos,
                "joint_vel": robot.data.joint_vel,
                "action": raw.action_manager.action,
                "previous_action": raw.action_manager.prev_action,
                "term_raw_action": term.raw_actions,
                "episode_length": raw.episode_length_buf,
            }.items()}

        def reset_hook(self, env_ids):
            before = state()
            critic = {name: value.clone() for name, value in
                      self.observation_manager.compute_group("critic", update_history=False).items()}
            contact = self.termination_manager.get_term_cfg("base_contact").params["sensor_cfg"]
            forces = self.scene[contact.name].data.net_forces_w_history[env_ids][:, :, contact.body_ids]
            record = {"stage": stage, "env_ids": env_ids.clone(), "before": before,
                      "critic_before_reset": critic,
                      "terminated_before_reset": self.termination_manager.terminated.clone(),
                      "truncated_before_reset": self.termination_manager.time_outs.clone(),
                      "illegal_body_max_contact_force": forces.norm(dim=-1).flatten(1).max(1).values}
            result = original_reset(env_ids)
            record["after"] = state()
            report["reset_events"].append(serializable(record))
            return result

        def step_hook(self, action):
            result = original_step(action)
            obs, reward, terminated, truncated, _ = result
            latest.update(terminated=terminated.clone(), truncated=truncated.clone())
            tensors = [action, reward]
            tensors += [tensor for group in obs.values() for tensor in group.values()]
            finite_steps.append(all(bool(torch.isfinite(value).all()) for value in tensors))
            return result

        raw._reset_idx = MethodType(reset_hook, raw)
        raw.step = MethodType(step_hook, raw)

        def step(action):
            result = wrapper.step(action)
            merged = (latest["terminated"] | latest["truncated"]).long()
            merge_results.append(bool(torch.equal(result[2], merged)))
            return result

        policy, extras = wrapper.get_observations()
        critic = extras["observations"]["critic"]
        report["runtime"] = serializable({
            "task": args.task, "seed": cfg.seed, "device": str(raw.device),
            "policy_shape": list(policy.shape), "critic_shape": list(critic.shape),
            "action_shape": [raw.num_envs, wrapper.num_actions],
            "observation_terms": wrapper.get_obs_format(), "num_actions": wrapper.num_actions,
            "resolved_joint_names": term._joint_names, "robot_joint_names": robot.joint_names,
            "action_scale": term._scale, "action_offset": term._offset,
            "action_clip": term.cfg.clip, "physics_dt": raw.physics_dt,
            "step_dt": raw.step_dt, "decimation": cfg.decimation,
            "max_episode_length": raw.max_episode_length, "is_finite_horizon": cfg.is_finite_horizon,
        })
        check("runtime_shapes", list(policy.shape) == [4, 96] and list(critic.shape) == [4, 99]
              and wrapper.num_actions == 29)
        check("control_timing", abs(raw.physics_dt - 0.005) < 1e-12
              and abs(raw.step_dt - 0.02) < 1e-12 and cfg.decimation == 4)
        action = torch.full((raw.num_envs, wrapper.num_actions), 0.02, device=raw.device)
        for _ in range(3):
            step(action)
        before = state()
        stage = "explicit_partial_reset"
        partial_obs, _ = raw.reset(env_ids=chosen)
        after = state()
        preserved = {name: bool(torch.equal(value[others], after[name][others]))
                     for name, value in before.items()}
        check("partial_reset_preserves_others", all(preserved.values()), env_ids=chosen,
              other_env_ids=others, preserved=preserved, before=before, after=after)
        initialized = {name: bool(torch.count_nonzero(after[name][chosen]) == 0)
                       for name in ("action", "previous_action", "term_raw_action", "episode_length")}
        initialized["policy_actions_observation"] = bool(
            torch.count_nonzero(partial_obs["policy"]["actions"][chosen]) == 0)
        check("partial_reset_initialization", all(initialized.values()), initialized=initialized)
        stage = "after_partial_reset"
        counter = raw.common_step_counter
        step(action)
        counts = raw.episode_length_buf.clone()
        root_delta = (robot.data.root_state_w[others] - after["root_state"][others]).abs().amax(dim=1)
        progressed = (counts[others] == after["episode_length"][others] + 1) | (
            latest["terminated"][others] | latest["truncated"][others])
        check("other_environments_continue", bool(progressed.all()) and bool((root_delta > 0).all())
              and raw.common_step_counter == counter + 1,
              episode_before=after["episode_length"], episode_after=counts,
              root_state_max_change=root_delta, terminated=latest["terminated"], truncated=latest["truncated"])

        stage = "prepare_timeout"
        raw.reset()
        stage = "artificial_timeout"
        raw.episode_length_buf[0] = raw.max_episode_length - 1
        event_start = len(report["reset_events"])
        policy, reward, done, extras = step(action)
        timeout_flags = {**latest, "wrapper_done": done.clone(),
                         "wrapper_time_outs": extras.get("time_outs"),
                         "episode_after": raw.episode_length_buf.clone()}
        check("timeout_branch", bool(latest["truncated"][0]) and bool(done[0]),
              construction="episode_length_buf[0] = max_episode_length - 1 before one real step",
              flags=timeout_flags)
        check("timeout_wrapper_extra", cfg.is_finite_horizon or torch.equal(
            extras["time_outs"], latest["truncated"]), flags=timeout_flags)
        timeout_events = report["reset_events"][event_start:]
        event = next((item for item in timeout_events if 0 in item["env_ids"]), None)
        post_actions = extras["observations"]["critic"][:, -wrapper.num_actions:]
        pre_count = event["before"]["episode_length"][0] if event else None
        pre_action = event["before"]["action"][0] if event else []
        check("terminal_state_vs_new_observation", event is not None
              and pre_count == raw.max_episode_length and raw.episode_length_buf[0].item() == 0
              and any(value != 0 for value in pre_action) and bool(torch.count_nonzero(post_actions[0]) == 0),
              before_episode_length=pre_count, returned_critic=extras["observations"]["critic"],
              returned_policy=policy, returned_reward=reward, flags=timeout_flags,
              note="reset hook captures terminal state; env.step returns new-episode observations")

        stage = "prepare_contact"
        raw.reset()
        stage = "artificial_physical_contact"
        contact_id = torch.tensor([1], device=raw.device)
        pose = robot.data.root_state_w[contact_id, :7].clone()
        pose[:, 2] = raw.scene.env_origins[contact_id, 2] + 0.10
        pose[:, 3:7] = torch.tensor([0.70710678, 0.0, 0.70710678, 0.0], device=raw.device)
        robot.write_root_pose_to_sim(pose, env_ids=contact_id)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=raw.device), env_ids=contact_id)
        raw.sim.forward()
        contact_result = None
        for index in range(args.contact_steps):
            _, _, done, _ = step(torch.zeros_like(action))
            if bool(latest["terminated"][1]):
                contact_result = serializable({"steps": index + 1, **latest, "wrapper_done": done})
                break
        report["checks"]["physical_contact_termination"] = {
            "status": "PASS" if contact_result else "BLOCKED",
            "construction": "env 1 root placed 0.10 m above floor, +90 degrees about world y, zero velocity",
            "result": contact_result, "step_limit": args.contact_steps,
            "note": "Original illegal_contact term used; no termination/reward/actuator configuration changed",
        }
        check("raw_flags_equal_wrapper_done", all(merge_results), actual_steps=len(merge_results))
        check("simulation_step_finite", all(finite_steps), actual_steps=len(finite_steps))
    finally:
        raw._reset_idx, raw.step = original_reset, original_step
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Instinct-Locomotion-Flat-G1-v0")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--contact-steps", type=int, default=200)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.num_envs != 4 or not 1 <= args.contact_steps <= 200:
        parser.error("E0 probe requires num_envs=4 and 1 <= contact-steps <= 200")
    output = Path(args.output_dir).resolve() / "reset_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic results: {output}")
    report = {"argv": sys.argv, "checks": {}, "diagnostic_only": True}
    simulation_app = None
    try:
        simulation_app = AppLauncher(args).app
        probe(args, report)
    except Exception:
        report["checks"]["execution"] = {"status": "FAIL", "traceback": traceback.format_exc()}
        traceback.print_exc()
    finally:
        statuses = [item["status"] for item in report["checks"].values()]
        report["status"] = "FAIL" if "FAIL" in statuses else "BLOCKED" if "BLOCKED" in statuses else "PASS"
        output.write_text(json.dumps(serializable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"[E0 RESET] {report['status']}: {output}", flush=True)
        if simulation_app is not None:
            simulation_app.close()
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
