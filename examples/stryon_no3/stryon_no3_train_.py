import argparse
import os
import pickle
import shutil
import torch
import genesis as gs
from stryon_no3_env import StryonNo3Env
from rsl_rl.runners import OnPolicyRunner


def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 1e-4,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "num_steps_per_env": 24,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
            "save_interval": 100,
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
        "empirical_normalization": None,
    }
    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 6,
        "mjcf": "./genesis/assets/xml/stryon_no3/stryon_no3.xml",
        # Joint names
        "default_joint_angles": {  # [rad]
            "left_front_knee_joint": 0.0,
            "left_back_knee_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_front_knee_joint": 0.0,
            "right_back_knee_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
        "joint_init_angles": {  # [rad]
            "left_front_knee_joint": 0.0,
            "left_back_knee_joint": 0.0,
            "left_wheel_joint": 0.0,
            "right_front_knee_joint": 0.0,
            "right_back_knee_joint": 0.0,
            "right_wheel_joint": 0.0,
        },
        "joint_names": [
            "left_front_knee_joint",
            "left_back_knee_joint",
            "left_wheel_joint",
            "right_front_knee_joint",
            "right_back_knee_joint",
            "right_wheel_joint",
        ],
        "joint_type": {  # joint/wheel
            "left_front_knee_joint": "joint",
            "left_back_knee_joint": "joint",
            "left_wheel_joint": "wheel",
            "right_front_knee_joint": "joint",
            "right_back_knee_joint": "joint",
            "right_wheel_joint": "wheel",
        },
        # lower upper
        "dof_limit": {
            "left_front_knee_joint": [-0.785398, 0.872665],
            "left_back_knee_joint": [-0.785398, 0.872665],
            "left_wheel_joint": [0.0, 0.0],
            "right_front_knee_joint": [-0.785398, 0.872665],
            "right_back_knee_joint": [-0.785398, 0.872665],
            "right_wheel_joint": [0.0, 0.0],
        },
        "safe_force": {
            "left_front_knee_joint": 3.41,
            "left_back_knee_joint": 3.41,
            "left_wheel_joint": 2.7,
            "right_front_knee_joint": 3.41,
            "right_back_knee_joint": 3.41,
            "right_wheel_joint": 2.7,
        },
        # PD
        "joint_kp": 3.0,  # TODO: Adjust KP
        "joint_kv": 0.8,  # TODO: Adjust KV
        "wheel_kv": 0.07,
        "damping": 0.01,
        "armature": 0.002,
        # Termination(degrees)
        "termination_if_roll_greater_than": 25,
        "termination_if_pitch_greater_than": 25,
        "termination_if_base_connect_plane_than": True,
        "connect_plane_links": [
            "body_link",
            "left_front_knee",
            "left_back_knee",
            "right_front_knee",
            "right_back_knee",
        ],
        "foot_link": [
            "left_wheel",
            "right_wheel",
        ],
        # base pose
        "base_init_pos": [0.0, 0.0, 0.2],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 3.0,
        "joint_action_scale": 0.5,
        "wheel_action_scale": 10.0,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "convexify": True,
        "decimate_aggressiveness": 4,
    }
    obs_cfg = {
        # num_obs = num_slice_obs + history_length * num_slice_obs
        "num_obs": 230,  # 23 + 9 * 23
        "num_slice_obs": 23,
        "history_length": 9,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 2.2,
            "base_euler": 1.0,
            "dof_pos": 1.0,
        },
        "noise": {
            "use": True,
            "ang_vel": [0.01, 0.01],
            "dof_pos": [0.01, 0.01],
            "dof_vel": [0.01, 0.01],
            "gravity": [0.01, 0.01],
            "base_euler": [0.25, 0.25],
        },
    }
    reward_cfg = {
        "tracking_linx_sigma": 0.5,
        "tracking_linx_alpha": 0.5,
        "tracking_ang_sigma": 0.6,
        "tracking_ang_alpha": 0.5,
        "tracking_grav_sigma": 0.05,
        "target_roll": 0.0,
        "target_pitch": -3.271,
        "tracking_base_euler_sigma": 3.0,
        "reward_scales": {
            "tracking_lin_x_vel": 1.0,
            "tracking_ang_vel": 1.1,
            "tracking_base_euler": 0.3,
            "tracking_leg_length": -3.0,
            "lin_vel_z": -0.02,
            "joint_action_rate": -0.01,
            "wheel_action_rate": -0.01,
            "dof_acc": -1e-6,
            "dof_force": -1e-6,
            "ang_vel_xy": -0.02,
            "collision": -0.0003,
            "survive": 1.0,
        },
    }
    command_cfg = {
        "num_commands": 4,
        "lin_vel_x_range": [-1.0, 1.0],
        "ang_vel_range": [-3.14, 3.14],
        "leg_length_range": [0.0, 0.4],
        "zero_stable": True,
    }
    curriculum_cfg = {
        "curriculum_step": 25,
        "curriculum_lin_vel_step": 0.003,
        "curriculum_ang_vel_step": 0.005,
        "curriculum_lin_vel_min_range": 0.01,
        "curriculum_ang_vel_min_range": 0.015,
        "err_mode": True,
        "lin_vel_err_range": [0.09, 0.15, 0.5],
        "ang_vel_err_range": [0.23, 0.4, 1.0],
    }
    domain_rand_cfg = {
        # TODO: Domain randomization (Not implemented)
    }
    terrain_cfg = {
        "terrain": True,
        "textures": "./genesis/assets/textures/stryon_no3_train.png",
        "respawn_points": [
            [4.2, 5.25, 0.0],
            [12.6, 5.25, 0.0],
        ],
        "horizontal_scale": 0.1,
        "vertical_scale": 0.001,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="stryon_no3_train")
    parser.add_argument("-B", "--num_envs", type=int, default=8192)
    parser.add_argument("--max_iterations", type=int, default=15000)
    args = parser.parse_args()

    gs.init(logging_level="warning", backend=gs.gpu)

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    env = StryonNo3Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg=curriculum_cfg,
        domain_rand_cfg=domain_rand_cfg,
        terrain_cfg=terrain_cfg,
        show_viewer=True,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
