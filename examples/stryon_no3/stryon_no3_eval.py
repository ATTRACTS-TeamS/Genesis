import argparse
import os
import pickle
import numpy as np
import torch
import copy
import sys
import pygame
import genesis as gs
from stryon_no3_env import StryonNo3Env
from rsl_rl.runners import OnPolicyRunner

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


def init_pygame():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("No joystick detected. Please connect your controller.")
        pygame.quit()
        sys.exit(1)
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick initialized: {joystick.get_name()}")
    return joystick


def get_joystick_commands(joystick):
    max_lin = 0.8
    max_ang = 3.0

    reset_flag = False
    pygame.event.pump()
    if joystick.get_button(1):
        reset_flag = True

    axis_left_y = joystick.get_axis(1)
    axis_right_x = joystick.get_axis(3)

    lin_vel_x = -axis_left_y * max_lin
    ang_vel = -axis_right_x * max_ang

    commands = [lin_vel_x, ang_vel, 0.0, 0.0]
    return commands, reset_flag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="stryon_no3_train")
    parser.add_argument("--ckpt", type=int, default=15000)
    args = parser.parse_args()

    gs.init(backend=gs.gpu)
    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, curriculum_cfg, domain_rand_cfg, terrain_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )

    train_cfg.setdefault("policy", {})["class_name"] = train_cfg["policy"].get("class_name", "ActorCritic")
    train_cfg.setdefault("algorithm", {})["class_name"] = train_cfg["algorithm"].get("class_name", "PPO")

    env = StryonNo3Env(
        num_envs=5,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        curriculum_cfg=curriculum_cfg,
        domain_rand_cfg=domain_rand_cfg,
        terrain_cfg=terrain_cfg,
        show_viewer=True,
        train_mode=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device="cuda:0")
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device="cuda:0")

    model = copy.deepcopy(runner.alg.actor_critic.actor).to("cpu")
    torch.jit.script(model).save(log_dir + "/policy.pt")

    joystick = init_pygame()

    obs, _ = env.reset()

    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rew, res, ext = env.step(actions)
            commands, reset_flag = get_joystick_commands(joystick)
            env.set_commands(np.arange(env.num_envs), commands)

            if reset_flag:
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
