import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
import numpy as np
import cv2


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


# TODO
# - Disabled damping curriculum
# - Domain randomization
# The complete code is https://github.com/Albusgive/wheel_legged_genesis/blob/main/locomotion/wheel_legged_env.py
class StryonNo3Env:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        curriculum_cfg,
        domain_rand_cfg,
        terrain_cfg,
        show_viewer=False,
        device="cuda",
        train_mode=True,
    ):
        self.num_envs = num_envs
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.curriculum_cfg = curriculum_cfg
        self.domain_rand_cfg = domain_rand_cfg
        self.terrain_cfg = terrain_cfg
        self.show_viewer = show_viewer
        self.device = torch.device(device)
        self.mode = train_mode

        self.lin_low_streak = 0
        self.ang_low_streak = 0
        self.leg_low_streak = 0

        self.simulate_action_latency = self.env_cfg.get("simulate_action_latency", False)
        self.dt = 0.01
        self.max_episode_length = math.ceil(self.env_cfg["episode_length_s"] / self.dt)
        self.obs_scales = self.obs_cfg["obs_scales"]
        self.reward_scales = self.reward_cfg["reward_scales"]
        self.history_length = obs_cfg["history_length"]
        self.num_commands = command_cfg["num_commands"]
        self.num_actions = env_cfg["num_actions"]
        self.noise = obs_cfg["noise"]

        # --- joint RMS 用 ---
        self.use_odo_for_lin_x_in_obs = False
        self.joint_names_only = [n for n in self.env_cfg["joint_names"]
                                 if self.env_cfg["joint_type"][n] == "joint"]
        self.n_joints = len(self.joint_names_only)
        self.joint_err_sse   = torch.zeros((self.num_envs, self.n_joints), device=self.device)  # Σ(err^2)
        self.joint_err_count = torch.zeros(self.num_envs, device=self.device)                   # サンプル数
        self.target_dof_pos_current = torch.zeros((self.num_envs, self.n_joints), device=self.device)

        # --- 相関（cmd_x vs vel_x[観測で使う値]）用：逐次更新で十分 ---
        self.corr_count  = torch.zeros(self.num_envs, device=self.device)
        self.corr_sum_x  = torch.zeros(self.num_envs, device=self.device)
        self.corr_sum_y  = torch.zeros(self.num_envs, device=self.device)
        self.corr_sum_x2 = torch.zeros(self.num_envs, device=self.device)
        self.corr_sum_y2 = torch.zeros(self.num_envs, device=self.device)
        self.corr_sum_xy = torch.zeros(self.num_envs, device=self.device)

        # Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                batch_dofs_info=True,
            ),
            show_viewer=self.show_viewer,
        )

        # Add plane
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # Init robot quat and pos
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        # Add terrain
        terrain_pos = []
        self.horizontal_scale = self.terrain_cfg["horizontal_scale"]
        self.vertical_scale = self.terrain_cfg["vertical_scale"]
        self.respawn_points = self.terrain_cfg["respawn_points"]
        self.height_field = cv2.imread(self.terrain_cfg["textures"], cv2.IMREAD_GRAYSCALE)
        self.terrain_height = torch.tensor(self.height_field, device=self.device) * self.vertical_scale
        if self.terrain_cfg["terrain"]:
            self.terrain = self.scene.add_entity(
                morph=gs.morphs.Terrain(
                    height_field=self.height_field,
                    horizontal_scale=self.horizontal_scale,
                    vertical_scale=self.vertical_scale,
                ),
            )
            if self.mode:
                for i in range(len(self.respawn_points)):
                    terrain_pos.append(self.respawn_points[i])

                self.num_respawn_points = len(terrain_pos)
                self.base_terrain_pos = torch.tensor(terrain_pos, device=self.device)
                self.base_terrain_pos[:, 2] += self.base_init_pos[2]

        # Add robot
        base_init_pos = self.base_init_pos.cpu().numpy()
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=self.env_cfg["mjcf"],
                pos=base_init_pos,
                quat=self.base_init_quat.cpu().numpy(),
                convexify=self.env_cfg["convexify"],
                decimate_aggressiveness=self.env_cfg["decimate_aggressiveness"],
            ),
            vis_mode="visual",
        )

        # Build
        self.scene.build(n_envs=self.num_envs)

        # Convert joint names to DOF indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]
        joint_dof_idx = []
        wheel_dof_idx = []
        self.joint_dof_idx = []
        self.wheel_dof_idx = []
        for i in range(len(self.env_cfg["joint_names"])):
            if self.env_cfg["joint_type"][self.env_cfg["joint_names"][i]] == "joint":
                joint_dof_idx.append(i)
                self.joint_dof_idx.append(self.motors_dof_idx[i])
            elif self.env_cfg["joint_type"][self.env_cfg["joint_names"][i]] == "wheel":
                wheel_dof_idx.append(i)
                self.wheel_dof_idx.append(self.motors_dof_idx[i])
        self.joint_dof_idx_np = np.array(joint_dof_idx)
        self.wheel_dof_idx_np = np.array(wheel_dof_idx)

        # PD control parameters
        self.kp = np.full((self.num_envs, self.num_actions), self.env_cfg["joint_kp"])
        self.kv = np.full((self.num_envs, self.num_actions), self.env_cfg["joint_kv"])
        self.kp[:, self.wheel_dof_idx_np] = 0.0
        self.kv[:, self.wheel_dof_idx_np] = self.env_cfg["wheel_kv"]
        self.robot.set_dofs_kp(self.kp, self.motors_dof_idx)
        self.robot.set_dofs_kv(self.kv, self.motors_dof_idx)

        damping = np.full((self.num_envs, self.robot.n_dofs), self.env_cfg["damping"])
        damping[:, :6] = 0
        self.damping_base = self.env_cfg["damping"]
        self.robot.set_dofs_damping(damping, np.arange(0, self.robot.n_dofs))

        armature = np.full((self.num_envs, self.robot.n_dofs), self.env_cfg["armature"])
        armature[:, :6] = 0
        self.robot.set_dofs_armature(armature, np.arange(0, self.robot.n_dofs))

        # DOF limits
        lower = [self.env_cfg["dof_limit"][name][0] for name in self.env_cfg["joint_names"]]
        upper = [self.env_cfg["dof_limit"][name][1] for name in self.env_cfg["joint_names"]]
        self.dof_pos_lower = torch.tensor(lower).to(self.device)
        self.dof_pos_upper = torch.tensor(upper).to(self.device)

        # Set safe force
        lower = np.array(
            [[-self.env_cfg["safe_force"][name] for name in self.env_cfg["joint_names"]] for _ in range(self.num_envs)]
        )
        upper = np.array(
            [[self.env_cfg["safe_force"][name] for name in self.env_cfg["joint_names"]] for _ in range(self.num_envs)]
        )
        self.robot.set_dofs_force_range(
            lower=torch.tensor(lower, device=self.device, dtype=torch.float32),
            upper=torch.tensor(upper, device=self.device, dtype=torch.float32),
            dofs_idx_local=self.motors_dof_idx,
        )

        # Prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # Initialize command parameters
        self.survive_ratio = 0.0

        # Prepare command_ranges lin_vel_x ang_vel height_target
        self.command_ranges = torch.zeros((self.num_envs, self.num_commands, 2), device=self.device, dtype=gs.tc_float)
        self.command_ranges[:, 0, 0] = (
            self.command_cfg["lin_vel_x_range"][0] * self.curriculum_cfg["curriculum_lin_vel_min_range"]
        )
        self.command_ranges[:, 0, 1] = (
            self.command_cfg["lin_vel_x_range"][1] * self.curriculum_cfg["curriculum_lin_vel_min_range"]
        )
        self.command_ranges[:, 1, 0] = (
            self.command_cfg["ang_vel_range"][0] * self.curriculum_cfg["curriculum_ang_vel_min_range"]
        )
        self.command_ranges[:, 1, 1] = (
            self.command_cfg["ang_vel_range"][1] * self.curriculum_cfg["curriculum_ang_vel_min_range"]
        )
        self.command_ranges[:, 2, 0] = (
            self.command_cfg["leg_length_angle_range"][0] * self.curriculum_cfg["curriculum_leg_min_range"]
        )
        self.command_ranges[:, 2, 1] = (
            self.command_cfg["leg_length_angle_range"][1] * self.curriculum_cfg["curriculum_leg_min_range"]
        )
        self.lin_vel_error = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.ang_vel_error = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.leg_error = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.curriculum_step = 0

        self.curriculum_stage = int(self.curriculum_cfg.get("start_stage", 0))  # 0: lin_x, 1: ang_yaw, 2: leg, 3: done
        self.phase_names = ["lin_x", "ang_yaw", "leg_length"]

        self.promote_streak  = int(self.curriculum_cfg.get("phase_promote_streak", 3))   # 何回連続で安定したら昇格か
        self.promote_margin  = float(self.curriculum_cfg.get("phase_promote_margin", 0.8)) # 拡張しきい値に対する何倍で昇格判定するか
        self.stage_low_streak = 0  # 現ステージの「安定連続回数」カウンタ

        # Initialize base state buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.odo_lin_vel_x = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.odo_ang_vel_z = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)

        # Initialize observation buffers
        self.slice_obs_buf = torch.zeros(
            (self.num_envs, self.obs_cfg["num_slice_obs"]), device=self.device, dtype=gs.tc_float
        )
        self.history_obs_buf = torch.zeros(
            (self.num_envs, self.obs_cfg["history_length"], self.obs_cfg["num_slice_obs"]),
            device=self.device,
            dtype=gs.tc_float,
        )
        self.obs_buf = torch.zeros((self.num_envs, self.obs_cfg["num_obs"]), device=self.device, dtype=gs.tc_float)
        self.history_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Initialize reward and reset buffers
        self.rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=self.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=self.device, dtype=gs.tc_int)

        # Initialize command buffer and normalization scale
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=self.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
                self.obs_scales["dof_pos"],
            ],
            device=self.device,
            dtype=gs.tc_float,
        )

        # Initialize action and joint state buffers
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.dof_force = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)

        # Initialize base pose buffers
        self.base_pos = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=self.device, dtype=gs.tc_float)

        # Default joint positions
        self.basic_default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=self.device,
            dtype=gs.tc_float,
        )

        # Expanded default/init joint positions for all environments
        default_dof_pos_list = [
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]]
        ] * self.num_envs
        self.default_dof_pos = torch.tensor(
            default_dof_pos_list,
            device=self.device,
            dtype=gs.tc_float,
        )

        init_dof_pos_list = [
            [self.env_cfg["joint_init_angles"][name] for name in self.env_cfg["joint_names"]]
        ] * self.num_envs
        self.init_dof_pos = torch.tensor(
            init_dof_pos_list,
            device=self.device,
            dtype=gs.tc_float,
        )

        self.action_smooth_alpha = 0.3
        self.max_joint_speed = 1.2
        self.last_actions = torch.zeros_like(self.actions)
        self.last_target_dof_pos = self.default_dof_pos[:, self.joint_dof_idx_np].clone()

        # Initialize contact force buffer
        self.connect_force = torch.zeros((self.num_envs, self.robot.n_links, 3), device=self.device, dtype=gs.tc_float)

        # Buffer for logging extra information
        self.extras = dict()  # Extra information for logging
        self.extras["observations"] = dict()

        # Base contact reset condition (note: use idx_local directly)
        if self.env_cfg["termination_if_base_connect_plane_than"] & self.mode:
            self.reset_links = [(self.robot.get_link(name).idx_local) for name in self.env_cfg["connect_plane_links"]]

        # Episode length tracking
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        # --- action smoothing / slew buffers ---
        self.alpha_joint = self.env_cfg.get("action_smooth_alpha_joint", 0.30)
        self.alpha_wheel = self.env_cfg.get("action_smooth_alpha_wheel", 0.90)
        self.max_joint_speed = self.env_cfg.get("max_joint_speed", 1.8)

        self.last_actions_joint = torch.zeros((self.num_envs, len(self.joint_dof_idx_np)),
                                            device=self.device, dtype=gs.tc_float)
        self.last_actions_wheel = torch.zeros((self.num_envs, len(self.wheel_dof_idx_np)),
                                            device=self.device, dtype=gs.tc_float)

        # “直前に送った関節目標角” を覚えてスルー制限で使う
        self.last_target_dof_pos = self.default_dof_pos[:, self.joint_dof_idx_np].clone()

        # Call reset to finalize initialization
        self.reset()

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.obs_buf, None

    def set_commands(self, envs_idx, commands):
        self.commands[envs_idx] = torch.tensor(commands, device=self.device, dtype=gs.tc_float)

    def step(self, actions):
        # ========= 0) 1回限りの安全初期化（__init__ に入れてもOK） =========
        if not hasattr(self, "alpha_joint"):
            self.alpha_joint = self.env_cfg.get("action_smooth_alpha_joint",
                                                self.env_cfg.get("action_smooth_alpha", 0.30))
        if not hasattr(self, "alpha_wheel"):
            self.alpha_wheel = self.env_cfg.get("action_smooth_alpha_wheel", 0.90)
        if not hasattr(self, "max_joint_speed"):
            self.max_joint_speed = self.env_cfg.get("max_joint_speed", 1.2)  # [rad/s]

        # バッファ（初回のみ確保）
        if not hasattr(self, "last_actions_joint"):
            self.last_actions_joint = torch.zeros(
                (self.num_envs, len(self.joint_dof_idx_np)), device=self.device, dtype=gs.tc_float
            )
        if not hasattr(self, "last_actions_wheel"):
            self.last_actions_wheel = torch.zeros(
                (self.num_envs, len(self.wheel_dof_idx_np)), device=self.device, dtype=gs.tc_float
            )
        if not hasattr(self, "last_target_dof_pos"):
            self.last_target_dof_pos = self.default_dof_pos[:, self.joint_dof_idx_np].clone()
        if not hasattr(self, "target_dof_pos_current"):
            self.target_dof_pos_current = self.last_target_dof_pos.clone()

        # RMS / 相関の蓄積バッファ（envごと）
        if not hasattr(self, "joint_err_sse"):
            J = len(self.joint_dof_idx_np)
            self.joint_err_sse   = torch.zeros((self.num_envs, J), device=self.device, dtype=gs.tc_float)
            self.joint_err_count = torch.zeros(self.num_envs,      device=self.device, dtype=gs.tc_float)
        if not hasattr(self, "corr_count"):
            self.corr_count  = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
            self.corr_sum_x  = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
            self.corr_sum_y  = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
            self.corr_sum_x2 = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
            self.corr_sum_y2 = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
            self.corr_sum_xy = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)

        # ========= 1) 行動ローパス（関節/車輪で係数を分ける）→ スルーレート制限（関節のみ） =========
        clip = self.env_cfg["clip_actions"]
        raw = torch.clamp(actions, -clip, clip)

        # 分割
        raw_j = raw[:, self.joint_dof_idx_np]
        raw_w = raw[:, self.wheel_dof_idx_np]

        # ローパス（指数移動平均）
        sm_j = (1.0 - self.alpha_joint) * self.last_actions_joint + self.alpha_joint * raw_j
        sm_w = (1.0 - self.alpha_wheel) * self.last_actions_wheel + self.alpha_wheel * raw_w

        # レイテンシ（有効時は一拍遅らせる）
        exec_j = self.last_actions_joint if self.simulate_action_latency else sm_j
        exec_w = self.last_actions_wheel if self.simulate_action_latency else sm_w

        # 関節：スケール → スルーレート制限 → 角度クリップ
        target_nominal = exec_j * self.env_cfg["joint_action_scale"] + self.default_dof_pos[:, self.joint_dof_idx_np]
        max_step = self.max_joint_speed * self.dt
        delta = torch.clamp(target_nominal - self.last_target_dof_pos, -max_step, +max_step)
        target_dof_pos = self.last_target_dof_pos + delta
        target_dof_pos = torch.clamp(
            target_dof_pos,
            min=self.dof_pos_lower[self.joint_dof_idx_np],
            max=self.dof_pos_upper[self.joint_dof_idx_np],
        )
        self.last_target_dof_pos[:] = target_dof_pos
        self.target_dof_pos_current = target_dof_pos  # RMS 用

        # 車輪：速度目標（スルー制限は通常なし）
        target_dof_vel = exec_w * self.env_cfg["wheel_action_scale"]

        # 観測用 self.actions を“平滑後”で更新（関節/車輪をマージ）
        if not hasattr(self, "actions"):
            self.actions = torch.zeros((self.num_envs, self.num_actions), device=self.device, dtype=gs.tc_float)
        self.actions[:, self.joint_dof_idx_np] = sm_j
        self.actions[:, self.wheel_dof_idx_np] = sm_w

        # コマンド投入
        self.robot.control_dofs_position(target_dof_pos, self.joint_dof_idx)
        self.robot.control_dofs_velocity(target_dof_vel, self.wheel_dof_idx)

        # ========= 2) 物理ステップ =========
        self.scene.step()

        # ========= 3) 状態の読み出し =========
        self.episode_length_buf += 1
        self.base_pos[:] = self.get_relative_terrain_pos(self.robot.get_pos())
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler_deg = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True, degrees=True,
        )
        self.base_euler = self.base_euler_deg * (math.pi / 180.0)

        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.dof_force[:] = self.robot.get_dofs_force(self.motors_dof_idx)

        # ========= 4) RMS 誤差を蓄積（ノイズ適用前） =========
        joint_pos = self.dof_pos[:, self.joint_dof_idx_np]
        err = joint_pos - self.target_dof_pos_current
        self.joint_err_sse   += err * err
        self.joint_err_count += 1.0

        # ========= 5) オドメトリ =========
        l_w = self.dof_vel[:, 2] * self.command_cfg["wheel_forward_sign"]["left_wheel_joint"]
        r_w = self.dof_vel[:, 5] * self.command_cfg["wheel_forward_sign"]["right_wheel_joint"]
        self.odo_lin_vel_x = 0.5 * self.command_cfg["wheel_radius"] * (l_w + r_w)
        self.odo_ang_vel_z = (self.command_cfg["wheel_radius"] / self.command_cfg["axle_width"]) * (r_w - l_w)

        # ========= 6) cmd_x と vel_x の相関を蓄積 =========
        use_odo = getattr(self, "use_odo_for_lin_x_in_obs", False)
        vel_x_obs = self.odo_lin_vel_x if use_odo else self.base_lin_vel[:, 0]
        x = self.commands[:, 0]
        y = vel_x_obs
        self.corr_count  += 1.0
        self.corr_sum_x  += x;    self.corr_sum_y  += y
        self.corr_sum_x2 += x * x
        self.corr_sum_y2 += y * y
        self.corr_sum_xy += x * y
        xm = x.mean(); ym = y.mean()
        xs = (x - xm).pow(2).mean().sqrt() + 1e-6
        ys = (y - ym).pow(2).mean().sqrt() + 1e-6
        self.extras["corr_cmdx_velx"] = (((x - xm) * (y - ym)).mean() / (xs * ys)).item()

        # ========= 7) センサノイズ =========
        if self.noise["use"]:
            self.base_ang_vel[:] += (
                torch.randn_like(self.base_ang_vel) * self.noise["ang_vel"][0]
                + (torch.rand_like(self.base_ang_vel) * 2 - 1) * self.noise["ang_vel"][1]
            )
            self.base_euler += (
                torch.randn_like(self.base_euler) * self.noise["base_euler"][0]
                + (torch.rand_like(self.base_euler) * 2 - 1) * self.noise["base_euler"][1]
            )
            self.dof_pos[:] += (
                torch.randn_like(self.dof_pos) * self.noise["dof_pos"][0]
                + (torch.rand_like(self.dof_pos) * 2 - 1) * self.noise["dof_pos"][1]
            )
            self.dof_vel[:] += (
                torch.randn_like(self.dof_vel) * self.noise["dof_vel"][0]
                + (torch.rand_like(self.dof_vel) * 2 - 1) * self.noise["dof_vel"][1]
            )

        # ========= 8) 残りの既存処理 =========
        self.connect_force = self.robot.get_links_net_contact_force()
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.last_base_ang_vel[:] = self.base_ang_vel[:]
        self.episode_lengths += 1

        envs_idx = ((self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
                    .nonzero(as_tuple=False).flatten())

        self.lin_vel_error += torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        self.ang_vel_error += torch.abs(self.commands[:, 1] - self.base_ang_vel[:, 2])
        self.leg_error     += torch.abs(self.commands[:, 2] - (self.dof_pos[:, 0] + self.dof_pos[:, 3]) / 2.0)

        if self.mode:
            self.check_termination()

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        if self.mode:
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        if self.mode:
            self.rew_buf[:] = 0.0
            for name, reward_func in self.reward_functions.items():
                rew = reward_func() * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew

        if self.mode:
            self.resample_commands(envs_idx)
            self.curriculum_commands()

        lin_vel_obs = self.base_lin_vel.clone()
        # lin_vel_obs[:, 0] = self.odo_lin_vel_x  # 観測にオドメトリを使う場合

        joint_names_only = [n for n in self.env_cfg["joint_names"] if self.env_cfg["joint_type"][n] == "joint"]
        joint_safe_force_vec = torch.tensor(
            [self.env_cfg["safe_force"][n] for n in joint_names_only], device=self.device, dtype=gs.tc_float
        )
        self.slice_obs_buf = torch.cat(
            [
                lin_vel_obs * self.obs_scales["lin_vel"],                   # 3
                self.base_ang_vel * self.obs_scales["ang_vel"],             # 3
                self.base_euler   * self.obs_scales["base_euler"],          # 3
                self.commands     * self.commands_scale,                    # 3
                self.dof_pos[:, self.joint_dof_idx_np] * self.obs_scales["dof_pos"],  # 4
                (self.dof_force[:, self.joint_dof_idx_np] / (joint_safe_force_vec + 1e-6)).clamp(-1.0, 1.0),  # 4
                self.actions / self.env_cfg["clip_actions"],                # 6（平滑後）
            ],
            dim=-1,
        )

        self.last_actions[:] = self.actions[:]          # 旧API互換（報酬の action_rate で使用）
        self.last_dof_vel[:] = self.dof_vel[:]

        self.obs_buf = torch.cat([self.history_obs_buf, self.slice_obs_buf.unsqueeze(1)], dim=1).view(self.num_envs, -1)
        if self.history_length > 1:
            self.history_obs_buf[:, :-1, :] = self.history_obs_buf[:, 1:, :].clone()
        self.history_obs_buf[:, -1, :] = self.slice_obs_buf
        self.extras["observations"]["critic"] = self.obs_buf

        # 次回用：平滑行動の内部状態を更新
        self.last_actions_joint[:] = sm_j
        self.last_actions_wheel[:] = sm_w

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def check_termination(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler_deg[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler_deg[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # --- Logging statistics before reset ---
        episode_lengths = self.episode_length_buf[envs_idx]
        avg_episode_length = episode_lengths.float().mean().item()
        max_episode_length = self.max_episode_length

        # Compute the survive ratio (average episode progress before reset)
        self.survive_ratio = avg_episode_length / max_episode_length

        # Reset DOF positions and velocities
        self.dof_pos[envs_idx] = self.init_dof_pos[envs_idx]
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # Apply base position and orientation
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        if self.terrain_cfg["terrain"] and self.mode and self.num_respawn_points > 0:
            n = len(envs_idx)
            use_spawn0 = (self.curriculum_stage == 0) and bool(self.curriculum_cfg.get("stage0_spawn0_only", False))
            if use_spawn0:
                rand_idx = torch.zeros(n, dtype=torch.long, device=self.device)
            else:
                rand_idx = torch.randint(low=0, high=self.num_respawn_points, size=(n,), device=self.device)
            self.base_pos[envs_idx] = self.base_terrain_pos[rand_idx]
        else:
            self.base_pos[envs_idx] = self.base_init_pos

        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)

        # Reset velocities
        self.base_lin_vel[envs_idx] = 0
        self.odo_lin_vel_x[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.odo_ang_vel_z[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # Reset internal buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # Fill logging extras with episode rewards
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        # === joint RMS 誤差（関節ごと）を出力 ===
        eps = 1e-9
        rms_env_joint = torch.sqrt(self.joint_err_sse[envs_idx] /
                                        (self.joint_err_count[envs_idx].unsqueeze(1) + eps))  # [E, n_joints]
        rms_per_joint = rms_env_joint.mean(dim=0)  # env 平均
        for j, name in enumerate(self.joint_names_only):
            self.extras["episode"][f"rms_joint_err/{name}"] = rms_per_joint[j].item()
        # まとめ指標（平均 & 最大）
        self.extras["episode"]["rms_joint_err/mean"] = rms_per_joint.mean().item()
        self.extras["episode"]["rms_joint_err/max"]  = rms_per_joint.max().item()

        # 蓄積をリセット
        self.joint_err_sse[envs_idx] = 0.0
        self.joint_err_count[envs_idx] = 0.0

        # === cmd_x vs vel_x(観測で使っている) の Pearson 相関 ===
        n   = self.corr_count[envs_idx]
        sx  = self.corr_sum_x[envs_idx];  sy  = self.corr_sum_y[envs_idx]
        sxx = self.corr_sum_x2[envs_idx]; syy = self.corr_sum_y2[envs_idx]
        sxy = self.corr_sum_xy[envs_idx]
        num = n * sxy - sx * sy
        den = torch.sqrt(torch.clamp(n * sxx - sx * sx, min=eps) *
                        torch.clamp(n * syy - sy * sy, min=eps))
        r_env  = torch.where(den > 0, num / den, torch.zeros_like(den))
        r_mean = torch.nanmean(r_env)
        self.extras["episode"]["corr_cmdx_velx"] = r_mean.item()
        # リセット
        self.corr_count[envs_idx] = 0.0
        self.corr_sum_x[envs_idx] = 0.0; self.corr_sum_y[envs_idx] = 0.0
        self.corr_sum_x2[envs_idx] = 0.0; self.corr_sum_y2[envs_idx] = 0.0
        self.corr_sum_xy[envs_idx] = 0.0

        self.last_actions_joint[envs_idx] = 0.0
        self.last_actions_wheel[envs_idx] = 0.0
        self.last_target_dof_pos[envs_idx] = self.default_dof_pos[envs_idx][:, self.joint_dof_idx_np]

        # Resample new commands
        self.resample_commands(envs_idx)

        # Reset episode length
        self.episode_lengths[envs_idx] = 0.0

        # Domain randomization
        self.domain_rand(envs_idx)

        # Reset curriculum
        self.lin_vel_error[envs_idx] = 0
        self.ang_vel_error[envs_idx] = 0
        self.leg_error[envs_idx] = 0

    def resample_commands(self, envs_idx):
        for idx in envs_idx:
            for command_idx in range(self.num_commands):
                low = self.command_ranges[idx, command_idx, 0]
                high = self.command_ranges[idx, command_idx, 1]
                self.commands[idx, command_idx] = gs_rand_float(low, high, (1,), self.device)
    
    def curriculum_commands(self):
        # 何ステップごとに評価するか
        self.curriculum_step += 1
        if self.curriculum_step < self.curriculum_cfg["curriculum_step"]:
            return
        self.curriculum_step = 0

        # ── 誤差の平均（1エピソードあたり）を算出 ──
        valid = self.episode_length_buf > 0
        if valid.any():
            self.mean_lin_vel_error = (self.lin_vel_error[valid] / self.episode_length_buf[valid]).mean().item()
            self.mean_ang_vel_error = (self.ang_vel_error[valid] / self.episode_length_buf[valid]).mean().item()
            self.mean_leg_error     = (self.leg_error[valid]     / self.episode_length_buf[valid]).mean().item()
        else:
            self.mean_lin_vel_error = float("nan")
            self.mean_ang_vel_error = float("nan")
            self.mean_leg_error     = float("nan")

        # 連続クリア回数の要求（未設定なら2回）
        expand_streak = int(self.curriculum_cfg.get("expand_streak", 1))

        # 便利関数：フルレンジ到達判定（平均で見て十分近ければOK）
        def _range_full(cmd_idx, low_full, high_full, eps=1e-5):
            lo = self.command_ranges[:, cmd_idx, 0].mean().item()
            hi = self.command_ranges[:, cmd_idx, 1].mean().item()
            return (lo <= low_full + abs(low_full) * eps) and (hi >= high_full - abs(high_full) * eps)

        # 便利関数：1次元の拡張・縮小（元のロジックを薄くラップ）
        def _expand_shrink_1d(cmd_idx, step_frac, base_range, mean_err, err_range_cfg, min_range_frac, tag):
            """
            cmd_idx: 0=lin_x, 1=ang_yaw, 2=leg
            step_frac: curriculum_*_step
            base_range: command_cfg[...] (tuple like [-1.0, 1.0])
            err_range_cfg: curriculum_cfg[...]_err_range
            min_range_frac: curriculum_*_min_range
            tag: logging label
            """
            if self.curriculum_cfg["err_mode"]:
                up_thr = err_range_cfg[0]
                hi_thr = err_range_cfg[1]
            else:
                ratio = (self.command_ranges[:, cmd_idx, 1].mean() / base_range[1]).item()
                up_thr = (err_range_cfg[0] + (err_range_cfg[1] - err_range_cfg[0]) * ratio)
                hi_thr = err_range_cfg[2]

            old = self.command_ranges[0, cmd_idx].clone().cpu().numpy()
            action_txt = "→ unchanged"

            # ヒステリシス：拡張は複数回連続で低誤差
            promote_local = False
            if mean_err < up_thr:
                if cmd_idx == 0:  self.lin_low_streak += 1
                if cmd_idx == 1:  self.ang_low_streak += 1
                if cmd_idx == 2:  self.leg_low_streak += 1
            else:
                if cmd_idx == 0:  self.lin_low_streak = 0
                if cmd_idx == 1:  self.ang_low_streak = 0
                if cmd_idx == 2:  self.leg_low_streak = 0

            low_step  = step_frac * base_range[0]
            high_step = step_frac * base_range[1]

            low_idx  = (self.lin_low_streak  if cmd_idx == 0 else (self.ang_low_streak if cmd_idx == 1 else self.leg_low_streak))
            if low_idx >= expand_streak:
                self.command_ranges[:, cmd_idx, 0] += low_step
                self.command_ranges[:, cmd_idx, 1] += high_step
                action_txt = f"↑ increased (streak={low_idx})"
                # streak をリセット（昇格判定は別カウンタで）
                if cmd_idx == 0:  self.lin_low_streak = 0
                if cmd_idx == 1:  self.ang_low_streak = 0
                if cmd_idx == 2:  self.leg_low_streak = 0

            # 縮小（即時）
            if mean_err > hi_thr:
                self.command_ranges[:, cmd_idx, 0] -= low_step
                self.command_ranges[:, cmd_idx, 1] -= high_step
                action_txt = "↓ decreased"
                if cmd_idx == 0:  self.lin_low_streak = 0
                if cmd_idx == 1:  self.ang_low_streak = 0
                if cmd_idx == 2:  self.leg_low_streak = 0

            # Clamp
            if cmd_idx == 0:
                min_lo = self.command_cfg["lin_vel_x_range"][0]
                max_hi = self.command_cfg["lin_vel_x_range"][1]
            elif cmd_idx == 1:
                min_lo = self.command_cfg["ang_vel_range"][0]
                max_hi = self.command_cfg["ang_vel_range"][1]
            else:
                min_lo = self.command_cfg["leg_length_angle_range"][0]
                max_hi = self.command_cfg["leg_length_angle_range"][1]

            self.command_ranges[:, cmd_idx, 0] = torch.clamp(
                self.command_ranges[:, cmd_idx, 0],
                min_lo if cmd_idx != 2 else min_lo * min_range_frac,  # leg は下限=最小比*base_lo
                min_range_frac * min_lo if cmd_idx in (0,1) else None  # この値は下で無視（Noneで扱わない）
            )
            # 上側クランプ
            self.command_ranges[:, cmd_idx, 1] = torch.clamp(
                self.command_ranges[:, cmd_idx, 1],
                min_range_frac * max_hi,
                max_hi
            )

            # torch.clamp の第3引数に None は入れられないので、下側は明示的に分ける
            if cmd_idx == 0:
                self.command_ranges[:, cmd_idx, 0] = torch.clamp(
                    self.command_ranges[:, cmd_idx, 0],
                    min_lo,
                    self.curriculum_cfg["curriculum_lin_vel_min_range"] * self.command_cfg["lin_vel_x_range"][0],
                )
                self.command_ranges[:, cmd_idx, 1] = torch.clamp(
                    self.command_ranges[:, cmd_idx, 1],
                    self.curriculum_cfg["curriculum_lin_vel_min_range"] * self.command_cfg["lin_vel_x_range"][1],
                    max_hi,
                )
            elif cmd_idx == 1:
                self.command_ranges[:, cmd_idx, 0] = torch.clamp(
                    self.command_ranges[:, cmd_idx, 0],
                    min_lo,
                    self.curriculum_cfg["curriculum_ang_vel_min_range"] * self.command_cfg["ang_vel_range"][0],
                )
                self.command_ranges[:, cmd_idx, 1] = torch.clamp(
                    self.command_ranges[:, cmd_idx, 1],
                    self.curriculum_cfg["curriculum_ang_vel_min_range"] * self.command_cfg["ang_vel_range"][1],
                    max_hi,
                )
            else:
                self.command_ranges[:, cmd_idx, 0] = torch.clamp(
                    self.command_ranges[:, cmd_idx, 0],
                    self.curriculum_cfg["curriculum_leg_min_range"] * self.command_cfg["leg_length_angle_range"][0],
                )
                self.command_ranges[:, cmd_idx, 1] = torch.clamp(
                    self.command_ranges[:, cmd_idx, 1],
                    self.curriculum_cfg["curriculum_leg_min_range"] * self.command_cfg["leg_length_angle_range"][1],
                    max_hi,
                )

            new = self.command_ranges[0, cmd_idx].clone().cpu().numpy()
            print(f"[Curriculum] {tag} | err={mean_err:.4f} | "
                f"range: [{old[0]:.3f}, {old[1]:.3f}] → [{new[0]:.3f}, {new[1]:.3f}] | {action_txt}")

            # 昇格用：拡張しきい値より「さらに厳しめ」に安定しているか
            promote_thr = up_thr * self.promote_margin
            is_stable = (mean_err < promote_thr)
            return is_stable

        if self.survive_ratio <= 0.9:
            return  # 生存率が低い間は拡張しない

        # ===== ステージ分岐 =====
        if self.curriculum_stage == 0:
            # --- 0) linear x のみ拡張 ---
            stable = _expand_shrink_1d(
                cmd_idx=0,
                step_frac=self.curriculum_cfg["curriculum_lin_vel_step"],
                base_range=self.command_cfg["lin_vel_x_range"],
                mean_err=self.mean_lin_vel_error,
                err_range_cfg=self.curriculum_cfg["lin_vel_err_range"],
                min_range_frac=self.curriculum_cfg["curriculum_lin_vel_min_range"],
                tag="lin_vel_x"
            )
            # フルレンジ到達かつ安定が一定回数続いたら昇格
            lin_full = _range_full(0, self.command_cfg["lin_vel_x_range"][0], self.command_cfg["lin_vel_x_range"][1])
            self.stage_low_streak = (self.stage_low_streak + 1) if stable else 0
            if lin_full and self.stage_low_streak >= self.promote_streak:
                self.curriculum_stage = 1
                self.stage_low_streak = 0
                print("[Curriculum] >>> promote to stage 1 (ang_yaw)")

        elif self.curriculum_stage == 1:
            # --- 1) yaw のみ拡張（lin_x は凍結） ---
            stable = _expand_shrink_1d(
                cmd_idx=1,
                step_frac=self.curriculum_cfg["curriculum_ang_vel_step"],
                base_range=self.command_cfg["ang_vel_range"],
                mean_err=self.mean_ang_vel_error,
                err_range_cfg=self.curriculum_cfg["ang_vel_err_range"],
                min_range_frac=self.curriculum_cfg["curriculum_ang_vel_min_range"],
                tag="ang_vel_yaw"
            )
            ang_full = _range_full(1, self.command_cfg["ang_vel_range"][0], self.command_cfg["ang_vel_range"][1])
            self.stage_low_streak = (self.stage_low_streak + 1) if stable else 0
            if ang_full and self.stage_low_streak >= self.promote_streak:
                self.curriculum_stage = 2
                self.stage_low_streak = 0
                print("[Curriculum] >>> promote to stage 2 (leg_length)")

        elif self.curriculum_stage == 2:
            # --- 2) leg のみ拡張（lin_x, yaw は凍結） ---
            stable = _expand_shrink_1d(
                cmd_idx=2,
                step_frac=self.curriculum_cfg["curriculum_leg_step"],
                base_range=self.command_cfg["leg_length_angle_range"],
                mean_err=self.mean_leg_error,
                err_range_cfg=self.curriculum_cfg["leg_err_range"],
                min_range_frac=self.curriculum_cfg["curriculum_leg_min_range"],
                tag="leg_length"
            )
            leg_full = _range_full(2, self.command_cfg["leg_length_angle_range"][0], self.command_cfg["leg_length_angle_range"][1])
            self.stage_low_streak = (self.stage_low_streak + 1) if stable else 0
            if leg_full and self.stage_low_streak >= self.promote_streak:
                self.curriculum_stage = 3
                self.stage_low_streak = 0
                print("[Curriculum] >>> promote to stage 3 (done)")

        else:
            # --- 3) done：拡張は行わない（凍結） ---
            return

    def get_relative_terrain_pos(self, base_pos):
        if not self.terrain_cfg["terrain"]:
            return base_pos

        x = base_pos[:, 0]
        y = base_pos[:, 1]

        fx = x / self.horizontal_scale
        fy = y / self.horizontal_scale

        x0 = torch.floor(fx).int()
        x1 = torch.min(x0 + 1, torch.full_like(x0, self.terrain_height.shape[1] - 1))
        y0 = torch.floor(fy).int()
        y1 = torch.min(y0 + 1, torch.full_like(y0, self.terrain_height.shape[0] - 1))

        x0 = torch.clamp(x0, 0, self.terrain_height.shape[1] - 1)
        x1 = torch.clamp(x1, 0, self.terrain_height.shape[1] - 1)
        y0 = torch.clamp(y0, 0, self.terrain_height.shape[0] - 1)
        y1 = torch.clamp(y1, 0, self.terrain_height.shape[0] - 1)

        Q11 = self.terrain_height[y0, x0]
        Q21 = self.terrain_height[y0, x1]
        Q12 = self.terrain_height[y1, x0]
        Q22 = self.terrain_height[y1, x1]
        wx = fx - x0
        wy = fy - y0
        height = (1 - wx) * (1 - wy) * Q11 + wx * (1 - wy) * Q21 + (1 - wx) * wy * Q12 + wx * wy * Q22
        base_pos[:, 2] -= height
        return base_pos

    def domain_rand(self, envs_idx):
        device = self.device
        n = len(envs_idx)
        L = self.robot.n_links
        link_ids = np.arange(L)

        # === 0) よく使う index・マスク ===
        base_idx = self.robot.get_link("body_link").idx_local
        link_mask = torch.ones(L, dtype=torch.bool, device=device)
        link_mask[base_idx] = False
        n_others = int(link_mask.sum())

        # === 1) Friction ratio（リンクごと）===
        fr_min, fr_max = self.domain_rand_cfg["friction_ratio_range"]
        fr_ratio = torch.empty((n, L), device=device).uniform_(fr_min, fr_max)
        self.robot.set_friction_ratio(fr_ratio, links_idx_local=link_ids, envs_idx=envs_idx)

        # === 2) Mass shift [kg] ===
        bmin, bmax = self.domain_rand_cfg["random_base_mass_shift_range"]
        omin, omax = self.domain_rand_cfg["random_other_mass_shift_range"]

        mass_shift = torch.zeros((n, L), device=device)
        mass_shift[:, base_idx] = torch.empty((n,), device=device).uniform_(bmin, bmax)
        mass_shift[:, link_mask] = torch.empty((n, n_others), device=device).uniform_(omin, omax)

        self.robot.set_mass_shift(mass_shift, links_idx_local=link_ids, envs_idx=envs_idx)

        # === 3) COM shift [m]（±r の立方体内一様）===
        r_base = self.domain_rand_cfg["random_base_com_shift"]
        r_other = self.domain_rand_cfg["random_other_com_shift"]
        com_shift = torch.zeros((n, L, 3), device=device)
        com_shift[:, base_idx, :] = (torch.rand((n, 3), device=device) * 2 - 1) * r_base
        com_shift[:, link_mask, :] = (torch.rand((n, n_others, 3), device=device) * 2 - 1) * r_other
        self.robot.set_COM_shift(com_shift, links_idx_local=link_ids, envs_idx=envs_idx)

        # === 4) PD ゲイン ===
        kp_lo, kp_hi = self.domain_rand_cfg["random_KP"]
        kv_lo, kv_hi = self.domain_rand_cfg["random_KV"]

        kp_rand = torch.empty((n, self.num_actions), device=device).uniform_(kp_lo, kp_hi) * torch.tensor(
            self.kp[0], device=device
        )
        kv_rand = torch.empty((n, self.num_actions), device=device).uniform_(kv_lo, kv_hi) * torch.tensor(
            self.kv[0], device=device
        )

        # ホイールは position KP を常に 0 に（速度制御のため）
        kp_rand[:, self.wheel_dof_idx_np] = 0.0

        self.robot.set_dofs_kp(kp_rand, self.motors_dof_idx, envs_idx=envs_idx)
        self.robot.set_dofs_kv(kv_rand, self.motors_dof_idx, envs_idx=envs_idx)

        # === 5) default joint angles（関節のみ）===
        dj_lo, dj_hi = self.domain_rand_cfg["random_default_joint_angles"]
        shift = torch.zeros((n, self.num_actions), device=device, dtype=gs.tc_float)
        shift[:, self.joint_dof_idx_np] = torch.empty(
            (n, len(self.joint_dof_idx_np)), device=device, dtype=gs.tc_float
        ).uniform_(dj_lo, dj_hi)
        self.default_dof_pos[envs_idx] = (
            self.basic_default_dof_pos + shift[0]
        )  # 全env同一にしたいなら [0]、env毎に変えたいならそのまま

        # === 6) Damping / Armature ===
        d_lo, d_hi = self.domain_rand_cfg["damping_range"]
        damping = torch.empty((n, self.robot.n_dofs), device=device).uniform_(d_lo, d_hi) * self.damping_base
        damping[:, :6] = 0
        self.robot.set_dofs_damping(damping=damping, dofs_idx_local=np.arange(0, self.robot.n_dofs), envs_idx=envs_idx)

        a_lo, a_hi = self.domain_rand_cfg["dof_armature_range"]
        armature = torch.empty((n, self.robot.n_dofs), device=device).uniform_(a_lo, a_hi)
        armature[:, :6] = 0
        self.robot.set_dofs_armature(
            armature=armature, dofs_idx_local=np.arange(0, self.robot.n_dofs), envs_idx=envs_idx
        )

    # def reward_tracking_lin_x_vel(self):
    #     cmd = self.commands[:, 0]
    #     err = cmd - self.base_lin_vel[:, 0]
    #     print("cmd, self.base_lin_vel[:, 0], err:", cmd[0].item(), self.base_lin_vel[0, 0].item(), err[0].item())
    #     e = err / (0.2 + torch.abs(cmd))
    #     sig = self.reward_cfg["tracking_linx_sigma"]
    #     r = torch.exp(-(e * e) / (sig * sig))
    #     if self.command_cfg["zero_stable"]:
    #         nz = (torch.abs(cmd) <= self.command_cfg["zero_eps"]).float()
    #         r = r + r * nz * self.command_cfg["zero_stable_gain"]
    #     return r

    # def reward_tracking_ang_vel(self):
    #     cmd = self.commands[:, 1]
    #     err = cmd - self.base_ang_vel[:, 2]
    #     e = err / (0.3 + torch.abs(cmd))
    #     sig = self.reward_cfg["tracking_ang_sigma"]
    #     r = torch.exp(-(e * e) / (sig * sig))
    #     if self.command_cfg["zero_stable"]:
    #         nz = (torch.abs(cmd) <= self.command_cfg["zero_eps"]).float()
    #         r = r + r * nz * self.command_cfg["zero_stable_gain"]
    #     return r

    def reward_tracking_lin_x_vel(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        print("cmd, self.base_lin_vel[:, 0], err:", self.commands[0, 0].item(), self.base_lin_vel[0, 0].item(), lin_vel_error[0].item())
        lin_vel_reward = torch.exp(-lin_vel_error / (self.reward_cfg["tracking_linx_sigma"] ** 2))
        alpha = self.reward_cfg["tracking_linx_alpha"]
        lin_vel_reward = lin_vel_reward * (self.survive_ratio + alpha * (1.0 - self.survive_ratio))
        return lin_vel_reward

    def reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.abs(self.commands[:, 1] - self.base_ang_vel[:, 2])
        ang_vel_reward = torch.exp(-ang_vel_error / (self.reward_cfg["tracking_ang_sigma"] ** 2))
        alpha = self.reward_cfg["tracking_ang_alpha"]
        ang_vel_reward = ang_vel_reward * (self.survive_ratio + alpha * (1.0 - self.survive_ratio))
        return ang_vel_reward

    def reward_tracking_roll(self):
        err = self.base_euler_deg[:, 0] - self.reward_cfg["target_roll"]
        sigma = self.reward_cfg.get("tracking_roll_sigma",
                                    self.reward_cfg["tracking_base_euler_sigma"])
        return torch.exp(- (err * err) / (sigma * sigma))

    def reward_tracking_pitch(self):
        err = self.base_euler_deg[:, 1] - self.reward_cfg["target_pitch"]
        sigma = self.reward_cfg.get("tracking_pitch_sigma",
                                    self.reward_cfg["tracking_base_euler_sigma"])
        return torch.exp(- (err * err) / (sigma * sigma))

    def reward_tracking_leg_length(self):
        knee_left = self.dof_pos[:, 0]
        knee_right = self.dof_pos[:, 3]
        avg_knee = (knee_left + knee_right) / 2.0
        err = torch.abs(avg_knee - self.commands[:, 2])
        sig = self.reward_cfg["tracking_leg_sigma"]
        return torch.exp(-(err**2) / (sig**2))

    def reward_similar_leg(self):
        left_sum = self.dof_pos[:, 0] + self.dof_pos[:, 1]
        right_sum = self.dof_pos[:, 3] + self.dof_pos[:, 4]
        diff = left_sum - right_sum
        sig = float(self.reward_cfg["similar_leg_sigma"])
        sig = max(sig, 1e-6)
        return torch.exp(-(diff * diff) / (sig * sig))

    def reward_joint_action_rate(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(self.last_actions[:, self.joint_dof_idx_np] - self.actions[:, self.joint_dof_idx_np]), dim=1
        )

    def reward_wheel_action_rate(self):
        # Penalize changes in actions
        return torch.sum(
            torch.square(self.last_actions[:, self.wheel_dof_idx_np] - self.actions[:, self.wheel_dof_idx_np]), dim=1
        )

    def reward_dof_acc(self):
        # Penalize changes in DOF velocities
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel) / self.dt), dim=1)

    def reward_dof_force(self):
        # Penalize changes in DOF forces
        return torch.sum(torch.square(self.dof_force), dim=1)

    def reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def reward_collision(self):
        # Penalize collisions with the environment
        collision = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        for idx in self.reset_links:
            collision += torch.square(self.connect_force[:, idx, :]).sum(dim=1)
        return collision

    def reward_survive(self):
        # Reward for surviving
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
