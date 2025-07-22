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
# - Terrain support (Genesis v0.2.1)
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

        self.simulate_action_latency = True
        self.dt = 0.01
        self.max_episode_length = math.ceil(self.env_cfg["episode_length_s"] / self.dt)
        self.obs_scales = self.obs_cfg["obs_scales"]
        self.reward_scales = self.reward_cfg["reward_scales"]
        self.history_length = obs_cfg["history_length"]
        self.num_commands = command_cfg["num_commands"]
        self.num_actions = env_cfg["num_actions"]
        self.noise = obs_cfg["noise"]

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

        # Add terrain
        self.horizontal_scale = self.terrain_cfg["horizontal_scale"]
        self.vertical_scale = self.terrain_cfg["vertical_scale"]
        self.height_field = cv2.imread(self.terrain_cfg["textures"], cv2.IMREAD_GRAYSCALE)
        if self.terrain_cfg["terrain"]:
            self.terrain = self.scene.add_entity(
                morph=gs.morphs.Terrain(
                    pos=(-4.2, -5.25, 0.0),
                    height_field=self.height_field,
                    horizontal_scale=self.horizontal_scale,
                    vertical_scale=self.vertical_scale,
                ),
            )

        # Init roboot quat and pos
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=self.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=self.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

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
        self.command_ranges[:, 2, 0] = self.command_cfg["leg_length_range"][0]
        self.command_ranges[:, 2, 1] = self.command_cfg["leg_length_range"][1]
        self.command_ranges[:, 3, 0] = self.command_cfg["leg_length_range"][0]
        self.command_ranges[:, 3, 1] = self.command_cfg["leg_length_range"][1]
        self.lin_vel_error = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.ang_vel_error = torch.zeros(self.num_envs, device=self.device, dtype=gs.tc_float)
        self.curriculum_step = 0

        # Initialize base state buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_lin_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        self.last_base_ang_vel = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)

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
        # Compute target DOF positions and velocities
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = (
            exec_actions[:, self.joint_dof_idx_np] * self.env_cfg["joint_action_scale"]
            + self.default_dof_pos[:, self.joint_dof_idx_np]
        )
        target_dof_vel = exec_actions[:, self.wheel_dof_idx_np] * self.env_cfg["wheel_action_scale"]

        # Apply DOF limits
        target_dof_pos = torch.clamp(
            target_dof_pos, min=self.dof_pos_lower[self.joint_dof_idx_np], max=self.dof_pos_upper[self.joint_dof_idx_np]
        )
        self.robot.control_dofs_position(target_dof_pos, self.joint_dof_idx)
        self.robot.control_dofs_velocity(target_dof_vel, self.wheel_dof_idx)

        # Step the scene
        self.scene.step()

        # Update base and joint state buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )

        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.dof_force[:] = self.robot.get_dofs_force(self.motors_dof_idx)

        # Add noise
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

        # Get contact force
        self.connect_force = self.robot.get_links_net_contact_force()

        # Update last state buffers
        self.last_base_lin_vel[:] = self.base_lin_vel[:]
        self.last_base_ang_vel[:] = self.base_ang_vel[:]

        # Update episode length
        self.episode_lengths += 1

        # Identify envs that need command resampling
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .flatten()
        )

        # compute curriculum
        self.lin_vel_error += torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
        self.ang_vel_error += torch.abs(self.commands[:, 1] - self.base_ang_vel[:, 2])

        # Check termination and reset
        if self.mode:
            self.check_termination()

        # Handle timeouts
        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).flatten()
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=self.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        if self.mode:
            self.reset_idx(self.reset_buf.nonzero(as_tuple=False).flatten())

        # Compute reward
        if self.mode:
            self.rew_buf[:] = 0.0
            for name, reward_func in self.reward_functions.items():
                rew = reward_func() * self.reward_scales[name]
                self.rew_buf += rew
                self.episode_sums[name] += rew

        # Resample commands
        if self.mode:
            self.resample_commands(envs_idx)
            self.curriculum_commands()

        # Construct current observation
        self.slice_obs_buf = torch.cat(
            [
                self.base_lin_vel * self.obs_scales["lin_vel"],
                self.base_ang_vel * self.obs_scales["ang_vel"],
                # self.projected_gravity,
                self.base_euler * self.obs_scales["base_euler"],
                self.commands * self.commands_scale,
                (self.dof_pos[:, self.joint_dof_idx_np] - self.default_dof_pos[:, self.joint_dof_idx_np])
                * self.obs_scales["dof_pos"],
                # self.dof_vel * self.obs_scales["dof_vel"],
                self.actions,
            ],
            axis=-1,
        )
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        # Combine the current observation with historical observations (e.g., along the time axis)
        self.obs_buf = torch.cat([self.history_obs_buf, self.slice_obs_buf.unsqueeze(1)], dim=1).view(self.num_envs, -1)

        # Update history buffer
        if self.history_length > 1:
            self.history_obs_buf[:, :-1, :] = self.history_obs_buf[:, 1:, :].clone()
        self.history_obs_buf[:, -1, :] = self.slice_obs_buf

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def check_termination(self):
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

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

        # Reset base orientation (quaternion)
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)

        # Apply base position and orientation
        self.base_pos[envs_idx] = self.base_init_pos
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)

        # Reset velocities
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
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

        # Resample new commands
        self.resample_commands(envs_idx)

        # Reset episode length
        self.episode_lengths[envs_idx] = 0.0

        # Reset curriculum
        self.lin_vel_error[envs_idx] = 0
        self.ang_vel_error[envs_idx] = 0

    def resample_commands(self, envs_idx):
        for idx in envs_idx:
            for command_idx in range(self.num_commands):
                low = self.command_ranges[idx, command_idx, 0]
                high = self.command_ranges[idx, command_idx, 1]
                self.commands[idx, command_idx] = gs_rand_float(low, high, (1,), self.device)
        # if self.survive_ratio < 0.7:
        #     self.commands[:, 2] = self.commands[:, 3]
        self.commands[:, 2] = self.commands[:, 3]

    def curriculum_commands(self):
        # Increment curriculum step counter
        self.curriculum_step += 1

        if self.curriculum_step >= self.curriculum_cfg["curriculum_step"]:
            self.curriculum_step = 0

            # Calculate mean linear and angular velocity errors
            valid_mask = self.episode_length_buf > 0
            if valid_mask.any():
                self.mean_lin_vel_error = (
                    (self.lin_vel_error[valid_mask] / self.episode_length_buf[valid_mask]).mean().item()
                )
                self.mean_ang_vel_error = (
                    (self.ang_vel_error[valid_mask] / self.episode_length_buf[valid_mask]).mean().item()
                )
            else:
                self.mean_lin_vel_error = float("nan")
                self.mean_ang_vel_error = float("nan")

            # ----- Linear velocity x-axis curriculum -----
            lin_err_high = 999
            if self.curriculum_cfg["err_mode"]:
                self.linx_range_up_threshold = self.curriculum_cfg["lin_vel_err_range"][0]
                lin_err_high = self.curriculum_cfg["lin_vel_err_range"][1]
            else:
                mean_linx_range = self.command_ranges[:, 0, 1].mean()
                ratio = mean_linx_range / self.command_cfg["lin_vel_x_range"][1]
                self.linx_range_up_threshold = (
                    self.curriculum_cfg["lin_vel_err_range"][0]
                    + (self.curriculum_cfg["lin_vel_err_range"][1] - self.curriculum_cfg["lin_vel_err_range"][0])
                    * ratio
                )
                lin_err_high = self.curriculum_cfg["lin_vel_err_range"][2]

            if self.survive_ratio > 0.9:
                old_lin_range = self.command_ranges[0, 0].clone().cpu().numpy()
                if self.mean_lin_vel_error < self.linx_range_up_threshold:
                    # Increase lin_vel_x command range
                    self.command_ranges[:, 0, 0] += (
                        self.curriculum_cfg["curriculum_lin_vel_step"] * self.command_cfg["lin_vel_x_range"][0]
                    )
                    self.command_ranges[:, 0, 1] += (
                        self.curriculum_cfg["curriculum_lin_vel_step"] * self.command_cfg["lin_vel_x_range"][1]
                    )
                    action = "↑ increased"
                elif self.mean_lin_vel_error > lin_err_high:
                    # Decrease lin_vel_x command range
                    self.command_ranges[:, 0, 0] -= (
                        self.curriculum_cfg["curriculum_lin_vel_step"] * self.command_cfg["lin_vel_x_range"][0]
                    )
                    self.command_ranges[:, 0, 1] -= (
                        self.curriculum_cfg["curriculum_lin_vel_step"] * self.command_cfg["lin_vel_x_range"][1]
                    )
                    action = "↓ decreased"
                else:
                    action = "→ unchanged"

                # Clamp the range within allowed bounds
                self.command_ranges[:, 0, 0] = torch.clamp(
                    self.command_ranges[:, 0, 0],
                    self.command_cfg["lin_vel_x_range"][0],
                    self.curriculum_cfg["curriculum_lin_vel_min_range"] * self.command_cfg["lin_vel_x_range"][0],
                )
                self.command_ranges[:, 0, 1] = torch.clamp(
                    self.command_ranges[:, 0, 1],
                    self.curriculum_cfg["curriculum_lin_vel_min_range"] * self.command_cfg["lin_vel_x_range"][1],
                    self.command_cfg["lin_vel_x_range"][1],
                )

                new_lin_range = self.command_ranges[0, 0].clone().cpu().numpy()
                print(
                    f"[Curriculum] lin_vel_x | error: {self.mean_lin_vel_error:.4f} | "
                    f"range: [{old_lin_range[0]:.3f}, {old_lin_range[1]:.3f}] → "
                    f"[{new_lin_range[0]:.3f}, {new_lin_range[1]:.3f}] | {action}"
                )

            # ----- Angular velocity yaw curriculum -----
            angv_err_high = 999
            if self.curriculum_cfg["err_mode"]:
                self.angv_range_up_threshold = self.curriculum_cfg["ang_vel_err_range"][0]
                angv_err_high = self.curriculum_cfg["ang_vel_err_range"][1]
            else:
                mean_angv_range = self.command_ranges[:, 1, 1].mean()
                ratio = mean_angv_range / self.command_cfg["ang_vel_range"][1]
                self.angv_range_up_threshold = (
                    self.curriculum_cfg["ang_vel_err_range"][0]
                    + (self.curriculum_cfg["ang_vel_err_range"][1] - self.curriculum_cfg["ang_vel_err_range"][0])
                    * ratio
                )
                angv_err_high = self.curriculum_cfg["ang_vel_err_range"][2]

            if self.survive_ratio > 0.9:
                old_ang_range = self.command_ranges[0, 1].clone().cpu().numpy()
                if self.mean_ang_vel_error < self.angv_range_up_threshold:
                    # Increase ang_vel_yaw command range
                    self.command_ranges[:, 1, 0] += (
                        self.curriculum_cfg["curriculum_ang_vel_step"] * self.command_cfg["ang_vel_range"][0]
                    )
                    self.command_ranges[:, 1, 1] += (
                        self.curriculum_cfg["curriculum_ang_vel_step"] * self.command_cfg["ang_vel_range"][1]
                    )
                    action = "↑ increased"
                elif self.mean_ang_vel_error > angv_err_high:
                    # Decrease ang_vel_yaw command range
                    self.command_ranges[:, 1, 0] -= (
                        self.curriculum_cfg["curriculum_ang_vel_step"] * self.command_cfg["ang_vel_range"][0]
                    )
                    self.command_ranges[:, 1, 1] -= (
                        self.curriculum_cfg["curriculum_ang_vel_step"] * self.command_cfg["ang_vel_range"][1]
                    )
                    action = "↓ decreased"
                else:
                    action = "→ unchanged"

                # Clamp the range within allowed bounds
                self.command_ranges[:, 1, 0] = torch.clamp(
                    self.command_ranges[:, 1, 0],
                    self.command_cfg["ang_vel_range"][0],
                    self.curriculum_cfg["curriculum_ang_vel_min_range"] * self.command_cfg["ang_vel_range"][0],
                )
                self.command_ranges[:, 1, 1] = torch.clamp(
                    self.command_ranges[:, 1, 1],
                    self.curriculum_cfg["curriculum_ang_vel_min_range"] * self.command_cfg["ang_vel_range"][1],
                    self.command_cfg["ang_vel_range"][1],
                )

                new_ang_range = self.command_ranges[0, 1].clone().cpu().numpy()
                print(
                    f"[Curriculum] ang_vel_yaw | error: {self.mean_ang_vel_error:.4f} | "
                    f"range: [{old_ang_range[0]:.3f}, {old_ang_range[1]:.3f}] → "
                    f"[{new_ang_range[0]:.3f}, {new_ang_range[1]:.3f}] | {action}"
                )

    # ------------ reward functions----------------
    def reward_tracking_lin_x_vel(self):
        # Tracking of linear velocity commands (x axes)
        lin_vel_error = torch.abs(self.commands[:, 0] - self.base_lin_vel[:, 0])
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

    def reward_tracking_base_euler(self):
        # Reward for aligning roll/pitch to target angles (degrees)
        roll_err = self.base_euler[:, 0] - self.reward_cfg["target_roll"]
        pitch_err = self.base_euler[:, 1] - self.reward_cfg["target_pitch"]
        angle_error = roll_err**2 + pitch_err**2
        return torch.exp(-angle_error / (self.reward_cfg["tracking_base_euler_sigma"] ** 2))

    def reward_tracking_leg_length(self):
        # Tracking of leg length (hip to knee)
        knee_error = torch.square(self.dof_pos[:, 1] - self.commands[:, 2])
        knee_error += torch.square(self.dof_pos[:, 4] - self.commands[:, 3])
        return knee_error

    def reward_similar_leg(self):
        # Penalize difference between left and right leg lengths
        leg_error = torch.square(self.dof_pos[:, 0] - self.dof_pos[:, 3])
        leg_error += torch.square(self.dof_pos[:, 1] - self.dof_pos[:, 4])
        return leg_error

    def reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

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
