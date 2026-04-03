import joblib
import torch
from pathlib import Path
from collections import defaultdict

from . import math_utils


class MotionData:
    def __init__(
        self,
        motion_data_dir: str,
        motion_data_weights: dict[str, float] | None = None,
        device: str = "cpu",
    ):
        self.motion_data_dir = motion_data_dir
        self.motion_data_weights = motion_data_weights
        self.joint_names = None
        self.body_names = None
        self.device = device
        self._load_motion_data()

    def _load_motion_data(self):
        motion_data_dir = Path(self.motion_data_dir)
        if not motion_data_dir.exists():
            raise ValueError(
                f"Motion data directory {str(motion_data_dir)} does not exist."
            )
        motion_files = list(motion_data_dir.rglob("*.pkl"))
        if len(motion_files) == 0:
            raise ValueError(
                f"No motion data files with .pkl extension found in {str(motion_data_dir)}"
            )
        motion_name2path = {p.stem: p for p in motion_files}

        if self.motion_data_weights is None:
            print(
                "⚠️ Did not specify the motion data weights, load all with weight 1.0!"
            )
            self.motion_data_weights = {f.stem: 1.0 for f in motion_files}

        # Load motion data
        self.motion_durations = []
        self.motion_fps = []
        self.motion_dt = []
        self.motion_num_frames = []
        self.motion_weights = []

        self.root_pos_w = []
        self.root_quat_w = []
        self.root_lin_vel_w = []
        self.root_ang_vel_w = []
        self.joint_pos = []
        self.joint_vel = []
        self.body_pos_b = []

        # only load the motion data files that are in the motion weights dict
        for motion_name, motion_weight in self.motion_data_weights.items():
            if motion_weight <= 0:
                continue
            # check if the motion file name is valid
            if motion_name not in motion_name2path.keys():
                raise ValueError(
                    f"Motion name {motion_name} defined in motion weights not found in motion data directory {str(motion_data_dir)}. Available names: {motion_name2path.keys()}"
                )

            # load the motion data file
            motion_path = motion_name2path[motion_name]
            print(
                f"[Motion Data Manager] Loading motion data from {str(motion_path)}..."
            )
            motion_raw_data = joblib.load(str(motion_path))
            if not isinstance(motion_raw_data, dict):
                raise ValueError(
                    f"Motion data file {str(motion_path)} does not contain a valid dictionary."
                )

            num_frames = len(motion_raw_data["root_pos_w"])
            if num_frames < 2:
                raise ValueError(
                    f"[MotionLoader] Motion has only {num_frames} frames, cannot compute velocity."
                )

            fps = motion_raw_data["fps"]
            root_pos_w = torch.from_numpy(motion_raw_data["root_pos_w"]).to(self.device)
            root_quat_w = torch.from_numpy(motion_raw_data["root_quat_w"]).to(
                self.device
            )  # w,x,y,z
            joint_pos = torch.from_numpy(motion_raw_data["joint_pos"]).to(self.device)
            body_pos_b = torch.from_numpy(motion_raw_data["body_pos_b"]).to(self.device)
            if not self.body_names:
                self.body_names = motion_raw_data["body_names"]
            if not self.joint_names:
                self.joint_names = motion_raw_data["joint_names"]

            # Calculate vel
            dt = 1.0 / fps

            root_lin_vel_w = torch.zeros_like(root_pos_w)
            root_lin_vel_w[:-1] = (root_pos_w[1:] - root_pos_w[:-1]) / dt
            root_lin_vel_w[-1] = root_lin_vel_w[-2]

            root_ang_vel_w = torch.zeros_like(root_pos_w)
            root_ang_vel_w[:-1] = (
                math_utils.quat_apply(
                    root_quat_w[:1],
                    math_utils.quat_box_minus(root_quat_w[1:], root_quat_w[:-1]),
                )
                / dt
            )
            root_ang_vel_w[-1] = root_ang_vel_w[-2]

            joint_vel = torch.zeros_like(joint_pos)
            joint_vel[:-1] = (joint_pos[1:] - joint_pos[:-1]) / dt
            joint_vel[-1] = joint_vel[-2]

            # Add motion data
            self.motion_durations.append(num_frames * dt)
            self.motion_fps.append(fps)
            self.motion_dt.append(dt)
            self.motion_num_frames.append(num_frames)
            self.motion_weights.append(motion_weight)

            self.root_pos_w.append(root_pos_w)
            self.root_quat_w.append(root_quat_w)
            self.root_lin_vel_w.append(root_lin_vel_w)
            self.root_ang_vel_w.append(root_ang_vel_w)
            self.joint_pos.append(joint_pos)
            self.joint_vel.append(joint_vel)
            self.body_pos_b.append(body_pos_b)

        self.motion_durations = torch.tensor(
            self.motion_durations, dtype=torch.float, device=self.device
        )
        self.motion_fps = torch.tensor(
            self.motion_fps, dtype=torch.float, device=self.device
        )
        self.motion_dt = torch.tensor(
            self.motion_dt, dtype=torch.float, device=self.device
        )
        self.motion_num_frames = torch.tensor(
            self.motion_num_frames, dtype=torch.long, device=self.device
        )
        self.motion_weights = torch.tensor(
            self.motion_weights, dtype=torch.float, device=self.device
        )

        self.root_pos_w = torch.cat(self.root_pos_w)
        self.root_quat_w = torch.cat(self.root_quat_w)
        self.root_lin_vel_w = torch.cat(self.root_lin_vel_w)
        self.root_ang_vel_w = torch.cat(self.root_ang_vel_w)
        self.joint_pos = torch.cat(self.joint_pos)
        self.joint_vel = torch.cat(self.joint_vel)
        self.body_pos_b = torch.cat(self.body_pos_b)

        # Some other information
        self.num_joints = self.joint_pos.shape[-1]
        self.num_bodies = self.body_pos_b.shape[-2]

        self.motion_ids = torch.arange(
            len(self.motion_durations), dtype=torch.long, device=self.device
        )

        lengths_shifted = self.motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self.motion_start_indices = torch.cumsum(lengths_shifted, dim=0)
        print(f"✅ Load motion data successfully on device {self.device}!")

    def sample_motion_ids(self, n: int) -> torch.Tensor:
        return torch.multinomial(self.motion_weights, num_samples=n, replacement=True)

    def sample_motion_times(
        self,
        motion_ids: torch.Tensor,
        truncate_time_start: float | None = None,
        truncate_time_end: float | None = None,
    ) -> torch.Tensor:
        motion_durations = self.motion_durations[motion_ids]

        # Calculate valid time range
        time_start = torch.zeros_like(motion_durations)
        time_end = motion_durations.clone()

        if truncate_time_start is not None:
            assert (
                truncate_time_start >= 0
            ), f"[MotionLoader] truncate_time_start must be non-negative, but got {truncate_time_start}."
            time_start = torch.clamp(
                time_start + truncate_time_start, min=0.0, max=motion_durations
            )

        if truncate_time_end is not None:
            assert (
                truncate_time_end >= 0
            ), f"[MotionLoader] truncate_time_end must be non-negative, but got {truncate_time_end}."
            time_end = torch.clamp(time_end - truncate_time_end, min=0.0)

        # Check if valid range exists
        valid_range = time_end - time_start
        if torch.any(valid_range <= 0.0):
            print(
                "[Warning] Some motions have invalid time range after truncation (start >= end)."
            )
            valid_range = torch.clamp(valid_range, min=1e-6)  # Prevent division by zero

        # Sample time within the valid range
        phase = torch.rand(motion_ids.shape, device=self.device)
        sample_times = time_start + phase * valid_range

        return sample_times

    def sample_motion_seq_times(
        self, motion_ids: torch.Tensor, n_steps: int, dt: float
    ) -> torch.Tensor:
        motion_seq_duration = n_steps * dt
        start_times = self.sample_motion_times(
            motion_ids, truncate_time_end=motion_seq_duration
        )  # (ids,)
        motion_seq_times = (
            start_times.reshape(-1, 1)
            + torch.arange(n_steps, device=self.device).reshape(1, -1) * dt
        )  # (ids, steps)
        return motion_seq_times

    def get_motion_data(
        self,
        motion_ids: torch.Tensor,
        motion_times: torch.Tensor,
        joint_names: list | None = None,
        body_names: list | None = None,
    ) -> dict[str, torch.Tensor]:
        if motion_ids.shape != motion_times.shape:
            raise ValueError(
                f"motion_ids shape {motion_ids.shape} should be equal with motion_times shape {motion_times.shape}"
            )

        phase = motion_times / self.motion_durations[motion_ids]
        num_frames = self.motion_num_frames[motion_ids]

        frame_idx0 = torch.floor((phase * (num_frames - 1))).long()
        frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
        blend = (phase * (num_frames - 1) - frame_idx0).reshape(-1, 1)

        frame_idx0 += self.motion_start_indices[motion_ids]
        frame_idx1 += self.motion_start_indices[motion_ids]

        root_pos_w_0 = self.root_pos_w[frame_idx0]
        root_pos_w_1 = self.root_pos_w[frame_idx1]
        root_quat_w_0 = self.root_quat_w[frame_idx0]
        root_quat_w_1 = self.root_quat_w[frame_idx1]
        root_lin_vel_w_0 = self.root_lin_vel_w[frame_idx0]
        root_lin_vel_w_1 = self.root_lin_vel_w[frame_idx1]
        root_ang_vel_w_0 = self.root_ang_vel_w[frame_idx0]
        root_ang_vel_w_1 = self.root_ang_vel_w[frame_idx1]
        joint_pos_0 = self.joint_pos[frame_idx0]
        joint_pos_1 = self.joint_pos[frame_idx1]
        joint_vel_0 = self.joint_vel[frame_idx0]
        joint_vel_1 = self.joint_vel[frame_idx1]
        body_pos_b_0 = self.body_pos_b[frame_idx0]
        body_pos_b_1 = self.body_pos_b[frame_idx1]

        # interpolate the values
        # root_quat_w = torch.cat([math_utils.quat_slerp(q0, q1, tau) for q0, q1, tau in zip(root_quat_w_0, root_quat_w_1, blend)]).float()
        root_quat_w = math_utils.quat_slerp(root_quat_w_0, root_quat_w_1, blend).float()

        root_pos_w = torch.lerp(root_pos_w_0, root_pos_w_1, blend)
        root_lin_vel_w = torch.lerp(root_lin_vel_w_0, root_lin_vel_w_1, blend)
        root_lin_vel_b = math_utils.quat_apply_inverse(root_quat_w, root_lin_vel_w)
        root_ang_vel_w = torch.lerp(root_ang_vel_w_0, root_ang_vel_w_1, blend)
        root_ang_vel_b = math_utils.quat_apply_inverse(root_quat_w, root_ang_vel_w)
        joint_pos = torch.lerp(joint_pos_0, joint_pos_1, blend)
        joint_vel = torch.lerp(joint_vel_0, joint_vel_1, blend)
        body_pos_b = torch.lerp(body_pos_b_0, body_pos_b_1, blend.reshape(-1, 1, 1))

        if joint_names:
            joint2isaac_idxs = torch.tensor(
                [self.joint_names.index(name) for name in joint_names],
                dtype=torch.long,
                device=self.device,
            )
            joint_pos = joint_pos[:, joint2isaac_idxs]
            joint_vel = joint_vel[:, joint2isaac_idxs]
        if body_names:
            body2isaac_idxs = torch.tensor(
                [self.body_names.index(name) for name in body_names],
                dtype=torch.long,
                device=self.device,
            )
            body_pos_b = body_pos_b[:, body2isaac_idxs]

        return {
            "root_pos_w": root_pos_w,
            "root_quat_w": root_quat_w,
            "root_lin_vel_w": root_lin_vel_w,
            "root_lin_vel_b": root_lin_vel_b,
            "root_ang_vel_w": root_ang_vel_w,
            "root_ang_vel_b": root_ang_vel_b,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_b": body_pos_b,
        }

    def get_motion_seq_data(
        self,
        motion_ids: torch.Tensor,
        motion_seq_times: torch.Tensor,
        joint_names: list = None,
        body_names: list = None,
    ) -> dict[str, torch.Tensor]:
        motion_seq_data = defaultdict(list)
        for seq in range(motion_seq_times.shape[-1]):
            motion_state = self.get_motion_data(
                motion_ids, motion_seq_times[:, seq], joint_names, body_names
            )
            for k, v in motion_state.items():
                motion_seq_data[k].append(v)
        for k, v in motion_seq_data.items():
            motion_seq_data[k] = torch.stack(v, dim=1)
        return dict(motion_seq_data)


if __name__ == "__main__":
    # Test
    device = "cuda:0"
    # device = 'cpu'
    motion_data = MotionData(
        motion_data_dir="/home/robot/hongtu/SimpleAMP/robot_assets/ths_23dof/motion_data",
        device=device,
    )
    for _ in range(3):
        motion_ids = motion_data.sample_motion_ids(4096)
        motion_seq_times = motion_data.sample_motion_seq_times(
            motion_ids=motion_ids, n_steps=3, dt=0.1
        )
        motion_seq_data = motion_data.get_motion_seq_data(motion_ids, motion_seq_times)
        amp_input = torch.cat(
            [
                motion_seq_data["root_lin_vel_b"],
                motion_seq_data["root_ang_vel_b"],
                motion_seq_data["joint_pos"],
                motion_seq_data["joint_vel"],
            ],
            dim=-1,
        ).to(
            "cuda:0"
        )  # (num_envs, n_steps, d)
