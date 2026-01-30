"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Franka Cube Pick
----------------
Use Jacobian matrix and inverse kinematics control of Franka robot to pick up a box.
Damped Least Squares method from: https://www.math.ucsd.edu/~sbuss/ResearchWeb/ikmethods/iksurvey.pdf
"""

# conda activate isaac
# python unitree_h1_retargeting_lsk.py
# import keyboard
import os
import sys
import numpy as np
import time
from tqdm import tqdm
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
import trimesh
import torch
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import axis_angle_to_quaternion
import pytorch_kinematics as pk
from diff_quat import quat_from_matrix
import open3d as o3d


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": "--controller", "type": str, "default": "ik",
     "help": "Controller to use for Franka. Options are {ik, osc}"},
    {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"},
    {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
    {"name": "--num_envs", "type": int, "default": 256, "help": "Number of environments to create"},
]
args = gymutil.parse_arguments(
    description="test",
    custom_parameters=custom_parameters,
)

# set torch device
# ipdb.set_trace()
device = args.sim_device if args.use_gpu_pipeline else 'cpu'

######################## configure sim ########################
sim_params = gymapi.SimParams()
sim_fps = 5
sim_params.dt = dt = 1.0 / sim_fps

# set ground normal
gymutil.parse_sim_config({"gravity": [0.0, 0.0, -9.81], "up_axis": 1}, sim_params)  # 0 is y, 1 is z

if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")
###############################################################


# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

show = True
save = True
save_fps = 10
if show:
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise Exception("Failed to create viewer")

asset_root = "../assets"
# asset_root = "/home/liuyun/Humanoid_IL_Benchmark/humanplus/HST/legged_gym/resources/robots/h1"

# load h1 asset
h1_asset_file = "h1_description/urdf/h1.urdf"
asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = False  # 用dae文件则是True
h1_asset = gym.load_asset(sim, asset_root, h1_asset_file, asset_options)
h1_dof_names = gym.get_asset_dof_names(h1_asset)

h1_dof_props = gym.get_asset_dof_properties(h1_asset)
h1_num_dofs = gym.get_asset_dof_count(h1_asset)
h1_dof_states = np.zeros(h1_num_dofs, dtype=gymapi.DofState.dtype)
h1_dof_types = [gym.get_asset_dof_type(h1_asset, i) for i in range(h1_num_dofs)]
h1_dof_positions = h1_dof_states['pos']
h1_lower_limits = h1_dof_props["lower"]
h1_upper_limits = h1_dof_props["upper"]
h1_ranges = h1_upper_limits - h1_lower_limits
h1_mids = 0.3 * (h1_upper_limits + h1_lower_limits)
h1_stiffnesses = h1_dof_props['stiffness']
h1_dampings = h1_dof_props['damping']
h1_armatures = h1_dof_props['armature']
h1_has_limits = h1_dof_props['hasLimits']
h1_dof_props['hasLimits'] = np.array([True]*h1_num_dofs)

num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

envs = []
actor_handles = []
object_actor_handles = []
joint_handles = {}

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# load h1 motion data
motion_data_path = "/home/liuyun/Humanoid_IL_Benchmark/retargeting/CORE4D_lift_box/box001_000/h1_kinematic_motions/20231108_123_068_112_F_2_data.npz"
motion_data_overall = np.load(motion_data_path, allow_pickle=True)["arr_0"].item()
motion_data = motion_data_overall["joint_angles"]
motion_global_rotations = motion_data_overall["global_rotations"]
motion_global_translations = motion_data_overall["global_translations"]
obj_mesh = motion_data_overall["object_mesh"]
obj_global_poses = motion_data_overall["object_poses"]  # (N_frame, 4, 4)

# prepare object
object_asset = gym.load_asset(sim, os.path.dirname(os.path.dirname(motion_data_path)), "scene_mesh.urdf", gymapi.AssetOptions())
object_list = []
object_collision_mesh = trimesh.Trimesh(vertices=np.float32(obj_mesh["vertices"]), faces=np.int32(obj_mesh["faces"]))
object_vertices, object_faces = np.float32(object_collision_mesh.vertices).copy(), np.uint32(object_collision_mesh.faces).copy()
tm_params = gymapi.TriangleMeshParams()
tm_params.nb_vertices = object_vertices.shape[0]
tm_params.nb_triangles = object_faces.shape[0]
object_list.append({"vertices": object_vertices, "faces": object_faces, "tm_params": tm_params})

env_object_ids = np.random.randint(0, len(object_list), num_envs)
assert num_envs == 1
env_origins = torch.zeros(num_envs, 3, device=device, requires_grad=False)
# object info
object_center_position = torch.zeros(num_envs, 3, dtype=torch.float, device=device, requires_grad=False)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add actor
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 0.0, 1.05)
    pose.r = gymapi.Quat(0, 0, 0, 1)
    actor_handle = gym.create_actor(env, h1_asset, pose, "actor", i, 1, 0)
    actor_handles.append(actor_handle)

    # add interacted object
    object_initial_pose = gymapi.Transform()
    object_initial_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
    object_initial_pose.r = gymapi.Quat(0, 0, 0, 1)
    object_actor_handle = gym.create_actor(env, object_asset, object_initial_pose, "object", i, 0, 0)
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])
    gym.set_rigid_body_color(env, object_actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
    object_actor_handles.append(object_actor_handle)

    # set default DOF positions
    gym.set_actor_dof_states(env, actor_handle, h1_dof_states, gymapi.STATE_ALL)

# position the camera
if show:
    # right view
    # cam_pos = gymapi.Vec3(3, 2.0, 0)
    # cam_target = gymapi.Vec3(-3, 0, 0)
    cam_pos = gymapi.Vec3(0, -3, 2.0)
    cam_target = gymapi.Vec3(0, 3, 0)
    # front view
    # cam_pos = gymapi.Vec3(0, 2.0, -2)
    # cam_target = gymapi.Vec3(0, 0, 2)
    gym.viewer_camera_look_at(viewer, envs[0], cam_pos, cam_target)

gym.prepare_sim(sim)     
for i in tqdm(range(motion_data.shape[0])):
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # set global pose
    t = torch.from_numpy(motion_global_translations[i])
    q_wxyz = axis_angle_to_quaternion(torch.from_numpy(motion_global_rotations[i]))  # (w, x, y, z)
    q_xyzw = torch.tensor([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).to(dtype=torch.float32)  # (x, y, z, w)
    actor_root_state = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(actor_root_state)  # (2, 13)
    
    root_states[0, :3] = t
    root_states[0, 3:7] = q_xyzw
    root_states[1, :3] = torch.from_numpy(obj_global_poses[i, :3, 3])
    root_states[1, 3:7] = quat_from_matrix(torch.from_numpy(obj_global_poses[i, :3, :3]))
    root_reset_actors_indices = torch.tensor([
        gym.get_actor_index(envs[0], actor_handles[0], gymapi.DOMAIN_SIM),
        gym.get_actor_index(envs[0], object_actor_handles[0], gymapi.DOMAIN_SIM),
    ]).to(dtype=torch.int32)
    # print(root_states, root_reset_actors_indices)
    gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(root_reset_actors_indices), len(root_reset_actors_indices))

    # set joint angles
    for j in range(motion_data.shape[1]):
        h1_dof_positions[j] = motion_data[i, j]
    gym.set_actor_dof_states(envs[0], actor_handles[0], h1_dof_states, gymapi.STATE_POS)

    # humanoid pose
    # print("pelvis global pose =", pose.p, pose.r)
    print("19DoF local poses =", h1_dof_positions)
    # print("19DoF joint names =", h1_dof_names)
    # joint_pose = gym.get_actor_joint_transforms(envs[0], actor_handles[0])  # len = 24, item: (3D translation vector P, 4D quaternion Q (x, y, z, w)), in world space, 含义是：沿这个关节的转动 = 沿世界系的P处的frame Q的x轴正向的转动
    # joint_names = gym.get_actor_joint_names(envs[0], actor_handles[0])  # 'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 'left_knee_joint', 'left_ankle_joint', 'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint', 'right_knee_joint', 'right_ankle_joint', 'torso_joint', 'd435_left_imager_joint', 'd435_rgb_module_joint', 'imu_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'logo_joint', 'mid360_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint']
    # print("24D joint global poses =", joint_pose, len(joint_pose), joint_pose[20][0])
    # print("24D joint names =", joint_names)

    if show:
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.clear_lines(viewer)
    gym.sync_frame_time(sim)

    if show and gym.query_viewer_has_closed(viewer):
        break

if show:
    gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

