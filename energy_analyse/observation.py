import numpy as np
import mujoco
from .math_utils import quat_conjugate, quat_rotate, quat_mul, matrix_from_quat

def compute_observations(data, model, ref_data, torso_id, q_ref_full, dq_ref_full, last_action, default_joint_pos_array, obs_qpos_indices, obs_qvel_indices, wo_state_est=False):
    """
    Construct the observation vector.
    wo_state_est: If True, remove motion_anchor_pos_b (3) and base_lin_vel (3)
    """

    # Update Reference Kinematics for Anchor (Torso)
    # We must use a separate MjData structure (ref_data) to compute FK for reference pose
    # Note: This is potentially expensive but necessary for correct Anchor tracking if Anchor != Root
    ref_data.qpos[:] = 0 # Clear
    
    # 1. Map Reference qpos (ONNX Order) to MuJoCo qpos (Physical Order) for FK
    # q_ref_full contains [Root(7), Joints(N)]
    # We need to put this into ref_data.qpos which expects [Root(7), Joints_Physical(N)]
    
    ref_root_pos = q_ref_full[:3]
    ref_root_quat = q_ref_full[3:7]
    ref_data.qpos[:3] = ref_root_pos
    ref_data.qpos[3:7] = ref_root_quat
    
    ref_joint_pos_onnx = q_ref_full[7:]
    
    # Scatter Ref Joints (ONNX) to Ref Data (Physical)
    # obs_qpos_indices[i] tells us the qpos address for ONNX joint i
    ref_data.qpos[obs_qpos_indices] = ref_joint_pos_onnx
    
    mujoco.mj_kinematics(model, ref_data)
    
    # 1. Robot State
    # Anchor: Torso
    robot_anchor_pos = data.xpos[torso_id]
    robot_anchor_quat = data.xquat[torso_id] # xquat gives body orientation
    robot_anchor_quat = np.array([robot_anchor_quat[0], robot_anchor_quat[1], robot_anchor_quat[2], robot_anchor_quat[3]]) # w,x,y,z
    
    # Base/Root for velocity (usually still referenced to base, or anchor?)
    # Config says: base_lin_vel. This is usually the Root Body.
    root_lin_vel = data.qvel[:3]
    root_ang_vel = data.qvel[3:6]
    root_quat = data.qpos[3:7]
    
    joint_pos = data.qpos[obs_qpos_indices]
    joint_vel = data.qvel[obs_qvel_indices]
    
    # 2. Reference State
    ref_anchor_pos = ref_data.xpos[torso_id]
    ref_anchor_quat = ref_data.xquat[torso_id]
    ref_anchor_quat = np.array([ref_anchor_quat[0], ref_anchor_quat[1], ref_anchor_quat[2], ref_anchor_quat[3]])
    
    ref_joint_pos = q_ref_full[7:]
    ref_joint_vel = dq_ref_full[6:]
    
    # -----------------------------------------------------------------------
    # Observe: command (Ref Joint Pos (ABSOLUTE) + Ref Joint Vel)
    # -----------------------------------------------------------------------
    # WHOLE_BODY_TRACKING Check: 
    # commands.py returns torch.cat([self.joint_pos, self.joint_vel]).
    # joint_pos is raw (Absolute).
    # Previous fix (Relative) is likely WRONG if checks valid absolute.
    obs_command = np.concatenate([ref_joint_pos, ref_joint_vel]) 
    
    # -----------------------------------------------------------------------
    # Observe: motion_anchor_pos_b (Ref Anchor Pos in Robot Anchor Frame)
    # -----------------------------------------------------------------------
    # pos_diff_world = ref_anchor_pos - robot_anchor_pos
    # obs = rotate_inverse(robot_anchor_quat, pos_diff_world)
    pos_diff_world = ref_anchor_pos - robot_anchor_pos
    robot_anchor_quat_inv = quat_conjugate(robot_anchor_quat)
    obs_anchor_pos = quat_rotate(robot_anchor_quat_inv, pos_diff_world)
    
    # -----------------------------------------------------------------------
    # Observe: motion_anchor_ori_b (Ref Anchor Ori in Robot Anchor Frame)
    # -----------------------------------------------------------------------
    obs_anchor_quat_diff = quat_mul(robot_anchor_quat_inv, ref_anchor_quat)
    mat = matrix_from_quat(obs_anchor_quat_diff)
    obs_anchor_rot6d = mat[:, :2].flatten() 
    
    # -----------------------------------------------------------------------
    # Observe: base_lin_vel (in Base Frame)
    # -----------------------------------------------------------------------
    # Assuming standard IsaacLab: Rotate World Lin Vel to Base Frame
    # Use Root Quat for "Base" (Pelvis)
    root_quat_inv = quat_conjugate(root_quat)
    obs_base_lin_vel = quat_rotate(root_quat_inv, root_lin_vel)
    
    # -----------------------------------------------------------------------
    # Observe: base_ang_vel
    # -----------------------------------------------------------------------
    # Assuming MuJoCo qvel 3:6 is local angular velocity (Standard)
    obs_base_ang_vel = root_ang_vel 
    
    # -----------------------------------------------------------------------
    # Observe: joint_pos (Relative to Default)
    # -----------------------------------------------------------------------
    # Logic: obs = current - default
    obs_joint_pos = joint_pos - default_joint_pos_array
    
    # -----------------------------------------------------------------------
    # Observe: joint_vel
    # -----------------------------------------------------------------------
    obs_joint_vel = joint_vel
    
    # -----------------------------------------------------------------------
    # Observe: actions (Previous)
    # -----------------------------------------------------------------------
    obs_actions = last_action
    
    # Concatenate
    
    if wo_state_est:
        # 154-dim: Remove anchor_pos (index 1) and base_lin_vel (index 3)
        obs = np.concatenate([
            obs_command,          # 58
            # obs_anchor_pos,     # REMOVED
            obs_anchor_rot6d,     # 6
            # obs_base_lin_vel,   # REMOVED
            obs_base_ang_vel,     # 3
            obs_joint_pos,        # 29
            obs_joint_vel,        # 29
            obs_actions           # 29
        ])
    else:
        # 160-dim: Full State
        obs = np.concatenate([
            obs_command,
            obs_anchor_pos,
            obs_anchor_rot6d,
            obs_base_lin_vel,
            obs_base_ang_vel,
            obs_joint_pos,
            obs_joint_vel,
            obs_actions
        ])
    
    return obs.astype(np.float32)
