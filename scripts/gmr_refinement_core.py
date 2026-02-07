import pickle
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import os
import sys

# Ensure GMR module can be found
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from general_motion_retargeting import GeneralMotionRetargeting as GMR

def load_motion(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    elif path.endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        # Convert to dict if needed, or handle npz object
        return {k: data[k] for k in data.files}
    else:
        raise ValueError("Unsupported file format. Please use .pkl or .npz")

def save_motion(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.endswith(".pkl"):
        with open(path, "wb") as f:
            pickle.dump(data, f)
    elif path.endswith(".npz"):
        np.savez(path, **data)
    print(f"Motion saved to: {path}")

def run_refinement(robot_name, input_path, output_path, modify_func, post_fk_func=None):
    """
    Core function to run the GMR refinement process.
    
    Args:
        robot_name (str): The robot name (e.g. 'unitree_g1')
        input_path (str): Path to input motion file
        output_path (str): Path to output motion file
        modify_func (callable): Function taking (frame_idx, root_pos, root_rot, dof_pos) 
                                and returning modified (root_pos, root_rot, dof_pos)
        post_fk_func (callable): Function taking (frame_idx, source_data) and returning modified source_data
                                 Run after FK computation but before GMR IK.
    """
    
    # 1. Load Data
    print(f"Loading motion from {input_path}...")
    motion_data = load_motion(input_path)
    
    # [Patch] Support raw 'qpos' (e.g. from Mujoco recordings) variables
    if "qpos" in motion_data and "root_pos" not in motion_data:
        print("Detected raw 'qpos', splitting into root/dof...")
        qpos = motion_data["qpos"]
        # Assuming [N, 7+D]
        motion_data["root_pos"] = qpos[:, :3]
        # Mujoco qpos is wxyz -> GMR wants xyzw?
        # Code line 108: `curr_root_rot_wxyz = np.array([curr_root_rot[3], curr_root_rot[0], ...])`
        # means if input is `curr_root_rot`=[x,y,z,w], then [3] is w, [0] is x.
        # So GMR wants xyzw internal.
        # Mujoco 'qpos' usually has wxyz.
        # wxyz -> xyzw: [1,2,3,0]
        root_rot_wxyz = qpos[:, 3:7]
        motion_data["root_rot"] = root_rot_wxyz[:, [1, 2, 3, 0]]
        motion_data["dof_pos"] = qpos[:, 7:]
    
    try:
        root_pos_arr = motion_data["root_pos"]
        root_rot_arr = motion_data["root_rot"] # Usually xyzw
        dof_pos_arr = motion_data["dof_pos"]
        fps = motion_data.get("fps", 30)
    except KeyError as e:
        print(f"Error: Missing key {e} in motion file.")
        return

    num_frames = len(root_pos_arr)
    print(f"Processing {num_frames} frames for robot: {robot_name}")

    # 2. Init GMR
    # distinct source name to avoid confusion, though logic overrides it
    print("Initializing GMR engine...")
    gmr = GMR(src_human="xrobot", tgt_robot=robot_name, verbose=False)
    
    # Disable internal offsets/scaling for 1:1 Clean-up mode
    for k in gmr.pos_offsets1: gmr.pos_offsets1[k] = np.zeros(3)
    for k in gmr.rot_offsets1: gmr.rot_offsets1[k] = R.from_quat([0, 0, 0, 1])
    for k in gmr.human_scale_table: gmr.human_scale_table[k] = 1.0
    
    # Setup auxiliary FK model
    fk_model = gmr.model
    fk_data = mj.MjData(fk_model)
    
    # Map Robot Links -> Source Targets
    robot_link_to_source = {}
    for robot_link, entry in gmr.ik_match_table1.items():
        robot_link_to_source[robot_link] = entry[0]
    
    new_root_pos = []
    new_root_rot = []
    new_dof_pos = []
    
    print("Running refinement loop...")
    for i in tqdm(range(num_frames)):
        
        # Get frame data
        curr_root_pos = root_pos_arr[i]
        curr_root_rot = root_rot_arr[i]
        curr_dof_pos = dof_pos_arr[i]
        
        # --- USER MODIFICATION ---
        curr_root_pos, curr_root_rot, curr_dof_pos = modify_func(i, curr_root_pos, curr_root_rot, curr_dof_pos)
        # -------------------------
        
        # Verify Modification Integrity
        if curr_dof_pos.shape != dof_pos_arr[i].shape:
             print(f"Warning: DOF shape changed at frame {i}!")
        
        # Compute FK for Modified State
        # Rot: xyzw -> wxyz
        curr_root_rot_wxyz = np.array([curr_root_rot[3], curr_root_rot[0], curr_root_rot[1], curr_root_rot[2]])
        full_qpos = np.concatenate([curr_root_pos, curr_root_rot_wxyz, curr_dof_pos])
        
        fk_data.qpos[:] = full_qpos
        mj.mj_kinematics(fk_model, fk_data)
        
        # Extract Constraints for IK
        source_data = {}
        for robot_link, source_bone in robot_link_to_source.items():
            link_id = mj.mj_name2id(fk_model, mj.mjtObj.mjOBJ_BODY, robot_link)
            if link_id != -1:
                pos = fk_data.xpos[link_id].copy()
                mat = fk_data.xmat[link_id].reshape(3, 3).copy()
                r = R.from_matrix(mat)
                q_xyzw = r.as_quat()
                q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
                source_data[source_bone] = (pos, q_wxyz)
        
        # --- POST FK MODIFICATION ---
        if post_fk_func is not None:
             source_data = post_fk_func(i, source_data)
        # ----------------------------

        # Run GMR IK
        clean_qpos = gmr.retarget(source_data, offset_to_ground=False)
        
        # Extract Cleaned Data
        res_root_pos = clean_qpos[:3]
        res_root_rot_wxyz = clean_qpos[3:7]
        res_dof_pos = clean_qpos[7:]
        
        # wxyz -> xyzw
        res_root_rot_xyzw = np.array([res_root_rot_wxyz[1], res_root_rot_wxyz[2], res_root_rot_wxyz[3], res_root_rot_wxyz[0]])
        
        new_root_pos.append(res_root_pos)
        new_root_rot.append(res_root_rot_xyzw)
        new_dof_pos.append(res_dof_pos)
        
    # Stack and Save
    output_data = {
        "fps": fps,
        "root_pos": np.array(new_root_pos),
        "root_rot": np.array(new_root_rot),
        "dof_pos": np.array(new_dof_pos)
    }
    
    save_motion(output_path, output_data)
