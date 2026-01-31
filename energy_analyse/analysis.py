import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort
import os
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

from .g1_config import get_action_scale, get_default_pos
from .observation import compute_observations
from .g1_physics import apply_g1_physics

def run_onnx_energy_analysis(xml_path, onnx_path, npz_path, output_dir=None, sim_dt=0.005, control_dt=0.02, record_video=True, npz_is_onnx_order=False):
    # Setup Output Directory
    if output_dir is None:
        # Create a 'result' folder in the script directory or parent
        # Adjusted to be relative to where the analysis is run or just CWD
        output_dir = os.path.join(os.getcwd(), "result")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Loading MuJoCo Model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    # [Physics Fix] Apply Training Physics Parameters (Armature/Damping)
    apply_g1_physics(model)
    
    data = mujoco.MjData(model)
    
    # Renderer Setup
    renderer = None
    frames = []
    video_fps = 30.0
    last_render_time = -1.0
    
    if record_video:
        try:
            renderer = mujoco.Renderer(model, height=720, width=1280)
            print("Video recording enabled.")
        except Exception as e:
            print(f"Warning: Could not initialize MuJoCo Renderer. Video will not be recorded. Error: {e}")
            record_video = False

    # Setup ONNX
    print(f"Loading ONNX Model: {onnx_path}")
    sess = ort.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    
    # --- Resolve Joint Order / Metadata ---
    # Attempt to read joint names from ONNX metadata to ensure order matches training
    onnx_joint_names = []
    
    # Placeholders for metadata params
    meta_action_scale = None
    meta_default_pos = None
    meta_stiffness = None
    meta_damping = None
    
    try:
        meta = sess.get_modelmeta()
        custom_props = meta.custom_metadata_map
        
        if 'joint_names' in custom_props:
            raw_names = custom_props['joint_names']
            onnx_joint_names = [n.strip() for n in raw_names.split(',')]
            print(f"Loaded {len(onnx_joint_names)} joint names from ONNX Metadata.")
            
        if 'action_scale' in custom_props:
            raw_vals = custom_props['action_scale']
            meta_action_scale = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)
            print("Loaded Action Scales from Metadata.")

        if 'default_joint_pos' in custom_props:
            raw_vals = custom_props['default_joint_pos']
            meta_default_pos = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)
            print("Loaded Default Joint Pos from Metadata.")
            
        if 'joint_stiffness' in custom_props:
            raw_vals = custom_props['joint_stiffness']
            meta_stiffness = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)
            print("Loaded Joint Stiffness (Kp) from Metadata.")
            
        if 'joint_damping' in custom_props:
            raw_vals = custom_props['joint_damping']
            meta_damping = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)
            print("Loaded Joint Damping (Kd) from Metadata.")

        # Check for Normalization Stats (Critical for RSL-RL models if not fused)
        meta_obs_mean = None
        meta_obs_std = None
        
        if 'observation_mean' in custom_props:
             raw_vals = custom_props['observation_mean']
             meta_obs_mean = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)
             print(f"Loaded Observation Mean (shape {meta_obs_mean.shape}) from Metadata.")
             
        if 'observation_std' in custom_props:
             raw_vals = custom_props['observation_std']
             meta_obs_std = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)
             print(f"Loaded Observation Std (shape {meta_obs_std.shape}) from Metadata.")

        if 'normalization_mean' in custom_props:  # Alternate key
             raw_vals = custom_props['normalization_mean']
             meta_obs_mean = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)
             print("Loaded Normalization Mean.")
             
        if 'normalization_std' in custom_props: # Alternate key
             raw_vals = custom_props['normalization_std']
             meta_obs_std = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)
             print("Loaded Normalization Std.")

        if 'body_names' in custom_props:
            print(f"Body Names: {custom_props['body_names']}")
            
        if 'observation_names' in custom_props:
            raw_obs_names = custom_props['observation_names']
            # Might be json or list string
            print(f"Observation Names: {raw_obs_names}")
            
        print("\n--- Metadata Inspection ---")
        print(f"Metadata Keys: {list(custom_props.keys())}")
            
    except Exception as e:
        print(f"Warning: Could not read metadata from ONNX: {e}")

    # MuJoCo Actuator mapping
    n_mujoco_joints = model.nu
    mujoco_actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(n_mujoco_joints)]
    # In G1 XML, actuator names usually match joint names, or we map via trnid
    mujoco_joint_names = []
    mujoco_act_to_joint_id = []
    
    for i in range(n_mujoco_joints):
        j_id = model.actuator_trnid[i, 0]
        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        mujoco_joint_names.append(j_name)
        mujoco_act_to_joint_id.append(j_id)

    # Determine the operative joint list
    # REVERTED: Using Metadata Order (Standard ONNX behavior) is likely correct.
    # The asymmetry must be diagnosed by inspecting scales/defaults.
    FORCE_MUJOCO_ORDER = False 
    
    if FORCE_MUJOCO_ORDER:
        print("\n!!! CRITICAL OVERRIDE: Using MuJoCo Actuator Order for Network Input !!!")
        print("Ignoring ONNX Metadata 'joint_names' order for vector construction sequences.")
        if onnx_joint_names:
            print(f"(Ignored ONNX List starts with: {onnx_joint_names[:3]}...)")
            
        joint_names_ordered = mujoco_joint_names
    elif onnx_joint_names:
        print("Using ONNX Metadata 'joint_names' order.")
        joint_names_ordered = onnx_joint_names
    else:
        print("Using MuJoCo actuator order as fallback.")
        joint_names_ordered = mujoco_joint_names

    n_joints = len(joint_names_ordered)
    print(f"Total Controlled Joints: {n_joints}")

    # Build Mappings
    # onnx_idx -> mujoco_actuator_idx
    # If FORCE_MUJOCO_ORDER is True, this becomes Identity (0->0, 1->1...)
    onnx_to_mujoco_idx = []
    
    for i, name in enumerate(joint_names_ordered):
        try:
            # If we enforce matches, Names must match exactly or close enough
            mj_idx = mujoco_joint_names.index(name)
            onnx_to_mujoco_idx.append(mj_idx)
        except ValueError:
            print(f"Error: Joint '{name}' required by ONNX not found in MuJoCo model actuators.")
            return

    onnx_to_mujoco_idx = np.array(onnx_to_mujoco_idx, dtype=int)
    
    # If FORCE_MUJOCO_ORDER is True ...
    # Wait, if we revert FORCE_MUJOCO_ORDER to False, we should NOT run this block.
    # The variable is defined above.
    
    if FORCE_MUJOCO_ORDER and onnx_joint_names:
         print("Re-mapping Metadata values from ONNX-Order to MuJoCo-Order...")
         
         # Map: For each joint in NEW list, where was it in the OLD list?
         # New: [L_Roll, L_Yaw, L_Pitch...] (MuJoCo)
         # Old: [L_Pitch, R_Pitch, L_Roll...] (Metadata)
         
         meta_perm_idx = []
         for name in joint_names_ordered:
             found_idx = -1
             # Try exact match
             if name in onnx_joint_names:
                 found_idx = onnx_joint_names.index(name)
             else:
                 pass
                 
             if found_idx == -1:
                 # print(f"Warning: MuJoCo joint '{name}' not found in ONNX Metadata list. Using index 0.")
                 found_idx = 0
             
             meta_perm_idx.append(found_idx)
         
         meta_perm_idx = np.array(meta_perm_idx, dtype=int)
         
         # Apply Permutation
         if meta_action_scale is not None:
             meta_action_scale = meta_action_scale[meta_perm_idx]
         if meta_default_pos is not None:
             meta_default_pos = meta_default_pos[meta_perm_idx]
         if meta_stiffness is not None:
             meta_stiffness = meta_stiffness[meta_perm_idx]
         if meta_damping is not None:
             meta_damping = meta_damping[meta_perm_idx]
             
         print("Metadata re-mapping complete.")
    
    # Precompute Observation Indices (MuJoCo State -> ONNX Input Order)
    obs_qpos_indices = []
    obs_qvel_indices = []
    for i in range(n_joints):
        mj_act_idx = onnx_to_mujoco_idx[i]
        j_id = mujoco_act_to_joint_id[mj_act_idx]
        obs_qpos_indices.append(model.jnt_qposadr[j_id])
        obs_qvel_indices.append(model.jnt_dofadr[j_id])
        
    obs_qpos_indices = np.array(obs_qpos_indices, dtype=int)
    obs_qvel_indices = np.array(obs_qvel_indices, dtype=int)
    print("Precomputed Observation Mapping Indices.")

    # Precompute Scales and Defaults in ONNX ORDER
    # Priority: Metadata > Helper Function > Default
    
    if meta_action_scale is not None and len(meta_action_scale) == n_joints:
        action_scales = meta_action_scale
        print("\n--- Action Scales & Defaults Inspection ---")
        # Print first few to check symmetry
        # indices for LP(0), RP(1), LR(3), RR(4) based on metadata usually
        try:
             print(f"Scale[0] ({joint_names_ordered[0]}): {action_scales[0]}")
             print(f"Scale[1] ({joint_names_ordered[1]}): {action_scales[1]}")
             print(f"Scale[3] ({joint_names_ordered[3]}): {action_scales[3]}")
             print(f"Scale[4] ({joint_names_ordered[4]}): {action_scales[4]}")
        except: pass
    else:
        print("Warning: Using estimated Action Scales (fallback).")
        action_scales = np.array([get_action_scale(n) for n in joint_names_ordered], dtype=np.float32)
        
    if meta_default_pos is not None and len(meta_default_pos) == n_joints:
        default_pos = meta_default_pos
        print(f"DefaultPos[0]: {default_pos[0]}, DefaultPos[1]: {default_pos[1]}")
        
        # Check defaults for Roll (Indices 3, 4 based on previous logs)
        try:
            print(f"DefaultPos[3] (L_Roll): {default_pos[3]}")
            print(f"DefaultPos[4] (R_Roll): {default_pos[4]}")
        except: pass
    else:
        print("Warning: Using estimated Default Pos (fallback).")
        default_pos = np.array([get_default_pos(n) for n in joint_names_ordered], dtype=np.float32)

    # Precompute Gains
    if meta_stiffness is not None and len(meta_stiffness) == n_joints:
        kp_gains = meta_stiffness
    else:
        print("Warning: Using default Kp=300.")
        kp_gains = np.full(n_joints, 300.0)
        
    if meta_damping is not None and len(meta_damping) == n_joints:
        kd_gains = meta_damping
    else:
        print("Warning: Using default Kd=10.")
        kd_gains = np.full(n_joints, 10.0)

    # Prepare Reference Motion
    print(f"Loading Reference Motion: {npz_path}")
    raw_data = np.load(npz_path, allow_pickle=True) # allow_pickle for metadata dictionaries
    
    # 1. Resolve Reference Joint Order
    ref_joint_names = []
    
    # Heuristics to find joint names in NPZ
    if 'joint_names' in raw_data:
        ref_joint_names = raw_data['joint_names']
        if isinstance(ref_joint_names, np.ndarray):
            ref_joint_names = ref_joint_names.tolist()
    elif 'skeleton' in raw_data:
        # Some formats store skeleton info
        pass 
        
    if not ref_joint_names:
        print("Warning: No 'joint_names' found in NPZ.")
        
        if npz_is_onnx_order:
            print("Assuming NPZ Data ALREADY follows ONNX/Network Order.")
            print("-> Skipping reordering. Mapping 1-to-1.")
            ref_map_indices = np.arange(n_joints, dtype=int)
        else:
            print("Assuming NPZ Data follows MuJoCo Keypoint/Actuator Order (Physical Order).")
            print("Reordering NPZ data to match ONNX Input Order...")
            
            # If NPZ order == MuJoCo Order, and we want ONNX Order:
            # We rely on onnx_to_mujoco_idx which maps ONNX_i -> MuJoCo_i
            ref_map_indices = onnx_to_mujoco_idx
    else:
        print(f"Found {len(ref_joint_names)} joints in NPZ data.")
        # Build mapping: ONNX_Joint[i] -> comes from -> NPZ_Joint[k]
        ref_map_indices = []
        for name in joint_names_ordered:
            # Fuzzy match or exact match
            # G1 names might be "left_hip_roll_joint" vs "left_hip_roll"
            
            # Try exact
            if name in ref_joint_names:
                ref_map_indices.append(ref_joint_names.index(name))
                continue
                
            # Try simple suffix cleaning
            clean_name = name.replace("_joint", "")
            found = False
            for j, ref_n in enumerate(ref_joint_names):
                if ref_n == clean_name or ref_n.replace("_joint", "") == clean_name:
                    ref_map_indices.append(j)
                    found = True
                    break
            
            if not found:
                print(f"Critical Warning: Reference motion is missing data for joint '{name}'. Filling with zeros.")
                ref_map_indices.append(-1) # Indicator for missing
                
        ref_map_indices = np.array(ref_map_indices)

    # 2. Extract and Reorder Data
    if 'qpos' in raw_data:
        # Raw is usually [Frames, 7 + Nu]
        qpos_raw = raw_data['qpos']
        
        # Root is first 7
        ref_root_all = qpos_raw[:, :7]
        ref_joints_all_raw = qpos_raw[:, 7:]
        
        # Reorder joints
        ref_joints_all = np.zeros((len(qpos_raw), n_joints))
        
        for i_onnx, i_ref in enumerate(ref_map_indices):
            if i_ref != -1:
                # Check bounds
                if i_ref < ref_joints_all_raw.shape[1]:
                    ref_joints_all[:, i_onnx] = ref_joints_all_raw[:, i_ref]
        
        # Reassemble
        # Note: qpos_ref_all MUST match the physical simulation layout (MuJoCo Order) for the root setting?
        # WAIT. The 'qpos_ref_all' is used to drive the "Reference Observation" (which needs ONNX order)
        # AND to drive the "Initial State" (which needs MuJoCo order).
        # This is a conflict if we just use one array.
        
        # Let's clean this up.
        # We need:
        # 1. qpos_ref_onnx: [Frames, 7 + n_joints_onnx] (Ordered for Policy Input)
        # 2. qpos_ref_mujoco: [Frames, 7 + n_joints_mujoco] (Ordered for Physics Init)
        
        # Construct ONNX-ordered Reference
        qpos_ref_onnx = np.concatenate([ref_root_all, ref_joints_all], axis=1)
        
        # Compute Velocity (in ONNX order)
        qvel_ref_onnx = np.zeros((len(qpos_ref_onnx), 6 + n_joints))
        dt_motion = 1.0/30.0
        if 'fps' in raw_data:
            dt_motion = 1.0 / float(raw_data['fps'])
            
        print("Computing Reference Velocities (Finite Diff)...")
        # Simple diff for now to save complexity, respecting quaternion diff could be added
        # Velocity for root (6D) + joints
        for t in range(len(qpos_ref_onnx) - 1):
             q1 = qpos_ref_onnx[t]
             q2 = qpos_ref_onnx[t+1]
             # Linear
             qvel_ref_onnx[t, 0:3] = (q2[0:3] - q1[0:3]) / dt_motion
             # Angular (Approx for now, ideal is mj_diff)
             # Joints
             qvel_ref_onnx[t, 6:] = (q2[7:] - q1[7:]) / dt_motion
             
        # Use ONNX-ordered reference for the Interpolators used in OBSERVATION
        qpos_ref_use = qpos_ref_onnx
        qvel_ref_use = qvel_ref_onnx
        
    else:
        print("Error: NPZ must contain 'qpos'")
        return

    n_frames = len(qpos_ref_use)
    # Determine FPS
    motion_fps = 30.0
    if 'fps' in raw_data:
        motion_fps = float(raw_data['fps'])
    motion_duration = n_frames / motion_fps
    
    print(f"Motion Duration: {motion_duration:.2f}s (FPS: {motion_fps})")

    # Simulation Loop
    decimation = int(control_dt / sim_dt)
    model.opt.timestep = sim_dt
    
    # Initialize Physics State
    # STRATEGY: Start exactly at Reference Frame 0 to match tracking intent.
    # Note: We trust npz order is handled correctly above.
    
    print("\n--- Initialization Strategy: REFERENCE START ---")
    print("Setting robot state to Reference Motion at t=0.")
    
    start_q = qpos_ref_use[0]
    start_v = qvel_ref_use[0]
    
    # Root
    data.qpos[:3] = start_q[:3]
    data.qpos[2] = max(data.qpos[2], 0.74) # Safety floor clip
    data.qpos[3:7] = start_q[3:7]
    
    # Velocity
    data.qvel[:3] = start_v[:3]
    data.qvel[3:6] = start_v[3:6]
    
    # Joints (Loop to handle mapping if needed, but qpos_ref_use is already ONNX-ordered)
    # However, data.qpos must be loaded in PHYSICAL order.
    # qpos_ref_use is ONNX order (policy input order).
    # We must map ONNX -> Physical for initialization.
    
    for i_onnx in range(n_joints):
        mj_act_idx = onnx_to_mujoco_idx[i_onnx]
        j_id = mujoco_act_to_joint_id[mj_act_idx]
        q_adr = model.jnt_qposadr[j_id]
        dof_adr = model.jnt_dofadr[j_id]
        
        # Position
        data.qpos[q_adr] = start_q[7 + i_onnx]
        # Velocity
        data.qvel[dof_adr] = start_v[6 + i_onnx]
        
    print("Initialized to Reference Frame 0.")
    print("-" * 90)
    
    # DEBUG: FORCE STATIC REFERENCE
    # If True, overrides the NPZ reference with the static Default Pose.
    # Use this to verify if the Robot can just "Stand Still" with the policy.
    # If it falls even with this, the Policy Observation or Action Mapping is broken.
    USE_STATIC_REF_DEBUG = False
    
    if USE_STATIC_REF_DEBUG:
        print("\n!!! DEBUG MODE: STATIC REFERENCE ENABLED !!!")
        print("Ignoring NPZ Motion. Feeding 'Static Standing' reference to Policy.")
        # Construct a static frame that matches Default Pos
        # Root: 0, 0, 0.75, Identity Quat
        # Joints: Default Pos
        debug_ref_q = np.zeros(7 + n_joints)
        debug_ref_q[0:3] = init_root_pos
        debug_ref_q[3:7] = init_root_quat # Maintain the upright yaw
        debug_ref_q[7:] = default_pos # ONNX Order
        
        debug_ref_v = np.zeros(6 + n_joints)
        
    mujoco.mj_forward(model, data)
    
    # Last action in ONNX order
    last_action = np.zeros(n_joints, dtype=np.float32)
    # Now using kp_gains and kd_gains loaded from metadata 
    time_log = []
    work_log = []
    torque_log = []
    energy_pot_log = []
    energy_kin_log = []
    
    current_work = 0.0
    
    # Interpolation Indices
    time_ref = np.linspace(0, motion_duration, n_frames)
    
    interp_qpos = interp1d(time_ref, qpos_ref_use, axis=0, fill_value="extrapolate")
    interp_qvel = interp1d(time_ref, qvel_ref_use, axis=0, fill_value="extrapolate")
    
    # Initialize FK Structures
    ref_data = mujoco.MjData(model)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_id == -1:
        print("Warning: 'torso_link' not found. Falling back to 'pelvis' (body 1) or 'world' (0).")
        # Usually body 1 is root link
        torso_id = 1 
    print(f"Using Body ID {torso_id} ('{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, torso_id)}') as Anchor.")

    # [Contact Detection Setup]
    # Identify "floor" geom ID
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_geom_id == -1:
        # Fallback if geom name is not "floor", maybe "ground" or just using generic class
        print("Warning: Geom 'floor' not found. Contact detection might fail.")
    
    # Launch Passive Viewer for real-time observation
    # viewer = mujoco.viewer.launch_passive(model, data)
    # print("Starting Simulation with Viewer...")
    
    # [Headless Mode]
    print("Starting Headless Simulation...")
    sim_time = 0.0
    
    # Logs for Contact
    contact_log = []
    
    while sim_time < motion_duration: # and viewer.is_running():
        # Viewer Sync (Disable for Headless)
        # viewer.sync()        
        # 1. Get Control (Policy)
        if int(sim_time / sim_dt) % decimation == 0:
            # Sample reference at this time
            if USE_STATIC_REF_DEBUG:
                 cur_q_ref = debug_ref_q
                 cur_v_ref = debug_ref_v
            else:
                 cur_q_ref = interp_qpos(sim_time)
                 cur_v_ref = interp_qvel(sim_time)
            
            # Determine if we need Wo State Est (based on model input size if available, or config)
            # Standard G1 Flat: 160 dim
            # Wo State Est: 154 dim (Removed 3 pos + 3 lin vel)
            
            enable_wo_state_est = False
            
            # Auto-detect based on ONNX input shape if possible
            if len(sess.get_inputs()) > 0:
                 input_shape = sess.get_inputs()[0].shape
                 # shape is usually [batch, dim] -> [1, 160] or [1, 154]
                 if len(input_shape) > 1 and input_shape[1] == 154:
                      enable_wo_state_est = True
                 elif len(input_shape) > 1 and input_shape[1] == 160:
                      enable_wo_state_est = False
            
            # Construct Obs
            obs = compute_observations(data, model, ref_data, torso_id, cur_q_ref, cur_v_ref, last_action, default_pos, obs_qpos_indices, obs_qvel_indices, wo_state_est=enable_wo_state_est)
            
            # [Fix] Apply Normalization if available
            if meta_obs_mean is not None and meta_obs_std is not None:
                # Ensure broadcasting or shape match
                # obs is (N,), mean is (N,)
                obs = (obs - meta_obs_mean) / (meta_obs_std + 1e-8)
                # Clip? Usually RSL-RL clips to +/- 100 or so, rarely bounds to [-5, 5] unless specified.
                obs = np.clip(obs, -100.0, 100.0)

            # Construct ONNX Inputs
            frame_idx = min(int(sim_time * motion_fps), n_frames - 1)
            
            # Note: Verify input names. Some models use 'obs', others use 'input', etc.
            # sess.get_inputs()[0].name is stored in input_name
            # If model needs time_step, we provide it.
            inputs = {input_name: obs.astype(np.float32).reshape(1, -1)}
            
            # Check if second input is needed (often time_step)
            if len(sess.get_inputs()) > 1:
                name2 = sess.get_inputs()[1].name
                inputs[name2] = np.array([[frame_idx]], dtype=np.float32) # or int64 depending on model
            
            outputs = sess.run(None, inputs)
            action = outputs[0][0]
            
            last_action = action
            
            # Compute Target in ONNX Order
            target_q_onnx = action * action_scales + default_pos
            
        # 2. Step Physics (PD Control)
        # We need to map target_q_onnx back to MuJoCo ctrl indices
        
        for i_onnx in range(n_joints):
            mj_act_idx = onnx_to_mujoco_idx[i_onnx]
            
            # Current State (Low level)
            j_id = mujoco_act_to_joint_id[mj_act_idx]
            q_adr = model.jnt_qposadr[j_id]
            dof_adr = model.jnt_dofadr[j_id]
            
            curr_q = data.qpos[q_adr]
            curr_v = data.qvel[dof_adr]
            
            des_q = target_q_onnx[i_onnx] # Target from Policy
            
            # toMatched PD Gains
            kp = kp_gains[i_onnx]
            kd = kd_gains[i_onnx]
            
            # Torque = Kp * error - Kd * velocity
            torque = kp * (des_q - curr_q) - kd * curr_v
            # Torque Limit
            limit = model.actuator_ctrlrange[mj_act_idx]
            torque = np.clip(torque, limit[0], limit[1])
            
            data.ctrl[mj_act_idx] = torque
            
        mujoco.mj_step(model, data)
        # Viewer Sync (Disable for Headless)
        # viewer.sync()
        
        # 3. Energy Analysis
        mujoco.mj_energyPos(model, data)
        mujoco.mj_energyVel(model, data)
        pe = data.energy[0]
        ke = data.energy[1]
        
        # 4. Contact Detection
        is_contacting = False
        for i_con in range(data.ncon):
            contact = data.contact[i_con]
            # Check if one of the geoms is the floor
            if contact.geom1 == floor_geom_id or contact.geom2 == floor_geom_id:
                is_contacting = True
                break
        
        inst_power = 0.0
        inst_torque_sq = 0.0
        for i_mj in range(model.nu):
             j_id = model.actuator_trnid[i_mj, 0]
             dof_adr = model.jnt_dofadr[j_id]
             v = data.qvel[dof_adr]
             tau = data.ctrl[i_mj]
             inst_power += abs(tau * v)
             inst_torque_sq += tau**2
             
        current_work += inst_power * sim_dt
        
        time_log.append(sim_time)
        contact_log.append(is_contacting)
        work_log.append(current_work)
        torque_log.append(inst_torque_sq)
        energy_pot_log.append(pe)
        energy_kin_log.append(ke)
        
        # 5. Advance Time
        sim_time += sim_dt

        # Video Recording
        if record_video and (sim_time - last_render_time >= 1.0 / video_fps):
            renderer.update_scene(data, camera=-1) # Free camera or tracking? -1 is usually acceptable default or specific cam name
            
            pixel_data = renderer.render()
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(pixel_data, cv2.COLOR_RGB2BGR)
            frames.append(frame_bgr)
            last_render_time = sim_time

    print("Simulation Complete.")
    # viewer.close()
    
    # Save Video
    if record_video and frames:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        video_path = os.path.join(output_dir, f"{base_name}_simulation.mp4")
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # or 'avc1'
        video = cv2.VideoWriter(video_path, fourcc, video_fps, (width, height))
        
        for f in frames:
            video.write(f)
        
        video.release()
        print(f"Video saved to: {video_path}")
    
    print("Simulation Complete.")
    
    # Plotting
    if output_dir is None:
        output_dir = os.path.dirname(npz_path)

    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    
    plt.figure(figsize=(10, 12)) # Taller figure 3x1

    # Helper for adding contact regions
    def add_contact_regions(ax, times, contacts):
        # Fill regions where contacts is True
        # To make it clean, we find starts and ends
        is_con_arr = np.array(contacts, dtype=bool)
        if not np.any(is_con_arr):
            return

        # Simple fill_between approach
        # We fill the entire Y range for True values
        # Getting Y limits is tricky before plotting, so we use transform
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        
        ax.fill_between(times, 0, 1, where=is_con_arr, 
                        facecolor='green', alpha=0.15, transform=trans, label='Ground Contact')


    # Plot 1: Energy Composition (Stack Plot style like high_freq_analysis)
    ax1 = plt.subplot(3, 1, 1)
    
    # Calculate baseline potential energy for better visualization (optional, similar to high_freq_analysis)
    min_pot = np.min(energy_pot_log)
    e_pot_adj = np.array(energy_pot_log) - min_pot
    
    ax1.stackplot(time_log, e_pot_adj, energy_kin_log, labels=['Potential (Adj)', 'Kinetic'], alpha=0.6, colors=['#1f77b4', '#ff7f0e'])
    add_contact_regions(ax1, time_log, contact_log)
    
    # Also plot Total Energy line
    total_energy = np.array(energy_pot_log) + np.array(energy_kin_log)
    ax1.plot(time_log, total_energy - min_pot, color='black', linestyle='--', alpha=0.5, label='Total (Adj)')
    
    ax1.set_title('Instantaneous Energy Composition')
    ax1.set_ylabel('Energy (J)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Work
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(time_log, work_log, color='red', label='Cumulative Work')
    add_contact_regions(ax2, time_log, contact_log)
    ax2.set_title(f"Cumulative Work: {current_work:.2f} J")
    ax2.set_ylabel("Work (J)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Instantaneous Effort (Torque Squared)
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(time_log, torque_log, color='purple', label='Effort (Torque^2)')
    ax3.fill_between(time_log, 0, torque_log, color='purple', alpha=0.1)
    add_contact_regions(ax3, time_log, contact_log)
    ax3.set_title("Instantaneous Effort")
    ax3.set_ylabel("Tau^2")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{base_name}_onnx_energy.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    
    return {
        "time": time_log,
        "work": work_log,
        "torque": torque_log,
        "energy_pot": energy_pot_log,
        "energy_kin": energy_kin_log,
        "total_work": current_work,
        "contact": contact_log
    }
