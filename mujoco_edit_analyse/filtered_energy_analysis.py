import sys
import os
import re
import numpy as np
import mujoco
import mujoco.viewer
import onnxruntime as ort
import cv2
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Ensure the parent directory is in sys.path to allow importing 'energy_analyse'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from energy_analyse.g1_config import get_action_scale, get_default_pos
from energy_analyse.observation import compute_observations
from energy_analyse.g1_physics import apply_g1_physics

def calculate_filtered_mechanical_energy(model, data, excluded_body_ids):
    """
    Calculates the total mechanical energy (PE + KE) of the system,
    ignoring the contribution of specified bodies.
    """
    total_pe = 0.0
    total_ke = 0.0
    
    # Gravity can be retrieved from options
    gravity = model.opt.gravity
    
    # Iterate over all bodies (skip world body 0)
    for i in range(1, model.nbody):
        if i in excluded_body_ids:
            continue
            
        mass = model.body_mass[i]
        
        # --- Potential Energy (Gravitational) ---
        # PE = - m * g^T * pos
        # data.xipos is the position of the body's COM in global frame
        pos = data.xipos[i]
        pe_body = -mass * np.dot(gravity, pos)
        total_pe += pe_body
        
        # --- Kinetic Energy ---
        # data.cvel[i] is the 6D spatial velocity of the body's COM in world frame
        # Format: [rot_x, rot_y, rot_z, lin_x, lin_y, lin_z]
        cvel = data.cvel[i]
        w = cvel[:3] # Angular velocity
        v = cvel[3:] # Linear velocity
        
        # Linear KE
        ke_trans = 0.5 * mass * np.dot(v, v)
        
        # Rotational KE
        # Inertia tensor in world frame: I_world = R * I_local * R^T
        # model.body_inertia[i] is the diagonal of the inertia tensor in the body frame
        
        # Rotation matrix from body to world (3x3)
        R = data.ximat[i].reshape(3, 3)
        
        # Local Inertia Tensor
        I_local = np.diag(model.body_inertia[i])
        
        # World Inertia Tensor
        I_world = R @ I_local @ R.T
        
        ke_rot = 0.5 * np.dot(w, I_world @ w)
        
        total_ke += (ke_trans + ke_rot)
        
    return total_pe, total_ke

def run_filtered_energy_analysis(xml_path, onnx_path, npz_path, output_dir=None, sim_dt=0.005, control_dt=0.02, record_video=True, npz_is_onnx_order=False):
    # Setup Output Directory
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "result")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    print(f"Loading MuJoCo Model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    
    # Apply Training Physics Parameters
    apply_g1_physics(model)
    
    data = mujoco.MjData(model)
    
    # Identify Excluded Bodies (Arms and Head)
    excluded_body_ids = set()
    excluded_keywords = ["shoulder", "elbow", "wrist", "hand", "head"]
    
    print("\n--- Body Exclusion for Energy Analysis ---")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            if any(k in name for k in excluded_keywords):
                excluded_body_ids.add(i)
                # print(f"Excluding body: {name}")
    print(f"Total Excluded Bodies: {len(excluded_body_ids)}")
    print("-" * 40)

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
    
    # --- Metadata Parsing (Simplified for brevity, assuming standard G1 format or copied from logic) ---
    onnx_joint_names = []
    meta_action_scale = None
    meta_default_pos = None
    meta_stiffness = None
    meta_damping = None
    meta_obs_mean = None
    meta_obs_std = None
    
    try:
        meta = sess.get_modelmeta()
        custom_props = meta.custom_metadata_map
        
        if 'joint_names' in custom_props:
            raw_names = custom_props['joint_names']
            onnx_joint_names = [n.strip() for n in raw_names.split(',')]
            
        if 'action_scale' in custom_props:
            raw_vals = custom_props['action_scale']
            meta_action_scale = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)

        if 'default_joint_pos' in custom_props:
            raw_vals = custom_props['default_joint_pos']
            meta_default_pos = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)
            
        if 'joint_stiffness' in custom_props:
            raw_vals = custom_props['joint_stiffness']
            meta_stiffness = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)
            
        if 'joint_damping' in custom_props:
            raw_vals = custom_props['joint_damping']
            meta_damping = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)

        if 'observation_mean' in custom_props:
             raw_vals = custom_props['observation_mean']
             meta_obs_mean = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)
             
        if 'observation_std' in custom_props:
             raw_vals = custom_props['observation_std']
             meta_obs_std = np.array([float(x) for x in raw_vals.split(',')], dtype=np.float32)
             
    except Exception as e:
        print(f"Warning: Could not read metadata from ONNX: {e}")

    # MuJoCo Indexing
    n_mujoco_joints = model.nu
    mujoco_joint_names = []
    mujoco_act_to_joint_id = []
    
    for i in range(n_mujoco_joints):
        j_id = model.actuator_trnid[i, 0]
        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        mujoco_joint_names.append(j_name)
        mujoco_act_to_joint_id.append(j_id)

    # Order Resolution
    if onnx_joint_names:
        joint_names_ordered = onnx_joint_names
    else:
        joint_names_ordered = mujoco_joint_names
        
    n_joints = len(joint_names_ordered)
    
    # Mapping
    onnx_to_mujoco_idx = []
    for i, name in enumerate(joint_names_ordered):
        try:
            mj_idx = mujoco_joint_names.index(name)
            onnx_to_mujoco_idx.append(mj_idx)
        except ValueError:
            print(f"Error: Joint '{name}' required by ONNX not found.")
            return
            
    onnx_to_mujoco_idx = np.array(onnx_to_mujoco_idx, dtype=int)
    
    # Re-map Metadata if needed (Assume ONNX order matches metadata, which is standard)
    
    # Precompute Obs Indices
    obs_qpos_indices = []
    obs_qvel_indices = []
    for i in range(n_joints):
        mj_act_idx = onnx_to_mujoco_idx[i]
        j_id = mujoco_act_to_joint_id[mj_act_idx]
        obs_qpos_indices.append(model.jnt_qposadr[j_id])
        obs_qvel_indices.append(model.jnt_dofadr[j_id])
        
    obs_qpos_indices = np.array(obs_qpos_indices, dtype=int)
    obs_qvel_indices = np.array(obs_qvel_indices, dtype=int)

    # Scales and Defaults
    if meta_action_scale is not None and len(meta_action_scale) == n_joints:
        action_scales = meta_action_scale
    else:
        action_scales = np.array([get_action_scale(n) for n in joint_names_ordered], dtype=np.float32)
        
    if meta_default_pos is not None and len(meta_default_pos) == n_joints:
        default_pos = meta_default_pos
    else:
        default_pos = np.array([get_default_pos(n) for n in joint_names_ordered], dtype=np.float32)

    if meta_stiffness is not None and len(meta_stiffness) == n_joints:
        kp_gains = meta_stiffness
    else:
        kp_gains = np.full(n_joints, 300.0)
        
    if meta_damping is not None and len(meta_damping) == n_joints:
        kd_gains = meta_damping
    else:
        kd_gains = np.full(n_joints, 10.0)

    # Load Reference Motion
    print(f"Loading Reference Motion: {npz_path}")
    raw_data = np.load(npz_path, allow_pickle=True)
    
    # Motion Mapping (NPZ -> ONNX)
    ref_joint_names = []
    if 'joint_names' in raw_data:
        ref_joint_names = raw_data['joint_names'].tolist()
        
    if not ref_joint_names:
        # Fallback logic
        if npz_is_onnx_order:
            ref_map_indices = np.arange(n_joints, dtype=int)
        else:
            # Assume MuJoCo order
            ref_map_indices = onnx_to_mujoco_idx
    else:
        ref_map_indices = []
        for name in joint_names_ordered:
            if name in ref_joint_names:
                ref_map_indices.append(ref_joint_names.index(name))
            else:
                 # Try approximate match
                 clean = name.replace("_joint", "")
                 found = False
                 for j, ref_n in enumerate(ref_joint_names):
                     if ref_n == clean or ref_n.replace("_joint", "") == clean:
                         ref_map_indices.append(j)
                         found = True
                         break
                 if not found:
                     ref_map_indices.append(-1)
        ref_map_indices = np.array(ref_map_indices)

    if 'qpos' in raw_data:
        qpos_raw = raw_data['qpos']
        ref_root_all = qpos_raw[:, :7]
        ref_joints_all_raw = qpos_raw[:, 7:]
        
        ref_joints_all = np.zeros((len(qpos_raw), n_joints))
        
        for i_onnx, i_ref in enumerate(ref_map_indices):
            if i_ref != -1 and i_ref < ref_joints_all_raw.shape[1]:
                ref_joints_all[:, i_onnx] = ref_joints_all_raw[:, i_ref]
                
        qpos_ref_onnx = np.concatenate([ref_root_all, ref_joints_all], axis=1)
        
        # Velocity
        qvel_ref_onnx = np.zeros((len(qpos_ref_onnx), 6 + n_joints))
        dt_motion = 1.0/30.0
        if 'fps' in raw_data:
            dt_motion = 1.0 / float(raw_data['fps'])
            
        for t in range(len(qpos_ref_onnx) - 1):
             q1 = qpos_ref_onnx[t]
             q2 = qpos_ref_onnx[t+1]
             # Simple linear diff
             qvel_ref_onnx[t, 0:3] = (q2[0:3] - q1[0:3]) / dt_motion
             qvel_ref_onnx[t, 6:] = (q2[7:] - q1[7:]) / dt_motion
             
        qpos_ref_use = qpos_ref_onnx
        qvel_ref_use = qvel_ref_onnx
    else:
        print("Error: NPZ missing 'qpos'")
        return

    n_frames = len(qpos_ref_use)
    motion_fps = float(raw_data.get('fps', 30.0))
    motion_duration = n_frames / motion_fps
    
    # Init Physics
    model.opt.timestep = sim_dt
    decimation = int(control_dt / sim_dt)
    
    start_q = qpos_ref_use[0]
    start_v = qvel_ref_use[0]
    
    data.qpos[:3] = start_q[:3]
    data.qpos[2] = max(data.qpos[2], 0.74)
    data.qpos[3:7] = start_q[3:7]
    data.qvel[:3] = start_v[:3]
    data.qvel[3:6] = start_v[3:6]
    
    for i_onnx in range(n_joints):
        mj_act_idx = onnx_to_mujoco_idx[i_onnx]
        j_id = mujoco_act_to_joint_id[mj_act_idx]
        q_adr = model.jnt_qposadr[j_id]
        dof_adr = model.jnt_dofadr[j_id]
        
        data.qpos[q_adr] = start_q[7 + i_onnx]
        data.qvel[dof_adr] = start_v[6 + i_onnx]
        
    mujoco.mj_forward(model, data)
    
    last_action = np.zeros(n_joints, dtype=np.float32)
    
    time_log = []
    work_log = []
    torque_log = []
    energy_pot_log = []
    energy_kin_log = []
    contact_log = []
    
    current_work = 0.0
    
    time_ref = np.linspace(0, motion_duration, n_frames)
    interp_qpos = interp1d(time_ref, qpos_ref_use, axis=0, fill_value="extrapolate")
    interp_qvel = interp1d(time_ref, qvel_ref_use, axis=0, fill_value="extrapolate")
    
    ref_data = mujoco.MjData(model)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if torso_id == -1: torso_id = 1
    
    floor_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    
    sim_time = 0.0
    print("Starting Simulation...")
    
    while sim_time < motion_duration:
        # Control Step
        if int(sim_time / sim_dt) % decimation == 0:
            cur_q_ref = interp_qpos(sim_time)
            cur_v_ref = interp_qvel(sim_time)
            
            enable_wo_state_est = False
            if len(sess.get_inputs()) > 0:
                 input_shape = sess.get_inputs()[0].shape
                 if len(input_shape) > 1 and input_shape[1] == 154:
                      enable_wo_state_est = True
            
            obs = compute_observations(data, model, ref_data, torso_id, cur_q_ref, cur_v_ref, last_action, default_pos, obs_qpos_indices, obs_qvel_indices, wo_state_est=enable_wo_state_est)
            
            if meta_obs_mean is not None and meta_obs_std is not None:
                obs = (obs - meta_obs_mean) / (meta_obs_std + 1e-8)
                obs = np.clip(obs, -100.0, 100.0)
                
            inputs = {input_name: obs.astype(np.float32).reshape(1, -1)}
            if len(sess.get_inputs()) > 1:
                name2 = sess.get_inputs()[1].name
                inputs[name2] = np.array([[min(int(sim_time * motion_fps), n_frames - 1)]], dtype=np.float32)

            action = sess.run(None, inputs)[0][0]
            last_action = action
            target_q_onnx = action * action_scales + default_pos
            
        # Physics Step
        for i_onnx in range(n_joints):
            mj_act_idx = onnx_to_mujoco_idx[i_onnx]
            j_id = mujoco_act_to_joint_id[mj_act_idx]
            q_adr = model.jnt_qposadr[j_id]
            dof_adr = model.jnt_dofadr[j_id]
            
            curr_q = data.qpos[q_adr]
            curr_v = data.qvel[dof_adr]
            des_q = target_q_onnx[i_onnx]
            
            torque = kp_gains[i_onnx] * (des_q - curr_q) - kd_gains[i_onnx] * curr_v
            limit = model.actuator_ctrlrange[mj_act_idx]
            torque = np.clip(torque, limit[0], limit[1])
            data.ctrl[mj_act_idx] = torque
            
        mujoco.mj_step(model, data)
        
        # Energy Analysis (Using Filtered Version)
        # mujoco.mj_energyPos(model, data) # Still need this to update data.xipos if not auto-updated? 
        # Actually mj_step updates kinematics. mj_energyPos just computes energy scalar.
        # But we need xipos and cvel which are computed during mj_step?
        # Yes, mj_step calls mj_forward which calls mj_kinematics etc.
        
        # Call our custom energy function
        pe, ke = calculate_filtered_mechanical_energy(model, data, excluded_body_ids)
        
        # Contact
        is_contacting = False
        if floor_geom_id != -1:
            for i_con in range(data.ncon):
                contact = data.contact[i_con]
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
        
        sim_time += sim_dt
        
        if record_video and (sim_time - last_render_time >= 1.0 / video_fps):
            renderer.update_scene(data, camera=-1)
            frame_bgr = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            frames.append(frame_bgr)
            last_render_time = sim_time

    if record_video and frames:
        base_name = os.path.splitext(os.path.basename(npz_path))[0]
        video_path = os.path.join(output_dir, f"{base_name}_filtered_sim.mp4")
        height, width, _ = frames[0].shape
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (width, height))
        for f in frames: video.write(f)
        video.release()
        print(f"Video saved to: {video_path}")
        
    print("Simulation Complete. Plotting...")
    
    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    plt.figure(figsize=(10, 12))

    def add_contact_regions(ax, times, contacts):
        is_con_arr = np.array(contacts, dtype=bool)
        if not np.any(is_con_arr): return
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(times, 0, 1, where=is_con_arr, facecolor='green', alpha=0.15, transform=trans, label='Ground Contact')

    # Plot 1
    ax1 = plt.subplot(3, 1, 1)
    min_pot = np.min(energy_pot_log)
    e_pot_adj = np.array(energy_pot_log) - min_pot
    
    ax1.stackplot(time_log, e_pot_adj, energy_kin_log, labels=['Potential (Filtered, Adj)', 'Kinetic (Filtered)'], alpha=0.6, colors=['#1f77b4', '#ff7f0e'])
    add_contact_regions(ax1, time_log, contact_log)
    total_energy = np.array(energy_pot_log) + np.array(energy_kin_log)
    ax1.plot(time_log, total_energy - min_pot, color='black', linestyle='--', alpha=0.5, label='Total (Filtered, Adj)')
    ax1.set_title('Instantaneous Energy Composition (Filtered: No Arms/Head)')
    ax1.set_ylabel('Energy (J)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(time_log, work_log, color='red', label='Cumulative Work')
    add_contact_regions(ax2, time_log, contact_log)
    ax2.set_title(f"Cumulative Work: {current_work:.2f} J")
    ax2.set_ylabel("Work (J)")
    ax2.grid(True, alpha=0.3)
    
    # Plot 3
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(time_log, torque_log, color='purple', label='Effort (Torque^2)')
    ax3.fill_between(time_log, 0, torque_log, color='purple', alpha=0.1)
    add_contact_regions(ax3, time_log, contact_log)
    ax3.set_title("Instantaneous Effort")
    ax3.set_ylabel("Tau^2")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{base_name}_filtered_energy.png")
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    # Settings
    xml_file = r"c:\Users\not a fat cat\Desktop\Motion_rebuild\assets\scene_29dof.xml"
    onnx_file = r"c:\Users\not a fat cat\Desktop\Motion_rebuild\onnx\2026-01-30_15-12-39_v1.onnx"
    # Choose a motion file
    npz_data = r"C:\Users\not a fat cat\Desktop\Motion_rebuild\data\Kjump\kjump-1to-1.npz"
    
    result_dir = r"c:\Users\not a fat cat\Desktop\Motion_rebuild\result"
    
    run_filtered_energy_analysis(xml_file, onnx_file, npz_data, output_dir=result_dir, record_video=True, npz_is_onnx_order=False)
