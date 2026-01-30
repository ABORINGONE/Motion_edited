import mujoco
import numpy as np
import mink
import sys
import os
from scipy.interpolate import CubicSpline

# Import the manager from the original file to maintain logic consistency
from landing_generator import LandingCushionManager

def generate_motion_high_freq(xml_path, input_npz_path, output_npz_path, cushion_depth, force_landing_frame=None, ik_fps=120):
    """
    Generates a new motion trajectory with cushion effect using High-Frequency IK.
    It interpolates the input trajectory to a higher FPS before running the IK solver.
    
    Args:
        ik_fps: The frequency to run the IK solver at (default 120Hz).
    """
    print(f"Generating High-Freq Motion ({ik_fps}Hz) for depth {cushion_depth:.3f}m...")
    
    model = mujoco.MjModel.from_xml_path(xml_path)
    configuration = mink.Configuration(model)
    ref_configuration = mink.Configuration(model)
    
    try:
        data_in = np.load(input_npz_path)
        original_qpos_traj = data_in['qpos'] if 'qpos' in data_in else data_in[data_in.files[0]]
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

    # Original Data Parameters
    orig_fps = 30.0
    orig_dt = 1.0 / orig_fps
    
    # High Frequency Parameters
    ik_dt = 1.0 / ik_fps
    
    # =========================================================
    # 1. Pre-Interpolation (Upsampling)
    # =========================================================
    n_frames_orig = len(original_qpos_traj)
    times_orig = np.arange(n_frames_orig) * orig_dt
    duration = times_orig[-1]
    
    times_ik = np.arange(0, duration, ik_dt)
    
    # Use CubicSpline for smooth interpolation of reference trajectory
    cs = CubicSpline(times_orig, original_qpos_traj, axis=0)
    ref_qpos_traj_high = cs(times_ik)
    
    # Scale the forced landing frame index to the new frequency
    if force_landing_frame is not None:
        force_landing_frame_high = int(force_landing_frame * (ik_fps / orig_fps))
    else:
        force_landing_frame_high = None

    # =========================================================
    # 2. Mink Task Setup
    # =========================================================
    pelvis_task = mink.FrameTask("pelvis", "body", position_cost=[1000.0, 1000.0, 2000.0], orientation_cost=50.0, lm_damping=1.0)
    posture_task = mink.PostureTask(model, cost=10.0)
    
    contact_body_names = [
        "left_ankle_roll_link", "right_ankle_roll_link", 
        "left_wrist_yaw_link", "right_wrist_yaw_link",
        "left_knee_link", "right_knee_link",
        "left_elbow_link", "right_elbow_link"
    ]
    
    contact_tasks = {
        name: mink.FrameTask(name, "body", position_cost=0.0, orientation_cost=0.0, lm_damping=1.0)
        for name in contact_body_names
    }

    all_tasks = [pelvis_task, posture_task] + list(contact_tasks.values())
    limits = [mink.ConfigurationLimit(model)]

    # Initialize Cushion Manager with the NEW dt
    cushion_manager = LandingCushionManager(ik_dt, cushion_depth=cushion_depth, cushion_duration=0.3, recovery_duration=0.5)
    
    new_qpos_traj = []
    
    # Initialize configurations with the first frame of interpolated data
    configuration.update(ref_qpos_traj_high[0])
    ref_configuration.update(ref_qpos_traj_high[0])
    
    contact_lock_poses = {name: None for name in contact_body_names}
    prev_pelvis_z = ref_configuration.get_transform_frame_to_world("pelvis", "body").translation()[2]
    
    force_height_targets = {} 

    # =========================================================
    # 3. High-Frequency IK Loop
    # =========================================================
    for i, raw_q in enumerate(ref_qpos_traj_high):
        ref_configuration.update(raw_q)
        
        curr_pelvis_z = ref_configuration.get_transform_frame_to_world("pelvis", "body").translation()[2]
        # Calculate vertical velocity from reference (can also differentiate spline)
        pelvis_vel_z = (curr_pelvis_z - prev_pelvis_z) / ik_dt
        prev_pelvis_z = curr_pelvis_z
        
        is_falling = pelvis_vel_z < -0.5 
        
        is_grounded_any = False
        
        # --- Contact Detection & Constraint Logic (Same as before) ---
        for name in contact_body_names:
            t_pose_ref = ref_configuration.get_transform_frame_to_world(name, "body")
            h_ref = t_pose_ref.translation()[2]
            
            t_pose_curr = configuration.get_transform_frame_to_world(name, "body")
            h_curr = t_pose_curr.translation()[2]
            
            check_h = min(h_ref, h_curr)
            
            is_landing_phase = (cushion_manager.state != "idle")
            is_foot = "ankle" in name
            threshold = 0.05 if is_foot else 0.05
            
            cond_normal = (check_h < threshold) and (is_falling or is_landing_phase)
            cond_clipping = (check_h < 0.025) and (h_ref < 0.1)
            
            should_contact = cond_normal or cond_clipping
            
            was_locked = contact_lock_poses[name] is not None
            should_maintain = was_locked and (h_ref < threshold)
            
            is_contact = should_contact or should_maintain
            
            if is_contact:
                is_grounded_any = True
                
                if not was_locked:
                    current_pose = configuration.get_transform_frame_to_world(name, "body")
                    contact_lock_poses[name] = current_pose
                
                # Bi-lateral foot check logic
                if "ankle" in name:
                    other_foot = "right_ankle_roll_link" if name == "left_ankle_roll_link" else "left_ankle_roll_link"
                    if contact_lock_poses[other_foot] is None:
                        # Need to check other foot current height
                        # Note: ref_configuration update is fast, but we need current config context
                        # Since configuration is updated inplace, we can check it directly
                        # But wait, looking at other foot body might require FK on current q
                        # mink configuration holds current q, so yes.
                        t_pose_other = configuration.get_transform_frame_to_world(other_foot, "body")
                        h_other = t_pose_other.translation()[2]
                        
                        if h_other < 0.20:
                            force_height_targets[other_foot] = contact_lock_poses[name].translation()[2]
            else:
                contact_lock_poses[name] = None

        # [Override] Force landing detection using scaled frame index
        if force_landing_frame_high is not None:
            is_grounded_any = (i >= force_landing_frame_high)
        
        z_offset = cushion_manager.update(is_grounded_any)

        # --- Task Weights Update ---
        if cushion_manager.state == "cushioning":
            posture_task.set_cost(0.1)
        elif cushion_manager.state == "recovering":
            posture_task.set_cost(1.0)
        else:
            posture_task.set_cost(100.0)
        
        # --- Task Targets Update ---
        ref_pelvis_pose = ref_configuration.get_transform_frame_to_world("pelvis", "body")
        target_pelvis_pos = ref_pelvis_pose.translation().copy()
        target_pelvis_pos[2] += z_offset 
        
        target_pelvis_se3 = mink.SE3.from_rotation_and_translation(ref_pelvis_pose.rotation(), target_pelvis_pos)
        pelvis_task.set_target(target_pelvis_se3)
        
        posture_task.set_target(raw_q)

        # --- Contact Tasks Update ---
        for name, task in contact_tasks.items():
            if contact_lock_poses[name] is not None:
                locked_pos = contact_lock_poses[name].translation()
                
                ref_pose = ref_configuration.get_transform_frame_to_world(name, "body")
                ref_z = ref_pose.translation()[2]
                ref_rot = ref_pose.rotation()
                
                target_z = min(locked_pos[2], ref_z)
                target_z = max(target_z, 0.03) 
                
                target_pos = np.array([locked_pos[0], locked_pos[1], target_z])
                target_se3 = mink.SE3.from_rotation_and_translation(ref_rot, target_pos)
                
                task.set_target(target_se3)
                task.set_position_cost([10000.0, 10000.0, 5000.0]) 
                task.set_orientation_cost(10.0) 
            else:
                t_pose = ref_configuration.get_transform_frame_to_world(name, "body")
                
                if name in force_height_targets:
                    target_z = force_height_targets[name]
                    pos = t_pose.translation().copy()
                    pos[2] = target_z
                    t_pose = mink.SE3.from_rotation_and_translation(t_pose.rotation(), pos)
                    task.set_position_cost(1000.0)
                else:
                    task.set_position_cost(200.0) 
                
                task.set_target(t_pose)
                task.set_orientation_cost(20.0)

        # --- Solve IK ---
        try:
            vel = mink.solve_ik(
                configuration, 
                all_tasks, 
                ik_dt, 
                solver="daqp", 
                limits=limits,
                damping=1e-3
            )
            configuration.integrate_inplace(vel, ik_dt)
            new_qpos_traj.append(configuration.q.copy())
        except Exception as e:
            # Fallback: keep previous pose
            new_qpos_traj.append(configuration.q.copy())

    # =========================================================
    # 4. Downsample back to Original Frequency
    # =========================================================
    new_qpos_traj = np.array(new_qpos_traj)
    
    # Create time grid for the actually generated high-freq trajectory
    # Note: len(new_qpos_traj) should match len(times_ik)
    times_generated = np.arange(len(new_qpos_traj)) * ik_dt
    
    # Interpolate back to original times
    # We use times_orig which corresponds to the input 30Hz frames
    # Ensure times_orig is within the range of times_generated to avoid extrapolation errors
    valid_times_mask = (times_orig <= times_generated[-1])
    times_orig_valid = times_orig[valid_times_mask]
    
    cs_res = CubicSpline(times_generated, new_qpos_traj, axis=0)
    output_qpos_traj = cs_res(times_orig_valid)
    
    print(f"Downsampled from {len(new_qpos_traj)} frames ({ik_fps}Hz) back to {len(output_qpos_traj)} frames ({orig_fps}Hz).")

    # Save output with original frequency
    np.savez(output_npz_path, qpos=output_qpos_traj, fps=orig_fps)
    return True

