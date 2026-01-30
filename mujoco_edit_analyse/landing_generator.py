import mujoco
import numpy as np
import mink
import sys
import os

# =========================================================
# Motion Generation Logic
# =========================================================

class LandingCushionManager:
    def __init__(self, dt, cushion_depth=0.15, cushion_duration=0.3, recovery_duration=0.5):
        self.dt = dt
        self.cushion_depth = cushion_depth
        self.cushion_steps = int(cushion_duration / dt)
        self.recovery_steps = int(recovery_duration / dt)
        
        self.state = "idle" 
        self.timer = 0
        self.current_offset = 0.0
        self.is_grounded = False
        self.prev_grounded = False

    def _logistic(self, t, k=10):
        x = k * (t - 0.5)
        val = 1 / (1 + np.exp(-x))
        v0 = 1 / (1 + np.exp(k/2))
        v1 = 1 / (1 + np.exp(-k/2))
        return (val - v0) / (v1 - v0)

    def update(self, is_grounded):
        if is_grounded and not self.prev_grounded:
            self.state = "cushioning"
            self.timer = 0
            # print(">>> Impact detected! Initiating cushion.")

        if self.state == "cushioning":
            progress = min(self.timer / self.cushion_steps, 1.0)
            factor = self._logistic(progress)
            self.current_offset = -self.cushion_depth * factor
            
            self.timer += 1
            if self.timer >= self.cushion_steps:
                self.state = "recovering"
                self.timer = 0
        
        elif self.state == "recovering":
            progress = min(self.timer / self.recovery_steps, 1.0)
            factor = self._logistic(progress)
            self.current_offset = -self.cushion_depth * (1.0 - factor)
            
            self.timer += 1
            if self.timer >= self.recovery_steps:
                self.state = "idle"
                self.current_offset = 0.0

        else:
            self.current_offset = 0.0

        self.prev_grounded = is_grounded
        return self.current_offset

def generate_motion(xml_path, input_npz_path, output_npz_path, cushion_depth, force_landing_frame=None):
    """
    Generates a new motion trajectory with cushion effect.
    """
    print(f"Generating motion for depth {cushion_depth:.3f}m...")
    model = mujoco.MjModel.from_xml_path(xml_path)
    configuration = mink.Configuration(model)
    ref_configuration = mink.Configuration(model)
    
    try:
        data_in = np.load(input_npz_path)
        original_qpos_traj = data_in['qpos'] if 'qpos' in data_in else data_in[data_in.files[0]]
    except Exception as e:
        print(f"Error loading data: {e}")
        return False

    dt = 1/30 

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

    cushion_manager = LandingCushionManager(dt, cushion_depth=cushion_depth, cushion_duration=0.3, recovery_duration=0.5)
    
    new_qpos_traj = []
    configuration.update(original_qpos_traj[0])
    
    contact_lock_poses = {name: None for name in contact_body_names}
    
    ref_configuration.update(original_qpos_traj[0])
    prev_pelvis_z = ref_configuration.get_transform_frame_to_world("pelvis", "body").translation()[2]

    for i, raw_q in enumerate(original_qpos_traj):
        ref_configuration.update(raw_q)
        
        curr_pelvis_z = ref_configuration.get_transform_frame_to_world("pelvis", "body").translation()[2]
        pelvis_vel_z = (curr_pelvis_z - prev_pelvis_z) / dt
        prev_pelvis_z = curr_pelvis_z
        
        is_falling = pelvis_vel_z < -0.5 # 放宽下落速度判定，因为直腿动作可能减缓了垂直速度 
        
        is_grounded_any = False
        force_height_targets = {} 
        
        for name in contact_body_names:
            t_pose_ref = ref_configuration.get_transform_frame_to_world(name, "body")
            h_ref = t_pose_ref.translation()[2]
            
            t_pose_curr = configuration.get_transform_frame_to_world(name, "body")
            h_curr = t_pose_curr.translation()[2]
            
            check_h = min(h_ref, h_curr)
            
            is_landing_phase = (cushion_manager.state != "idle")
            is_foot = "ankle" in name
            threshold = 0.05 if is_foot else 0.05
            
            # 判定条件：
            cond_normal = (check_h < threshold) and (is_falling or is_landing_phase)
            
            # [Fix] 穿模保护：只有当参考动作也在地面附近时才生效，防止起跳时被锁住
            cond_clipping = (check_h < 0.025) and (h_ref < 0.1)
            
            should_contact = cond_normal or cond_clipping
            
            was_locked = contact_lock_poses[name] is not None
            # [Fix] 保持接触：只有当参考动作也在地面附近时才保持，防止起跳时粘在地上
            should_maintain = was_locked and (h_ref < threshold)
            
            is_contact = should_contact or should_maintain
            
            if is_contact:
                is_grounded_any = True
                
                if not was_locked:
                    current_pose = configuration.get_transform_frame_to_world(name, "body")
                    contact_lock_poses[name] = current_pose
                
                if "ankle" in name:
                    other_foot = "right_ankle_roll_link" if name == "left_ankle_roll_link" else "left_ankle_roll_link"
                    if contact_lock_poses[other_foot] is None:
                        h_other = configuration.get_transform_frame_to_world(other_foot, "body").translation()[2]
                        if h_other < 0.20:
                            force_height_targets[other_foot] = contact_lock_poses[name].translation()[2]

            else:
                contact_lock_poses[name] = None

        # [Override] Force landing detection at specific frame if requested
        if force_landing_frame is not None:
            # If forced, we ignore the physical detection for the cushion trigger
            # but we might still want physical detection for foot locking?
            # The user said "instead of original logic", so we override the trigger signal.
            is_grounded_any = (i >= force_landing_frame)
        
        z_offset = cushion_manager.update(is_grounded_any)

        if cushion_manager.state == "cushioning":
            posture_task.set_cost(0.1)
        elif cushion_manager.state == "recovering":
            posture_task.set_cost(1.0)
        else:
            posture_task.set_cost(100.0)
        
        ref_pelvis_pose = ref_configuration.get_transform_frame_to_world("pelvis", "body")
        target_pelvis_pos = ref_pelvis_pose.translation().copy()
        target_pelvis_pos[2] += z_offset 
        
        target_pelvis_se3 = mink.SE3.from_rotation_and_translation(ref_pelvis_pose.rotation(), target_pelvis_pos)
        pelvis_task.set_target(target_pelvis_se3)
        
        posture_task.set_target(raw_q)

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
                # 提高 Z 轴权重以防止脚被压入地下，但不要太高以免阻止膝盖弯曲
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

        try:
            vel = mink.solve_ik(
                configuration, 
                all_tasks, 
                dt, 
                solver="daqp", 
                limits=limits,
                damping=1e-3
            )
            configuration.integrate_inplace(vel, dt)
            new_qpos_traj.append(configuration.q.copy())
        except Exception as e:
            new_qpos_traj.append(configuration.q.copy())

    np.savez(output_npz_path, qpos=np.array(new_qpos_traj), fps=1/dt)
    return True
