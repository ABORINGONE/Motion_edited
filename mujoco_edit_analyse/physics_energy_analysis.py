import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.interpolate import CubicSpline

def run_physics_energy_analysis(xml_path, npz_path, output_dir=None, sim_dt=0.002, 
                                # 默认增益配置 (可根据 G1 机器人实际文档调整)
                                default_kp=300.0, default_kd=10.0,
                                custom_gains=None):
    """
    运行基于物理的能量分析，支持分部位增益设置。
    """
    
    print(f"--- Starting Physics-Based Energy Analysis ---")
    print(f"Model: {xml_path}")
    print(f"Motion: {npz_path}")

    # 1. Load Model
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return
        
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Set simulation timestep
    model.opt.timestep = sim_dt
    
    # 2. Load Reference Motion
    if not os.path.exists(npz_path):
        print(f"Error: NPZ file not found at {npz_path}")
        return
        
    try:
        raw_data = np.load(npz_path, allow_pickle=True)
        if 'qpos' in raw_data:
            qpos_ref = raw_data['qpos']
        else:
            # Fallback
            qpos_ref = raw_data[raw_data.files[0]]
        print("Data keys:", raw_data.files)
    except Exception as e:
        print(f"Error loading NPZ: {e}")
        return

    n_frames = len(qpos_ref)
    # Assume 30fps for input motion if not specified (typical for mocap/generated)
    input_fps = 30.0 
    duration = (n_frames - 1) / input_fps
    
    print(f"Reference Motion: {n_frames} frames, {duration:.2f}s duration (assumed {input_fps} FPS)")
    
    # 3. Prepare Reference Splines (for continuous time tracking)
    time_ref = np.linspace(0, duration, n_frames)
    
    # Create CubicSpline for qpos
    # Note: Using simpler linear interpolation for quaternions if needed or just spline on raw components
    # Justification: For high-freq data, component-wise spline is a reasonable approximation for driving PD targets.
    spline_qpos = CubicSpline(time_ref, qpos_ref, axis=0)
    spline_qvel = spline_qpos.derivative(nu=1) # Analytical velocity from spline
    
    # 4. Identify Actuators, Prepare PD Gains
    # Map actuator index to joint index (qpos index)
    # model.actuator_trnid[:, 0] gives the joint id (qpos address logic needed later)
    # model.jnt_qposadr gives the address in qpos vector for a given joint ID
    
    actuator_joint_ids = model.actuator_trnid[:, 0]
    
    # Prepare gain arrays
    kp_array = np.full(model.nu, default_kp)
    kd_array = np.full(model.nu, default_kd)
    
    # Apply custom gains if provided
    if custom_gains:
        print("Applying custom gains based on joint names...")
        for i, joint_id in enumerate(actuator_joint_ids):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            matched = False
            if joint_name:
                for key, (kp, kd) in custom_gains.items():
                    if key in joint_name:
                        kp_array[i] = kp
                        kd_array[i] = kd
                        matched = True
                        break
    
    # 5. Simulation Loop
    sim_steps = int(duration / sim_dt)
    
    # Records
    time_log = []
    energy_pot_log = []
    energy_kin_log = []
    power_log = [] # Mechanical power (torque * velocity)
    work_log = []  # Accumulated work
    torque_squared_log = [] # Torque squared

    current_work = 0.0
    
    # Initialize state
    data.qpos[:] = qpos_ref[0]
    
    # If we have initial velocity from spline
    v_init = spline_qvel(0)
    data.qvel[:] = v_init
    
    mujoco.mj_forward(model, data)
    
    # Warmup to settle contacts? Maybe not needed if motion starts in air or standing.
    
    print(f"Simulating {sim_steps} steps...")
    
    for step in range(sim_steps):
        t = data.time
        
        # A. Get Reference State
        q_target = spline_qpos(t)
        v_target = spline_qvel(t)
        
        # B. Compute PD Control
        # We only apply control to actuated joints
        for i, joint_id in enumerate(actuator_joint_ids):
            # Find qpos index for this joint
            q_adr = model.jnt_qposadr[joint_id]
            dof_adr = model.jnt_dofadr[joint_id]
            
            # Simple 1DOF handling. 
            # Note: quaternion joints (free/ball) are usually not actuated directly in this simple loop
            # unless using specific actuators. Here we assume standard revolute/prismatic logic.
            
            current_q = data.qpos[q_adr]
            current_v = data.qvel[dof_adr]
            
            target_q = q_target[q_adr]
            target_v = v_target[dof_adr]
            
            # PD Law with per-actuator gains
            kp = kp_array[i]
            kd = kd_array[i]
            
            torque = kp * (target_q - current_q) + kd * (target_v - current_v)
            
            # Check limits
            ctrl_range = model.actuator_ctrlrange[i]
            if model.actuator_ctrllimited[i]:
                 torque = np.clip(torque, ctrl_range[0], ctrl_range[1])
            
            data.ctrl[i] = torque

        # C. Step Physics
        mujoco.mj_step(model, data)
        
        # D. Record Data
        # Energy
        mujoco.mj_energyPos(model, data)
        mujoco.mj_energyVel(model, data)
        
        pe = data.energy[0]
        ke = data.energy[1]
        
        # Power Calculation: Sum(|torque * velocity|)
        inst_power = 0.0
        inst_torque_sq = 0.0
        
        for i, joint_id in enumerate(actuator_joint_ids):
            dof_adr = model.jnt_dofadr[joint_id]
            vel = data.qvel[dof_adr]
            trq = data.ctrl[i]
            
            inst_power += abs(trq * vel) # Metabolic-like cost
            inst_torque_sq += trq ** 2
            
        current_work += inst_power * sim_dt
        
        if step % 500 == 0:
            print(f"Step {step}/{sim_steps}, t={t:.2f}s, Work={current_work:.2f}")

        time_log.append(t)
        energy_pot_log.append(pe)
        energy_kin_log.append(ke)
        power_log.append(inst_power)
        work_log.append(current_work)
        torque_squared_log.append(inst_torque_sq)

    # 6. Plotting
    if output_dir is None:
        output_dir = os.path.dirname(npz_path)
    
    base_name = os.path.splitext(os.path.basename(npz_path))[0]
    
    plt.figure(figsize=(12, 12))
    
    # Plot 1: Standard energies
    plt.subplot(3, 1, 1)
    plt.plot(time_log, energy_pot_log, label='Potential E', color='blue')
    plt.plot(time_log, energy_kin_log, label='Kinetic E', color='orange')
    plt.plot(time_log, np.array(energy_pot_log) + np.array(energy_kin_log), label='Total Mechanical E', color='green', linestyle='--')
    plt.title('System Mechanical Energy (Physics Simulation)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Work / Consumption
    plt.subplot(3, 1, 2)
    plt.plot(time_log, work_log, label='Cumulative Work (Σ|τ·ω|dt)', color='red')
    plt.title('Cumulative Energy Cost (Work)')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Instantaneous Effort
    plt.subplot(3, 1, 3)
    plt.plot(time_log, torque_squared_log, label='Effort (Στ²)', color='purple')
    plt.title('Instantaneous Effort (Sum of Squared Torques)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{base_name}_physics_energy.png")
    plt.savefig(save_path)
    print(f"Results saved to: {save_path}")
    
if __name__ == "__main__":
    # Example Usage
    target_xml = r"c:\Users\not a fat cat\Desktop\Motion_rebuild\assets\g1_29dof.xml"
    target_npz = r"c:\Users\not a fat cat\Desktop\Motion_rebuild\batch_output\kjump_cushion_0cm.npz"
    
    # 针对 G1 机器人的建议参数示例 (如果 Retargeting 过程没有使用特定增益)
    gains_config = {
        'ankle': (200, 5),
        'knee':  (300, 10),
        'hip':   (300, 10),
        'waist': (200, 10),
        'shoulder': (100, 5),
        'elbow': (80, 5),
        'wrist': (50, 2)
    }

    run_physics_energy_analysis(target_xml, target_npz, custom_gains=gains_config)
