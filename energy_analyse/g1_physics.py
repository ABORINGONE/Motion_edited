import numpy as np
import mujoco
import re

# -----------------------------------------------------------------------------
# Physics Parameters from whole_body_tracking/robots/g1.py
# -----------------------------------------------------------------------------
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

# Calculated Damping (2 * ratio * armature * omega)
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

# Joint Config Mapping (Regex -> (Armature, Damping))
# Based on usage in g1.py ImplicitActuatorCfg
JOINT_PHYSICS_CONFIG = [
    # Legs (7520_14 / 7520_22)
    (r".*_hip_yaw_joint", ARMATURE_7520_14, DAMPING_7520_14),
    (r".*_hip_roll_joint", ARMATURE_7520_22, DAMPING_7520_22),
    (r".*_hip_pitch_joint", ARMATURE_7520_14, DAMPING_7520_14),
    (r".*_knee_joint", ARMATURE_7520_22, DAMPING_7520_22),
    
    # Feet (5020 x 2)
    (r".*_ankle_pitch_joint", 2.0 * ARMATURE_5020, 2.0 * DAMPING_5020),
    (r".*_ankle_roll_joint", 2.0 * ARMATURE_5020, 2.0 * DAMPING_5020),
    
    # Waist (5020 x 2)
    (r"waist_roll_joint", 2.0 * ARMATURE_5020, 2.0 * DAMPING_5020),
    (r"waist_pitch_joint", 2.0 * ARMATURE_5020, 2.0 * DAMPING_5020),
    
    # Waist Yaw (7520_14)
    (r"waist_yaw_joint", ARMATURE_7520_14, DAMPING_7520_14),
    
    # Arms (5020 usually, wrist pitch/yaw 4010)
    (r".*_shoulder_pitch_joint", ARMATURE_5020, DAMPING_5020),
    (r".*_shoulder_roll_joint", ARMATURE_5020, DAMPING_5020),
    (r".*_shoulder_yaw_joint", ARMATURE_5020, DAMPING_5020),
    (r".*_elbow_joint", ARMATURE_5020, DAMPING_5020),
    (r".*_wrist_roll_joint", ARMATURE_5020, DAMPING_5020),
    
    # Wrists (4010)
    (r".*_wrist_pitch_joint", ARMATURE_4010, DAMPING_4010),
    (r".*_wrist_yaw_joint", ARMATURE_4010, DAMPING_4010),
]

def apply_g1_physics(model: mujoco.MjModel):
    """
    Overwrites the MuJoCo model's armature and damping parameters 
    to match the specific values used in whole_body_tracking (Isaac Lab).
    """
    print("\n[Physics Fix] Applying precise G1 Armature/Damping parameters...")
    
    count = 0
    # Iterate over all joints
    for i in range(model.njnt):
        # mj_id2name can return None for some aux joints, but G1 joints are named
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if not name:
            continue
            
        # Find match
        matched = False
        armature = 0.0
        damping = 0.0
        
        for pattern, arm, damp in JOINT_PHYSICS_CONFIG:
            if re.fullmatch(pattern, name) or re.match(pattern, name):
                armature = arm
                damping = damp
                matched = True
                break
        
        if matched:
            # DOF Address
            dof_adr = model.jnt_dofadr[i]
            
            # Apply
            # Note: G1 joints are 1-DOF, so dof_adr is unique scalar index
            model.dof_armature[dof_adr] = armature
            model.dof_damping[dof_adr] = damping
            count += 1
            # print(f"  -> Updated {name}: Arm={armature:.5f}, Damp={damping:.5f}")
            
    print(f"[Physics Fix] Updated physics for {count} joints.")
