import re

# -----------------------------------------------------------------------------
# G1 Robot Configuration (Copied/Adapted from g1.py)
# -----------------------------------------------------------------------------
# Default Joint Positions (matches g1.py)
DEFAULT_JOINT_POS = {
    "left_hip_pitch_joint": -0.312,
    "right_hip_pitch_joint": -0.312,
    "left_knee_joint": 0.669,
    "right_knee_joint": 0.669,
    "left_ankle_pitch_joint": -0.363,
    "right_ankle_pitch_joint": -0.363,
    "left_elbow_joint": 0.6,
    "right_elbow_joint": 0.6,
    "left_shoulder_roll_joint": 0.2,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
    "right_shoulder_pitch_joint": 0.2,
    # Others are 0.0
}

# Physical Params for Action Scale Calculation
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

# Simplified Actuator Config mapping (Regex -> Value)
# Used to compute scale = 0.25 * effort / stiffness
ACTUATOR_CONFIGS = [
    # Legs
    (r".*_hip_yaw_joint", 88.0, STIFFNESS_7520_14),
    (r".*_hip_roll_joint", 139.0, STIFFNESS_7520_22),
    (r".*_hip_pitch_joint", 88.0, STIFFNESS_7520_14),
    (r".*_knee_joint", 139.0, STIFFNESS_7520_22),
    # Feet
    (r".*_ankle_pitch_joint", 50.0, 2.0 * STIFFNESS_5020),
    (r".*_ankle_roll_joint", 50.0, 2.0 * STIFFNESS_5020),
    # Waist
    (r"waist_roll_joint", 50.0, 2.0 * STIFFNESS_5020),
    (r"waist_pitch_joint", 50.0, 2.0 * STIFFNESS_5020),
    (r"waist_yaw_joint", 88.0, STIFFNESS_7520_14),
    # Arms
    (r".*_shoulder_pitch_joint", 25.0, STIFFNESS_5020),
    (r".*_shoulder_roll_joint", 25.0, STIFFNESS_5020),
    (r".*_shoulder_yaw_joint", 25.0, STIFFNESS_5020),
    (r".*_elbow_joint", 25.0, STIFFNESS_5020),
    (r".*_wrist_roll_joint", 25.0, STIFFNESS_5020),
    (r".*_wrist_pitch_joint", 5.0, STIFFNESS_4010),
    (r".*_wrist_yaw_joint", 5.0, STIFFNESS_4010),
]

def get_action_scale(joint_name):
    effort = 0.0
    stiffness = 1.0
    found = False
    for pattern, e, k in ACTUATOR_CONFIGS:
        if re.match(pattern, joint_name):
            effort = e
            stiffness = k
            found = True
            break
    if found and stiffness > 0:
        return 0.25 * effort / stiffness
    return 1.0 # Default if not found (should not happen for standard G1)

def get_default_pos(joint_name):
    # Regex matching for default content
    for pattern, val in DEFAULT_JOINT_POS.items():
        if re.fullmatch(pattern, joint_name) or pattern == joint_name:
            return val
    return 0.0
