import os
import sys
import numpy as np

# Import the core logic from the adjacent script
# (Assuming this script is run from the 'scripts' folder or 'GMR' root correctly)
try:
    from gmr_refinement_core import run_refinement
except ImportError:
    # If running from GMR root, we might need 'scripts.gmr_refinement_core'
    # or ensure sys.path includes the current directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from gmr_refinement_core import run_refinement

# ==================================================================================
# CONFIGURATION
# ==================================================================================

# 1. Robot Name (must match GMR config keys, e.g., 'unitree_g1', 'unitree_h1')
ROBOT_NAME = "unitree_g1"

# 2. Input Motion File Path
# Can be absolute or relative path
INPUT_MOTION_PATH = "my_data/kjump/motion.pkl"  # <-- CHANGE THIS TO YOUR FILE

# 3. Output Configuration
# If None, it will automatically append "_refined" to the input filename
OUTPUT_MOTION_PATH = None 


# ==================================================================================
# USER MODIFICATION LOGIC
# ==================================================================================

def user_modify_motion(frame_idx, root_pos, root_rot_xyzw, dof_pos):
    """
    Define how you want to modify the motion frame by frame.
    
    Args:
        frame_idx (int): The current frame number (0-based).
        root_pos (np.array): Shape (3,), Position [x, y, z]
        root_rot_xyzw (np.array): Shape (4,), Quaternion [x, y, z, w]
        dof_pos (np.array): Shape (N,), Joint angles in radians.
        
    Returns:
        tuple: (modified_root_pos, modified_root_rot_xyzw, modified_dof_pos)
    """
    
    # Clone arrays to avoid side effects if generic referencing is used somewhere
    # (Not strictly necessary if fully overwriting, but good practice)
    mod_dof_pos = dof_pos.copy()
    mod_root_pos = root_pos.copy()
    mod_root_rot = root_rot_xyzw.copy()

    # -------------------------------------------------------------------------
    # WRITE YOUR CUSTOM LOGIC BELOW
    # -------------------------------------------------------------------------
    
    # EXAMPLE 1: Apply a constant offset to a specific joint (e.g., Joint 10)
    # mod_dof_pos[10] += 0.2
    
    # EXAMPLE 2: Make the robot squat lower by reducing Z height
    # mod_root_pos[2] -= 0.05
    
    # EXAMPLE 3: Freeze the arms (assuming arm indices are known, e.g., 15-20)
    # mod_dof_pos[15:21] = 0.0

    # -------------------------------------------------------------------------
    
    return mod_root_pos, mod_root_rot, mod_dof_pos


# ==================================================================================
# MAIN EXECUTION (Do not modify unless necessary)
# ==================================================================================

if __name__ == "__main__":
    
    # Resolve paths
    if not os.path.exists(INPUT_MOTION_PATH):
        print(f"[ERROR] Input file not found: {INPUT_MOTION_PATH}")
        print("Please edit 'INPUT_MOTION_PATH' in this script.")
        sys.exit(1)
        
    if OUTPUT_MOTION_PATH is None:
        base, ext = os.path.splitext(INPUT_MOTION_PATH)
        output_path = f"{base}_refined{ext}"
    else:
        output_path = OUTPUT_MOTION_PATH
        
    print(f"--- Starting Motion Refinement ---")
    print(f"Robot:  {ROBOT_NAME}")
    print(f"Input:  {INPUT_MOTION_PATH}")
    print(f"Output: {output_path}")
    print(f"----------------------------------")
    
    run_refinement(
        robot_name=ROBOT_NAME,
        input_path=INPUT_MOTION_PATH,
        output_path=output_path,
        modify_func=user_modify_motion
    )
    
    print("\nAll done!")
