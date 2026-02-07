import os
import sys
import numpy as np
import argparse

# Add visualization module path
# Motion_rebuild folder
MOTION_REBUILD_PATH = "/mnt/c/Users/not a fat cat/Desktop/Motion_rebuild"
if MOTION_REBUILD_PATH not in sys.path:
    sys.path.append(MOTION_REBUILD_PATH)

try:
    from visualization.mujoco_player_new import MuJoCoAnimationPlayer
except ImportError as e:
    print(f"Failed to import MuJoCoAnimationPlayer: {e}")
    sys.exit(1)

def convert_gmr_to_qpos(gmr_npz_path, output_npz_path):
    """
    Converts GMR output (root_pos, root_rot[xyzw], dof_pos) to Mujoco qpos (pos, rot[wxyz], dof).
    """
    data = np.load(gmr_npz_path)
    root_pos = data["root_pos"]
    root_rot_xyzw = data["root_rot"]
    dof_pos = data["dof_pos"]
    
    # Convert xyzw -> wxyz
    # Mujoco expects wxyz
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    
    qpos_list = []
    for i in range(len(root_pos)):
        q = np.concatenate([root_pos[i], root_rot_wxyz[i], dof_pos[i]])
        qpos_list.append(q)
        
    np.savez(output_npz_path, qpos=np.array(qpos_list), fps=data.get("fps", 30))
    print(f"converted {len(qpos_list)} frames.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input GMR .npz file")
    parser.add_argument("--xml", "-x", type=str, required=True, help="Robot XML file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output MP4 file")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    # Create temporary converted file
    temp_npz = args.input.replace(".npz", "_temp_qpos.npz")
    try:
        convert_gmr_to_qpos(args.input, temp_npz)
        
        print(f"Initializing Player with {args.xml}...")
        player = MuJoCoAnimationPlayer(args.xml, temp_npz, add_ground=True)
        
        print(f"Rendering to {args.output}...")
        player.save_video(
            output_path=args.output,
            fps=30,
            width=1280,
            height=720,
            camera_mode="follow", # Assuming camera following is desired
            camera_distance=3.5,
            camera_elevation=-20,
            camera_azimuth=45
        )
        print("Done.")
        
    except Exception as e:
        print(f"Error during rendering: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_npz):
            os.remove(temp_npz)

if __name__ == "__main__":
    main()
