import os
import sys

def generate_video(xml_path, npz_path, output_video_path, width=960, height=540, fps=30):
    """
    Generates a video from a Mujoco simulation based on an NPZ trajectory.
    """
    # Try to import MuJoCoAnimationPlayer
    try:
        from visualization.mujoco_player_new import MuJoCoAnimationPlayer
    except ImportError:
        # If not found in current path, try adding the directory of this script or parent
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        try:
            from visualization.mujoco_player_new import MuJoCoAnimationPlayer
        except ImportError:
            print("Error: MuJoCoAnimationPlayer could not be imported.")
            return False

    try:
        print(f"Generating video for {os.path.basename(npz_path)}...")
        # Adjusted for the specific MuJoCoAnimationPlayer implementation used in this workspace
        player = MuJoCoAnimationPlayer(xml_path=str(xml_path), npz_path=str(npz_path))
        player.save_video(str(output_video_path), fps=fps, width=width, height=height)
        # print(f"Video saved: {os.path.basename(output_video_path)}") # save_video already prints
        return True
    except TypeError as e:
        print(f"Video generation signature mismatch: {e}")
        # Fallback for alternative implementation if present in history, but based on file read this is the correct one.
        return False
    except Exception as e:
        print(f"Video generation failed: {e}")
        return False
