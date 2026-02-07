import os
import sys
import numpy as np
import argparse

# Ensure we can import from the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Add visualization module path (Motion_rebuild folder)
MOTION_REBUILD_PATH = "/mnt/c/Users/not a fat cat/Desktop/Motion_rebuild"
if MOTION_REBUILD_PATH not in sys.path:
    sys.path.append(MOTION_REBUILD_PATH)

try:
    from visualization.mujoco_player_new import MuJoCoAnimationPlayer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    print("Warning: visualization.mujoco_player_new not found. Video generation will be disabled.")

from gmr_refinement_core import run_refinement

# ==================================================================================
# PARAMETERS
# ==================================================================================
ROBOT_NAME = "unitree_g1"
ROBOT_XML_PATH = os.path.join(MOTION_REBUILD_PATH, "assets", "g1_29dof.xml")
# Assuming input file location based on GMR folder structure
INPUT_MOTION_PATH = os.path.abspath(os.path.join(current_dir, "../my_data/Kjump/kjump-1to-1.npz"))
OUTPUT_DIR_NAME = "batch_output_gmr"
OUTPUT_DIR = os.path.abspath(os.path.join(current_dir, f"../{OUTPUT_DIR_NAME}"))
FPS = 30.0
DT = 1.0 / FPS
FORCE_LANDING_FRAME = 39

# ==================================================================================
# LOGIC
# ==================================================================================

class LandingCushionManager:
    """
    Manages the crouching cushion effect relative to a generated offset.
    Based on logic from mujoco_edit_analyse/landing_generator.py
    """
    def __init__(self, dt, cushion_depth=0.15, cushion_duration=0.3, recovery_duration=0.5):
        self.dt = dt
        self.cushion_depth = cushion_depth
        self.cushion_steps = int(cushion_duration / dt)
        self.recovery_steps = int(recovery_duration / dt)
        
        self.state = "idle" 
        self.timer = 0
        self.current_offset = 0.0
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

        if self.state == "cushioning":
            # 0 to 1
            if self.cushion_steps > 0:
                progress = min(self.timer / self.cushion_steps, 1.0)
                factor = self._logistic(progress)
                self.current_offset = -self.cushion_depth * factor
                
                self.timer += 1
                if self.timer >= self.cushion_steps:
                    self.state = "recovering"
                    self.timer = 0
            else:
                 self.current_offset = -self.cushion_depth
                 self.state = "recovering" # Immediately transition if 0 duration
        
        elif self.state == "recovering":
            if self.recovery_steps > 0:
                progress = min(self.timer / self.recovery_steps, 1.0)
                factor = self._logistic(progress)
                self.current_offset = -self.cushion_depth * (1.0 - factor)
                
                self.timer += 1
                if self.timer >= self.recovery_steps:
                    self.state = "idle"
                    self.current_offset = 0.0
            else:
                self.state = "idle"
                self.current_offset = 0.0

        else:
            self.current_offset = 0.0

        self.prev_grounded = is_grounded
        return self.current_offset

def render_video_for_file(npz_path, output_mp4_path):
    if not VISUALIZER_AVAILABLE:
        print("[Error] Visualizer not available. Cannot generate video.")
        return

    if not os.path.exists(ROBOT_XML_PATH):
        print(f"[Error] Robot XML not found at {ROBOT_XML_PATH}")
        return

    # Create temp qpos file
    temp_npz = npz_path.replace(".npz", "_temp_qpos.npz")
    try:
        # Load GMR data
        data = np.load(npz_path)
        root_pos = data["root_pos"]
        root_rot_xyzw = data["root_rot"]
        dof_pos = data["dof_pos"]
        
        # xyzw -> wxyz
        # Mujoco expects wxyz
        root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
        
        qpos_list = []
        for i in range(len(root_pos)):
            q = np.concatenate([root_pos[i], root_rot_wxyz[i], dof_pos[i]])
            qpos_list.append(q)
            
        np.savez(temp_npz, qpos=np.array(qpos_list), fps=data.get("fps", 30))
        
        print(f"  -> Rendering video to {os.path.basename(output_mp4_path)}...")
        player = MuJoCoAnimationPlayer(ROBOT_XML_PATH, temp_npz, add_ground=True)
        player.save_video(
            output_path=output_mp4_path,
            fps=30,
            width=1280,
            height=720,
            camera_mode="follow",
            camera_distance=3.5,
            camera_elevation=-20,
            camera_azimuth=45
        )
    except Exception as e:
        print(f"  [Error] Failed to render video: {e}")
    finally:
        if os.path.exists(temp_npz):
            os.remove(temp_npz)

def process_batch(gen_video=False):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    depths = np.arange(-0.10, 0.21, 0.01) # 0 to 20 cm
    
    print(f"--- GMR Batch Landing Adjustment ---")
    print(f"Input: {INPUT_MOTION_PATH}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"Total Variations: {len(depths)}")
    print(f"Generate Video: {gen_video}")
    print(f"------------------------------------")
    
    for depth in depths:
        depth_cm = int(round(depth * 100))
        output_filename = f"Kjump_straight_leg_{depth_cm}cm.npz" 
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        print(f"\nProcessing depth: {depth_cm} cm...")
        
        # Instance per run
        cushion_manager = LandingCushionManager(DT, cushion_depth=depth)
        
        # State to track identified feet keys
        runtime_state = {
            "feet_keys": []
        }
        
        def modify_func(frame_idx, root_pos, root_rot, dof_pos):
             # Force landing detection (simple frame trigger)
             is_landing = (frame_idx >= FORCE_LANDING_FRAME)
             
             z_offset = cushion_manager.update(is_landing)
             
             mod_root_pos = root_pos.copy()
             mod_root_pos[2] += z_offset
             
             return mod_root_pos, root_rot, dof_pos

        def post_fk_func(frame_idx, source_data):
            # Identify feet keys on first run if empty
            if not runtime_state["feet_keys"]:
                all_keys = list(source_data.keys())
                # Heuristic: "Foot", "Ankle", "Toe"
                for k in all_keys:
                    k_lower = k.lower()
                    if "foot" in k_lower or "ankle" in k_lower or "toe" in k_lower:
                        runtime_state["feet_keys"].append(k)
                
                if frame_idx == 0:
                     if runtime_state["feet_keys"]:
                        print(f"  [Info] Identified stabilizer keys: {runtime_state['feet_keys']}")
                     else:
                        print(f"  [Warning] No foot keys found in source_data! Available: {all_keys}")

            # Apply correction
            # Since root was lowered by z_offset (negative), 
            # the source feet (derived from FK) are now lower by that amount relative to world.
            # To simulate "feet planted on ground" while body lowers, 
            # we must Shift Up the feet targets by the same amount we lowered the root.
            
            z_offset = cushion_manager.current_offset
            
            # z_offset is typically negative (e.g. -0.10).
            # We want to add (-z_offset) to Z.
            correction = -z_offset
            
            if abs(correction) > 0.0001:
                for k in runtime_state["feet_keys"]:
                    if k in source_data:
                        pos, rot = source_data[k]
                        new_pos = pos.copy()
                        new_pos[2] += correction
                        source_data[k] = (new_pos, rot)
            
            return source_data

        try:
            run_refinement(
                robot_name=ROBOT_NAME,
                input_path=INPUT_MOTION_PATH,
                output_path=output_path,
                modify_func=modify_func,
                post_fk_func=post_fk_func
            )
        except Exception as e:
            print(f"Failed to process depth {depth_cm}: {e}")
            import traceback
            traceback.print_exc()

        if gen_video:
            video_path = output_path.replace(".npz", ".mp4")
            render_video_for_file(output_path, video_path)

    print("\nBatch processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process landing motions with GMR refinement.")
    parser.add_argument("--video", "-v", action="store_true", help="Generate video for each output")
    args = parser.parse_args()
    
    process_batch(gen_video=args.video)
