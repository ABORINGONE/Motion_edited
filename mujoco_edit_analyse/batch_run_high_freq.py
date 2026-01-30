import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add parent directory to path to find imports if necessary
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our split modules
from landing_generator import generate_motion
from high_freq_analysis import ComplianceAnalyzerHighFreq, plot_comparison_high_freq
from video_generator import generate_video

def main():
    # Setup Paths
    # Setup Paths
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    root_dir = current_dir.parent
    
    xml_path = root_dir / "assets" / "g1_29dof.xml"
    input_npz = root_dir / "data" / "Kjump" / "kjump-1to-1_straight_leg.npz"
    output_dir = root_dir / "batch_output_high_freq"
    output_dir.mkdir(exist_ok=True)

    # Verify inputs
    if not input_npz.exists():
        print(f"Error: Input file not found at {input_npz}")
        return

    # Simulation Parameters
    # We use a very small dt for accurate energy analysis
    SIM_DT = 0.002 # 500Hz
    
    # Batch Parameters
    # 0cm to 20cm, step 1cm
    depths = np.arange(0.00, 0.21, 0.01)

    print("Initializing High-Freq Analyzer...")
    analyzer = ComplianceAnalyzerHighFreq(str(xml_path))
    
    print("Analyzing Original Trajectory...")
    # Analyze the input file once
    res_orig = analyzer.process_trajectory(str(input_npz), sim_dt=SIM_DT)
    
    if res_orig is None:
        print("Failed to analyze original trajectory. Exiting.")
        return

    total = len(depths)
    
    for idx, depth in enumerate(depths):
        depth_cm = int(round(depth * 100))
        print(f"\n[{idx+1}/{total}] Processing cushion depth: {depth_cm} cm")
        
        filename = f"Kjump_straight_leg_{depth_cm}cm"
        output_npz = output_dir / f"{filename}.npz"
        output_img = output_dir / f"{filename}_analysis.png"
        
        # 1. Motion Generation (using split module)
        success = generate_motion(
            str(xml_path), 
            str(input_npz), 
            str(output_npz), 
            cushion_depth=depth, 
            force_landing_frame=39 # Original 30Hz frame index
        )
        
        if not success:
            continue

        # 2. Video Generation (using split module)
        output_video = output_dir / f"{filename}.mp4"
        generate_video(
            str(xml_path), 
            str(output_npz), 
            str(output_video), 
            width=960, 
            height=540,
            fps=30 # Back to original FPS
        )

        # 3. High-Freq Analysis (using split module)
        print("Running analysis...")
        res_opt = analyzer.process_trajectory(
            str(output_npz), 
            sim_dt=SIM_DT
        )
        
        # 4. Plot Comparison
        if res_opt:
            plot_comparison_high_freq(res_orig, res_opt, str(output_img), title_suffix=f"({depth_cm}cm)")
            print(f"Plot saved: {output_img.name}")

    print(f"\nBatch processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
