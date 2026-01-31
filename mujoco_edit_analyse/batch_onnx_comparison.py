import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to find imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from energy_analyse.analysis import run_onnx_energy_analysis

def plot_comparison(res_base, res_curr, filename, title_suffix=""):
    plt.figure(figsize=(10, 12))
    
    # Common Time Axis (assuming similar duration, or we plot against their own time)
    t_base = res_base["time"]
    t_curr = res_curr["time"]
    
    # 1. Cumulative Work Comparison
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t_base, res_base["work"], 'k--', label=f'Baseline Work ({res_base["total_work"]:.1f}J)', alpha=0.7)
    ax1.plot(t_curr, res_curr["work"], 'r-', label=f'Current Work ({res_curr["total_work"]:.1f}J)')
    
    # Contact Regions (Current)
    if "contact" in res_curr:
        import matplotlib.transforms as mtransforms
        trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
        # Fill where contact is True
        is_con = np.array(res_curr["contact"], dtype=bool)
        if len(is_con) == len(t_curr):
             ax1.fill_between(t_curr, 0, 1, where=is_con, facecolor='green', alpha=0.1, transform=trans, label='Contact (Current)')

    ax1.set_title(f"Cumulative Work Comparison {title_suffix}")
    ax1.set_ylabel("Work (J)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Effort (Torque Squared) Comparison
    ax2 = plt.subplot(3, 1, 2)
    # Smoothed for visibility? No, raw is better for spikes.
    ax2.plot(t_base, res_base["torque"], 'k--', label='Baseline Effort', alpha=0.5, linewidth=1)
    ax2.plot(t_curr, res_curr["torque"], 'b-', label='Current Effort', alpha=0.8, linewidth=1)
    
    ax2.set_title("Instantaneous Effort (Tau^2)")
    ax2.set_ylabel("Effort")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 3. Total Mechanical Energy Comparison
    ax3 = plt.subplot(3, 1, 3)
    
    # Calc Totals
    min_pot_base = np.min(res_base["energy_pot"])
    tot_base = (np.array(res_base["energy_pot"]) - min_pot_base) + np.array(res_base["energy_kin"])
    
    min_pot_curr = np.min(res_curr["energy_pot"])
    tot_curr = (np.array(res_curr["energy_pot"]) - min_pot_curr) + np.array(res_curr["energy_kin"])
    
    ax3.plot(t_base, tot_base, 'k--', label='Baseline Total Energy', alpha=0.6)
    ax3.plot(t_curr, tot_curr, 'g-', label='Current Total Energy', alpha=0.8)
    
    ax3.set_title("Total Mechanical Energy (Relative)")
    ax3.set_ylabel("Energy (J)")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # --- Configuration ---
    root_dir = parent_dir # Motion_rebuild folder
    
    # Settings
    xml_file = os.path.join(root_dir, "assets", "scene_29dof.xml")
    onnx_file = os.path.join(root_dir, "onnx", "2026-01-30_15-12-39_v1.onnx")
    
    baseline_npz = os.path.join(root_dir, "data", "Kjump", "kjump-1to-1.npz")
    batch_dir = os.path.join(root_dir, "batch_output_high_freq")
    
    output_dir = os.path.join(batch_dir, "onnx_comparisons")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("="*60)
    print("BATCH ONNX ENERGY COMPARISON")
    print("="*60)
    print(f"ONNX Model: {os.path.basename(onnx_file)}")
    print(f"Baseline:   {os.path.basename(baseline_npz)}")
    print(f"Batch Dir:  {batch_dir}")
    print(f"Output Dir: {output_dir}")
    print("-" * 60)

    # 1. Run Baseline Analysis
    print("\n>>> Analyzing Baseline...")
    # NOTE: output_dir=None to prevent saving individual baseline plots in the comparison folder if desired, 
    # but run_onnx_energy_analysis saves to output_dir if provided. 
    # We can use a temp dir or just let it save to result/baseline.
    # Let's just suppress video to save time.
    
    # We create a specific folder for baseline logs if needed, or just run it.
    res_base = run_onnx_energy_analysis(
        xml_file, 
        onnx_file, 
        baseline_npz, 
        output_dir=os.path.join(output_dir, "baseline_logs"), # Keep logs separate
        record_video=False,
        npz_is_onnx_order=False
    )
    
    if res_base is None:
        print("Error: Baseline analysis failed.")
        return

    # 2. Iterate Batch Files
    # Look for Kjump_straight_leg_*.npz
    pattern = os.path.join(batch_dir, "Kjump_straight_leg_*cm.npz")
    files = glob.glob(pattern)
    
    # Sort files naturally (0cm, 1cm, ... 10cm)
    # Extract number
    def sort_key(f):
        base = os.path.basename(f)
        try:
            # Extract number between 'leg_' and 'cm'
            num = base.split('leg_')[1].split('cm')[0]
            return int(num)
        except:
            return 0
            
    files.sort(key=sort_key)
    
    print(f"\nFound {len(files)} files to process.")
    
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        print(f"\n[{i+1}/{len(files)}] Processing {filename}...")
        
        # Run Analysis
        # We don't need to record video for every batch comparison unless requested
        res_curr = run_onnx_energy_analysis(
            xml_file,
            onnx_file,
            file_path,
            output_dir=os.path.join(output_dir, "individual_logs"), # Suppress clutter
            record_video=True, 
            npz_is_onnx_order=False # Assuming generated NPZs are in standard format? 
            # Note: generated NPZs from landing_generator usually follow valid structure.
        )
        
        if res_curr is None:
            print(f"Skipping {filename} due to failure.")
            continue
            
        # Plot Comparison
        save_name = os.path.splitext(filename)[0] + "_vs_baseline.png"
        save_path = os.path.join(output_dir, save_name)
        
        depth_label = filename.split("measure_")[-1] if "measure" in filename else filename
        
        plot_comparison(res_base, res_curr, save_path, title_suffix=f"({filename})")
        print(f"Comparison saved: {save_path}")

    print("\nBatch Comparison Complete!")

if __name__ == "__main__":
    main()
