import mujoco
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.interpolate import CubicSpline
from pathlib import Path

# Ensure we can import from parent directory if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ComplianceAnalyzerHighFreq:
    def __init__(self, xml_path):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"XML model not found: {xml_path}")
            
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        try:
            self.pelvis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        except:
            self.pelvis_id = 1
            
    def process_trajectory(self, npz_path, orig_dt=1/30, sim_dt=0.002):
        print(f"Processing: {os.path.basename(npz_path)} with sim_dt={sim_dt}s")
        try:
            raw_data = np.load(npz_path)
            # Handle different npz keys
            if 'qpos' in raw_data:
                qpos_traj = raw_data['qpos']
            else:
                # Fallback to first array if qpos not found
                qpos_traj = raw_data[raw_data.files[0]]
        except Exception as e:
            print(f"Failed to load {npz_path}: {e}")
            return None

        n_frames_orig = len(qpos_traj)
        times_orig = np.arange(n_frames_orig) * orig_dt
        
        # Target high frequency time grid
        duration = times_orig[-1]
        times_fine = np.arange(0, duration, sim_dt)
        
        print(f"Interpolating from {n_frames_orig} frames (30Hz) to {len(times_fine)} frames ({1/sim_dt:.0f}Hz)...")
        
        # Cubic Spline Interpolation
        # axis=0 interpolates along the time dimension for each joint
        cs = CubicSpline(times_orig, qpos_traj, axis=0)
        
        qpos_fine = cs(times_fine)
        
        # Calculate qvel (velocity)
        # Note: qpos (nq) and qvel (nv) dimensions might differ (e.g. quaternions)
        # If nq != nv, we cannot simply use spline derivative.
        
        nq = self.model.nq
        nv = self.model.nv
        n_fine = len(qpos_fine)
        
        if nq == nv:
            # Simple case: 1-to-1 mapping (no quaternions)
            print(" Dimensions match (nq=nv). Using spline derivative for velocity.")
            qvel_fine = cs(times_fine, 1) # First derivative
        else:
            # Complex case: Quaternions involved (nq > nv)
            # Use MuJoCo's finite difference to get correct Rotational Velocity
            print(f" Dimensions differ (nq={nq}, nv={nv}). Using mj_differentiatePos.")
            qvel_fine = np.zeros((n_fine, nv))
            
            # Use a tiny time step for finite difference derivative
            dt_diff = 1e-6
            times_lookahead = times_fine + dt_diff
            qpos_lookahead = cs(times_lookahead)
            
            # Reuse a buffer for velocity
            v_tmp = np.zeros(nv)
            
            for i in range(n_fine):
                q1 = qpos_fine[i]
                q2 = qpos_lookahead[i]
                
                # qvel = (q2 - q1) / dt_diff (handling quaternions correctly)
                mujoco.mj_differentiatePos(self.model, v_tmp, dt_diff, q1, q2)
                
                qvel_fine[i] = v_tmp.copy()
        
        energies = []
        energies_pot = []
        energies_kin = []
        pelvis_heights = []
        
        print("Running high-frequency energy analysis...")
        
        for i in range(n_fine):
            q = qpos_fine[i]
            v = qvel_fine[i]
            
            self.data.qpos[:] = q
            self.data.qvel[:] = v
            
            mujoco.mj_forward(self.model, self.data)
            
            # Record Pelvis Height
            pelvis_heights.append(self.data.xpos[self.pelvis_id][2])
            
            # Calculate Energy
            mujoco.mj_energyPos(self.model, self.data)
            mujoco.mj_energyVel(self.model, self.data)
            
            pot = self.data.energy[0]
            kin = self.data.energy[1]
            total = pot + kin
            
            energies.append(total)
            energies_pot.append(pot)
            energies_kin.append(kin)
            
        energies = np.array(energies)
        energies_pot = np.array(energies_pot)
        energies_kin = np.array(energies_kin)
        pelvis_heights = np.array(pelvis_heights)
        
        # Calculate Power Loss (dE/dt)
        # We look for where energy is DECREASING (dissipation)
        energy_diff = np.diff(energies, prepend=energies[0])
        power_flow = energy_diff / sim_dt
        
        # Only count negative changes as "Loss" (make positive for plotting)
        power_loss = np.maximum(0, -power_flow)
        
        return {
            "times": times_fine,
            "energy": energies,
            "energy_pot": energies_pot,
            "energy_kin": energies_kin,
            "power_loss": power_loss,
            "height": pelvis_heights,
            "sim_dt": sim_dt
        }

    def plot_results(self, res, output_path):
        times = res['times']
        
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        
        # 1. Energy Stack
        axs[0].plot(times, res['energy'], label='Total Energy', color='black', linewidth=1.5)
        axs[0].stackplot(times, res['energy_pot'], res['energy_kin'], labels=['Potential', 'Kinetic'], alpha=0.5, colors=['#1f77b4', '#ff7f0e'])
        axs[0].set_ylabel('Energy (J)')
        axs[0].set_title('High-Frequency Energy Analysis')
        axs[0].legend(loc='upper right')
        axs[0].grid(True, alpha=0.3)
        
        # 2. Power Loss
        axs[1].plot(times, res['power_loss'], color='red', label='Power Loss (Dissipation)', linewidth=1)
        axs[1].fill_between(times, 0, res['power_loss'], color='red', alpha=0.1)
        axs[1].set_ylabel('Power Loss (W)')
        axs[1].legend(loc='upper right')
        axs[1].grid(True, alpha=0.3)
        
        # 3. Height
        axs[2].plot(times, res['height'], color='green', label='Pelvis Height')
        axs[2].set_ylabel('Height (m)')
        axs[2].set_xlabel('Time (s)')
        axs[2].legend(loc='upper right')
        axs[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")

def plot_comparison_high_freq(orig_res, opt_res, output_path, title_suffix=""):
    """
    Plots comparison between Original (Red) and Optimized (Green) 
    using high-frequency analysis results.
    """
    if orig_res is None or opt_res is None:
        return

    # Assuming both are resampled to the same sim_dt time grid
    times = orig_res['times']
    
    # Check if times match, if not trim to shorter
    min_len = min(len(orig_res['times']), len(opt_res['times']))
    times = times[:min_len]
    
    # Helper to crop arrays
    def crop(arr): return arr[:min_len]

    # Use GridSpec for better layout: Side-by-side composition analysis
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.25, wspace=0.05)
    
    # Calculate baseline potential energy (minimum of both trajectories) for adjustment
    min_pot = min(np.min(orig_res['energy_pot']), np.min(opt_res['energy_pot']))

    # 1. Total Energy (Top, spanning both columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(times, crop(orig_res['energy']), color='red', alpha=0.6, label='Original', linewidth=1.5)
    ax1.plot(times, crop(opt_res['energy']), color='green', alpha=0.9, label='Optimized', linewidth=2.0)
    ax1.set_ylabel('Total Energy (J)')
    ax1.set_title(f'High-Freq Energy Comparison {title_suffix}', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Energy Composition (Original) - Left Column
    ax2 = fig.add_subplot(gs[1, 0])
    e_pot_orig = crop(orig_res['energy_pot']) - min_pot
    e_kin_orig = crop(orig_res['energy_kin'])
    ax2.stackplot(times, e_pot_orig, e_kin_orig, labels=['Potential', 'Kinetic'], alpha=0.6, colors=['#1f77b4', '#ff7f0e'])
    ax2.plot(times, crop(orig_res['energy']) - min_pot, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Energy (J)')
    ax2.set_title('Composition (Original)', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # 3. Energy Composition (Optimized) - Right Column (Sharing Y-axis)
    ax3 = fig.add_subplot(gs[1, 1], sharey=ax2)
    e_pot_opt = crop(opt_res['energy_pot']) - min_pot
    e_kin_opt = crop(opt_res['energy_kin'])
    ax3.stackplot(times, e_pot_opt, e_kin_opt, labels=['Potential', 'Kinetic'], alpha=0.6, colors=['#1f77b4', '#ff7f0e'])
    ax3.plot(times, crop(opt_res['energy']) - min_pot, color='black', linestyle='--', alpha=0.5)
    ax3.set_title('Composition (Optimized)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_yticklabels(), visible=False) # Hide Y labels for cleaner look

    # 4. Accumulated Loss (Bottom, spanning both columns)
    orig_cum_loss = np.cumsum(crop(orig_res['power_loss'])) * orig_res['sim_dt']
    opt_cum_loss = np.cumsum(crop(opt_res['power_loss'])) * opt_res['sim_dt']
    
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(times, orig_cum_loss, color='red', linestyle='--', label='Cum. Loss (Original)')
    ax4.plot(times, opt_cum_loss, color='green', linewidth=2, label='Cum. Loss (Optimized)')
    ax4.set_ylabel('Total Dissipated Energy (J)')
    ax4.set_xlabel('Time (s)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Calculate totals
    total_loss_orig = orig_cum_loss[-1]
    total_loss_opt = opt_cum_loss[-1]
    
    info_text = (
        f"Energy Analysis:\n"
        f"----------------\n"
        f"[Original]  Total Loss: {total_loss_orig:.2f} J\n"
        f"[Optimized] Total Loss: {total_loss_opt:.2f} J\n"
        f"----------------\n"
        f"Reduction: {(total_loss_orig - total_loss_opt):.2f} J\n"
        f"Ratio: {total_loss_opt / (total_loss_orig + 1e-6):.2f}x"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax4.text(0.02, 0.6, info_text, transform=ax4.transAxes, fontsize=11, verticalalignment='top', bbox=props, fontfamily='monospace')

    plt.savefig(output_path)
    plt.close()

def main():
    # Adjust paths relative to this script
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    root_dir = current_dir.parent
    
    xml_path = root_dir / "assets" / "g1_29dof.xml"
    input_npz = root_dir / "data" / "Kjump" / "kjump-1to-1_straight_leg.npz"
    
    if not input_npz.exists():
        print(f"Input file not found: {input_npz}")
        # Try finding any npz in batch_output as a fallback for testing
        fallback_dir = root_dir / "batch_output_Kjump_straight_leg"
        if fallback_dir.exists():
            files = list(fallback_dir.glob("*.npz"))
            if files:
                input_npz = files[0]
                print(f"Using fallback file: {input_npz}")
            else:
                return
        else:
            return

    output_img = current_dir / "high_freq_energy_analysis.png"
    
    analyzer = ComplianceAnalyzerHighFreq(str(xml_path))
    
    # Run analysis with 2ms dt (500Hz)
    res = analyzer.process_trajectory(str(input_npz), orig_dt=1/30, sim_dt=0.002)
    
    if res is not None:
        analyzer.plot_results(res, str(output_img))
        
        # Calculate stats
        total_loss = np.sum(res['power_loss']) * res['sim_dt']
        print(f"Total Energy Dissipated: {total_loss:.2f} J")
        print(f"Analysis saved to {output_img}")

if __name__ == "__main__":
    main()
