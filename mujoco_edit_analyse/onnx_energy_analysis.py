import sys
import os

# Ensure the parent directory is in sys.path to allow importing 'energy_analyse'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from energy_analyse.analysis import run_onnx_energy_analysis

if __name__ == "__main__":
    # Settings
    # Use the scene XML that includes the ground plane
    xml_file = r"c:\Users\not a fat cat\Desktop\Motion_rebuild\assets\scene_29dof.xml"
    onnx_file = r"c:\Users\not a fat cat\Desktop\Motion_rebuild\onnx\2026-01-30_14-23-03_v1.onnx"
    # Choose a motion file
    npz_data = r"C:\Users\not a fat cat\Desktop\Motion_rebuild\data\cropped_data\LAFAN\fallAndGetUp1_subject1_crop_590_790.npz"
    
    # onnx_file = r"c:\Users\not a fat cat\Desktop\Motion_rebuild\onnx\2026-01-30_15-12-39_v1.onnx"
    # # Choose a motion file
    # npz_data = r"C:\Users\not a fat cat\Desktop\Motion_rebuild\data\Kjump\kjump-1to-1.npz"

    # onnx_file = r"c:\Users\not a fat cat\Desktop\Motion_rebuild\onnx\2026-01-30_15-42-08_v1.onnx"
    # # Choose a motion file
    # npz_data = r"C:\Users\not a fat cat\Desktop\Motion_rebuild\data\cropped_data\LAFAN\run1_subject2_crop_4300_4650.npz"
    # Define Explicit Result Directory
    result_dir = r"c:\Users\not a fat cat\Desktop\Motion_rebuild\result"
    
    run_onnx_energy_analysis(xml_file, onnx_file, npz_data, output_dir=result_dir, record_video=True, npz_is_onnx_order=False)
