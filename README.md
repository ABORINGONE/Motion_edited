# G1 机器人运动重构与分析 (G1 Robot Motion Rebuild & Analysis)

本项目为 **G1 人形机器人** 提供了一个基于 MuJoCo 的综合仿真、运动生成和分析框架。项目主要致力于 Sim2Sim 验证、高频控制分析、柔顺落地（Compliant Landing）验证以及训练好的 ONNX 控制策略的能效评估。

## 📂 项目结构

### 核心模块

*   **`energy_analyse/`**
    *   用于评估控制策略性能和能耗的工具。
    *   `analysis.py`: 运行 ONNX 策略仿真并生成能量指标的主程序入口。
    *   `g1_physics.py`: 定义 G1 机器人特有的物理参数和执行器限制。
    *   `observation.py`: 处理策略所需的观测空间（Observation Space）计算。

*   **`mujoco_edit_analyse/`**
    *   用于程序化动作生成和批量仿真的脚本。
    *   `landing_generator.py`: 生成参数化的落地动作（缓冲、恢复平衡）。
    *   `high_freq_analysis.py`: 分析高控制频率下的柔顺性和稳定性。
    *   `batch_run_high_freq.py`: 自动化跨参数（如不同跌落高度）的仿真运行。
    *   `onnx_energy_analysis.py`: 不同模型版本之间的对比分析。

*   **`assets/`**
    *   G1 机器人的 MuJoCo MJCF (`.xml`) 模型文件（23DoF, 29DoF 版本）及场景定义。

### 数据与输出

*   **`onnx/`**: 存储导出为 ONNX 格式的训练好的控制策略模型。
*   **`batch_output_high_freq/`**: 存储仿真结果，包括：
    *   `.npz`: 记录的轨迹数据。
    *   `onnx_comparisons/`: 策略性能与基线对比的图表。
*   **`data/`**: 数据处理工具（例如 `convert_to_npz.py`, `npz_cropper.py`）。

## ✨ 主要特性

1.  **Sim2Sim 验证**:
    *   在高保真物理环境（Sim）中验证训练好的策略（Sim），以确保在实机部署前的行为一致性。
    
2.  **柔顺落地生成 (Compliant Landing Generation)**:
    *   程序化生成落地轨迹，测试机器人吸收冲击（缓冲）和恢复平衡的能力。

3.  **高频分析**:
    *   评估高仿真频率下的控制回路性能和稳定性。

4.  **能量分析 (Energy Profiling)**:
    *   计算传输代价 (CoT)、机械功和关节扭矩分布，以评估运动效率。

## 🚀 使用方法

### 1. 环境设置
确保已安装所需的 Python 依赖库：
```bash
pip install mujoco numpy matplotlib onnxruntime scipy opencv-python mk-mink
```
*（注意：如果是自定义依赖，请参考 `mink/` 文件夹或 PyPI 获取具体的 `mink` 库安装方法）*

### 2. 运行仿真
运行高频落地场景的批量分析：
```bash
python mujoco_edit_analyse/batch_run_high_freq.py
```

### 3. 能量分析
针对记录的动作分析特定的 ONNX 策略：
```bash
python -m energy_analyse.analysis
```

## 📝 许可证
[License Information Here]
