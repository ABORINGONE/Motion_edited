#!/usr/bin/env python3
"""
MuJoCo机器人动画播放器 - 兼容MuJoCo 3.3.6
修改版V2：在3D场景中直接显示浮动帧数标签，解决UI不显示的问题
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
import os
import sys

class MuJoCoAnimationPlayer:
    def __init__(self, xml_path: str, npz_path: str, add_ground: bool = True, save_video_path: str = None):
        self.temp_xml_file = None
        self.xml_path = xml_path
        self.npz_path = npz_path
        self.save_video_path = save_video_path
        
        if add_ground:
            xml_path = self._add_ground_to_xml(xml_path)
        
        print(f"加载MuJoCo模型: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        print(f"加载NPZ数据: {npz_path}")
        npz_data = np.load(npz_path, allow_pickle=True)
        self.qpos_data = npz_data['qpos']
        
        print(f"帧数: {len(self.qpos_data)}")

    def render(self):
        if self.save_video_path:
            self.save_video(self.save_video_path)
        else:
            print("Warning: render() called but no save_video_path provided.")
    
    def __del__(self):
        if self.temp_xml_file and os.path.exists(self.temp_xml_file):
            try:
                os.remove(self.temp_xml_file)
            except:
                pass
    
    def _add_ground_to_xml(self, xml_path: str) -> str:
        """(保持原样) 添加地面平面到XML文件"""
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            worldbody = root.find('.//worldbody')
            if worldbody is not None:
                has_ground = any(g.get('name') == 'ground' or g.get('type') == 'plane' for g in worldbody.findall('.//geom'))
                if not has_ground:
                    asset = root.find('.//asset')
                    if asset is None: asset = ET.SubElement(root, 'asset')
                    ET.SubElement(asset, 'texture', {'name': 'gp', 'type': '2d', 'builtin': 'checker', 'rgb1': '0.2 0.3 0.4', 'rgb2': '0.1 0.2 0.3', 'width': '512', 'height': '512'})
                    ET.SubElement(asset, 'material', {'name': 'gp', 'texture': 'gp', 'texrepeat': '5 5', 'texuniform': 'true', 'reflectance': '0.2'})
                    ground = ET.Element('geom', {'name': 'ground', 'type': 'plane', 'size': '20 20 0.1', 'material': 'gp', 'condim': '3'})
                    worldbody.insert(0, ground)
                    visual = root.find('.//visual')
                    if visual is None: visual = ET.SubElement(root, 'visual')
                    if visual.find('.//headlight') is None: ET.SubElement(visual, 'headlight', {'ambient': '0.4 0.4 0.4', 'diffuse': '0.8 0.8 0.8'})
                    
                    import os
                    xml_dir = os.path.dirname(os.path.abspath(xml_path))
                    temp_xml_path = os.path.join(xml_dir, os.path.basename(xml_path).replace('.xml', '_with_ground.xml'))
                    tree.write(temp_xml_path, encoding='unicode', xml_declaration=True)
                    self.temp_xml_file = temp_xml_path
                    return temp_xml_path
        except Exception:
            pass
        return xml_path
    
    def play_animation(self, fps: int = 30, loop: bool = True, show_ground: bool = True):
        """播放动画，带3D浮动文字标签"""
        frame_dt = 1.0 / fps
        total_frames = len(self.qpos_data)
        
        print(f"开始播放动画 (FPS: {fps})")
        print("提示: 红色文字 [Frame: XXX] 会直接显示在机器人上方")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 基础设置
            if hasattr(viewer, 'opt'):
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            if hasattr(viewer, 'scn'):
                viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
            
            frame_idx = 0
            last_time = time.time()
            
            while viewer.is_running():
                # 处理暂停 (如果GUI支持)
                if hasattr(viewer, 'is_paused') and viewer.is_paused():
                    viewer.sync()
                    time.sleep(0.1)
                    continue

                current_time = time.time()
                
                if (current_time - last_time) >= frame_dt:
                    # 1. 设置关节位置
                    qpos = self.qpos_data[frame_idx]
                    if len(qpos) <= self.model.nq:
                        self.data.qpos[:len(qpos)] = qpos
                    else:
                        self.data.qpos[:] = qpos[:self.model.nq]
                    
                    # 2. 前向运动学
                    mujoco.mj_forward(self.model, self.data)
                    
                    # 3. 终端打印
                    sys.stdout.write(f"\r当前帧: {frame_idx:04d} / {total_frames} ({(frame_idx/total_frames)*100:.1f}%)")
                    sys.stdout.flush()
                    
                    # 4. 更新下一帧
                    frame_idx = (frame_idx + 1) % len(self.qpos_data)
                    if frame_idx == 0 and not loop:
                        break
                    last_time = current_time

                # === 核心修改：添加3D浮动文字 ===
                if hasattr(viewer, 'user_scn'):
                    # 计算显示位置：取整个物体的重心上方 1.5 米处
                    # 如果没有计算重心，默认显示在 (0, 0, 2.0)
                    text_pos = np.array([0.0, 0.0, 2.0])
                    if hasattr(self.data, 'subtree_com'):
                         # 使用第一个body（通常是world或base）的子树重心
                        text_pos = self.data.subtree_com[0] + np.array([0, 0, 1.2])

                    # 格式化显示内容
                    label_text = f"Frame: {frame_idx}"
                    
                    # 初始化一个 Label Geom
                    viewer.user_scn.ngeom = 1
                    geom = viewer.user_scn.geoms[0]
                    # 设置类型为 LABEL (文字)
                    geom.type = mujoco.mjtGeom.mjGEOM_LABEL
                    geom.pos = text_pos
                    geom.label = label_text
                    # 设置颜色 (RGBA): 红色，不透明
                    geom.rgba = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
                    # 文字大小
                    geom.size = np.array([0.5, 0, 0], dtype=np.float32) 
                # ===============================

                viewer.sync()
                time.sleep(0.001)

    def save_video(self, output_path: str, fps: int = 30, width: int = 640, height: int = 480, 
                   camera_mode: str = "free", camera_distance: float = 3.0, 
                   camera_azimuth: float = 45.0, camera_elevation: float = -30.0,
                   camera_target: tuple = (0, 0, 1), show_ground: bool = True):
        """保存视频 (逻辑保持不变)"""
        try:
            import cv2
        except ImportError:
            print("错误: 需要 pip install opencv-python")
            return
        
        print(f"开始录制: {output_path}")
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        
        if show_ground and hasattr(renderer, 'scene'):
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        for frame_idx in range(len(self.qpos_data)):
            qpos = self.qpos_data[frame_idx]
            if len(qpos) <= self.model.nq: self.data.qpos[:len(qpos)] = qpos
            else: self.data.qpos[:] = qpos[:self.model.nq]
            
            mujoco.mj_kinematics(self.model, self.data)
            
            # 这里也添加文字到视频里
            if hasattr(renderer, 'scene'):
                renderer.scene.ngeom = 1
                g = renderer.scene.geoms[0]
                g.type = mujoco.mjtGeom.mjGEOM_LABEL
                # 位置
                if hasattr(self.data, 'subtree_com'):
                    g.pos = self.data.subtree_com[0] + np.array([0, 0, 1.2])
                else:
                    g.pos = np.array([0, 0, 2.0])
                g.label = f"Frm:{frame_idx}"
                g.rgba = np.array([1, 0, 0, 1], dtype=np.float32)
                g.size = np.array([0.5, 0, 0], dtype=np.float32)

            renderer.update_scene(self.data)
            
            # 简单的摄像机逻辑
            if hasattr(renderer, 'camera'):
                if camera_mode == "follow":
                    renderer.camera.lookat[:] = self.data.qpos[:3]
                elif camera_mode == "free":
                    renderer.camera.lookat[:] = camera_target
                renderer.camera.distance = camera_distance
                renderer.camera.azimuth = camera_azimuth
                renderer.camera.elevation = camera_elevation
                
            out.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            if frame_idx % 20 == 0:
                sys.stdout.write(f"\r录制进度: {frame_idx}/{len(self.qpos_data)}")
                sys.stdout.flush()
        
        out.release()
        renderer.close()
        print("\n视频保存完成")

    def analyze_motion(self):
        """简单分析"""
        print(f"数据总帧数: {len(self.qpos_data)}")
        print(f"自由度: {self.model.nq}")

    def save_high_res_video_info(self):
        print("高分辨率录制请修改XML: <visual><global offwidth='1920' offheight='1080'/></visual>")

    def show_ground_plane_info(self):
        print("请参考代码中的XML修改逻辑")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', required=True)
    parser.add_argument('--npz', required=True)
    parser.add_argument('--fps', type=int, default=30)
    parser.add_argument('--no-loop', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    parser.add_argument('--save-video', type=str)
    parser.add_argument('--video-width', type=int, default=640)
    parser.add_argument('--video-height', type=int, default=480)
    parser.add_argument('--no-ground', action='store_true')
    # 简化摄像机参数
    parser.add_argument('--camera-mode', default='free')
    parser.add_argument('--camera-distance', type=float, default=3.0)
    parser.add_argument('--camera-azimuth', type=float, default=45.0)
    parser.add_argument('--camera-elevation', type=float, default=-30.0)
    parser.add_argument('--camera-target-x', type=float, default=0.0)
    parser.add_argument('--camera-target-y', type=float, default=0.0)
    parser.add_argument('--camera-target-z', type=float, default=1.0)
    parser.add_argument('--high-res-info', action='store_true')
    parser.add_argument('--ground-plane-info', action='store_true')
    
    args = parser.parse_args()
    if not os.path.exists(args.xml) or not os.path.exists(args.npz):
        print("错误: 文件不存在")
        return

    player = MuJoCoAnimationPlayer(args.xml, args.npz, add_ground=not args.no_ground)
    
    if args.save_video:
        player.save_video(args.save_video, args.fps, args.video_width, args.video_height,
            args.camera_mode, args.camera_distance, args.camera_azimuth, args.camera_elevation, 
            (args.camera_target_x, args.camera_target_y, args.camera_target_z), not args.no_ground)
    elif args.analyze:
        player.analyze_motion()
    else:
        player.play_animation(args.fps, not args.no_loop, not args.no_ground)

if __name__ == "__main__":
    main()

