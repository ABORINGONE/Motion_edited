#!/usr/bin/env python3
"""
MuJoCo机器人动画播放器 - 兼容MuJoCo 3.3.6
直接使用MuJoCo物理引擎播放qpos数据，带地面显示
"""

import numpy as np
import mujoco
import mujoco.viewer
import time
import argparse
import os

class MuJoCoAnimationPlayer:
    def __init__(self, xml_path: str, npz_path: str, add_ground: bool = True):
        self.xml_path = xml_path
        self.npz_path = npz_path
        self.temp_xml_file = None  # 用于跟踪临时文件
        
        # 如果需要添加地面，修改XML
        if add_ground:
            xml_path = self._add_ground_to_xml(xml_path)
        
        # 加载MuJoCo模型
        print(f"加载MuJoCo模型: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 加载动画数据
        print(f"加载NPZ数据: {npz_path}")
        npz_data = np.load(npz_path, allow_pickle=True)
        self.qpos_data = npz_data['qpos']
        
        print(f"模型自由度: {self.model.nq}")
        print(f"qpos数据形状: {self.qpos_data.shape}")
        print(f"动画帧数: {len(self.qpos_data)}")
        
        # 验证数据维度
        if self.qpos_data.shape[1] != self.model.nq:
            print(f"警告: qpos维度不匹配 - 数据:{self.qpos_data.shape[1]}, 模型:{self.model.nq}")
    
    def __del__(self):
        """清理临时文件"""
        if self.temp_xml_file and os.path.exists(self.temp_xml_file):
            try:
                os.remove(self.temp_xml_file)
                print(f"已清理临时文件: {self.temp_xml_file}")
            except:
                pass
    
    def _add_ground_to_xml(self, xml_path: str) -> str:
        """添加地面平面到XML文件"""
        import xml.etree.ElementTree as ET
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 检查是否已经有地面
            worldbody = root.find('.//worldbody')
            if worldbody is not None:
                # 检查是否已有名为ground的geom
                has_ground = any(
                    geom.get('name') == 'ground' or geom.get('type') == 'plane'
                    for geom in worldbody.findall('.//geom')
                )
                
                if not has_ground:
                    print("XML中未检测到地面，正在添加地面平面...")
                    
                    # 添加地面纹理和材质到asset
                    asset = root.find('.//asset')
                    if asset is None:
                        asset = ET.SubElement(root, 'asset')
                    
                    # 添加棋盘格纹理
                    texture = ET.SubElement(asset, 'texture', {
                        'name': 'groundplane',
                        'type': '2d',
                        'builtin': 'checker',
                        'rgb1': '0.2 0.3 0.4',
                        'rgb2': '0.1 0.2 0.3',
                        'width': '512',
                        'height': '512'
                    })
                    
                    # 添加材质
                    material = ET.SubElement(asset, 'material', {
                        'name': 'groundplane',
                        'texture': 'groundplane',
                        'texrepeat': '5 5',
                        'texuniform': 'true',
                        'reflectance': '0.2'
                    })
                    
                    # 在worldbody最前面添加地面
                    ground = ET.Element('geom', {
                        'name': 'ground',
                        'type': 'plane',
                        'size': '20 20 0.1',
                        'material': 'groundplane',
                        'condim': '3',
                        'rgba': '0.5 0.5 0.5 1'
                    })
                    worldbody.insert(0, ground)
                    
                    # 确保有光照
                    visual = root.find('.//visual')
                    if visual is None:
                        visual = ET.SubElement(root, 'visual')
                    
                    headlight = visual.find('.//headlight')
                    if headlight is None:
                        headlight = ET.SubElement(visual, 'headlight', {
                            'ambient': '0.4 0.4 0.4',
                            'diffuse': '0.8 0.8 0.8',
                            'specular': '0.1 0.1 0.1'
                        })
                    
                    # 保存到原XML同目录下的临时文件，保持相对路径有效
                    import os
                    xml_dir = os.path.dirname(os.path.abspath(xml_path))
                    xml_basename = os.path.basename(xml_path)
                    temp_xml_name = xml_basename.replace('.xml', '_with_ground.xml')
                    temp_xml_path = os.path.join(xml_dir, temp_xml_name)
                    
                    tree.write(temp_xml_path, encoding='unicode', xml_declaration=True)
                    print(f"已添加地面平面到临时文件: {temp_xml_path}")
                    self.temp_xml_file = temp_xml_path  # 保存路径以便后续清理
                    return temp_xml_path
                else:
                    print("XML中已有地面平面")
            
        except Exception as e:
            print(f"添加地面时出错: {e}")
        
        self.temp_xml_file = None
        return xml_path
    
    def play_animation(self, fps: int = 30, loop: bool = True, show_ground: bool = True):
        """播放动画"""
        frame_dt = 1.0 / fps
        
        print(f"开始播放动画 (FPS: {fps})")
        print("控制:")
        print("  点击关闭窗口退出")
        print("  鼠标拖拽旋转视角")
        print("  滚轮缩放")
        if show_ground:
            print("  显示地面网格: 开启")


        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # 设置可视化选项
            if hasattr(viewer, 'opt'):
                # 启用阴影和反射
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
                
            # 设置场景渲染选项
            if hasattr(viewer, 'scn'):
                viewer.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
                viewer.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
            
            # 设置更亮的背景
            if hasattr(viewer, 'user_scn') and viewer.user_scn is not None:
                # 设置天空盒颜色为浅蓝色
                viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
            
            frame_idx = 0
            last_time = time.time()
            
            while viewer.is_running():
                current_time = time.time()
                
                if (current_time - last_time) >= frame_dt:
                    # 设置关节位置
                    qpos = self.qpos_data[frame_idx]
                    
                    # 确保数据维度匹配
                    if len(qpos) <= self.model.nq:
                        self.data.qpos[:len(qpos)] = qpos
                    else:
                        self.data.qpos[:] = qpos[:self.model.nq]
                    
                    # 前向运动学计算
                    mujoco.mj_forward(self.model, self.data)
                    
                    # 更新帧索引
                    frame_idx = (frame_idx + 1) % len(self.qpos_data)
                    if frame_idx == 0 and not loop:
                        print("动画播放完成")
                        break
                    
                    last_time = current_time
                
                # 同步viewer
                viewer.sync()
                time.sleep(0.001)
    
    def save_video(self, output_path: str, fps: int = 30, width: int = 640, height: int = 480, 
                   camera_mode: str = "free", camera_distance: float = 3.0, 
                   camera_azimuth: float = 45.0, camera_elevation: float = -30.0,
                   camera_target: tuple = (0, 0, 1), show_ground: bool = True):
        """保存视频文件"""
        try:
            import cv2
        except ImportError:
            print("错误: 需要安装opencv-python来保存视频")
            print("pip install opencv-python")
            return
        
        print(f"开始录制视频: {output_path}")
        print(f"分辨率: {width}x{height}")
        print(f"摄像机模式: {camera_mode}")
        if show_ground:
            print(f"地面显示: 开启")
        
        # 创建离屏渲染器
        try:
            renderer = mujoco.Renderer(self.model, height=height, width=width)
        except ValueError as e:
            if "framebuffer width" in str(e):
                print("警告: 请求的分辨率过高，使用默认640x480")
                width, height = 640, 480
                renderer = mujoco.Renderer(self.model, height=height, width=width)
            else:
                raise e
        
        # 设置渲染选项 - 启用地面网格和阴影
        if show_ground:
            if hasattr(renderer, 'scene'):
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = True
        
        # 初始化视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print("错误: 无法创建视频文件")
            return
        
        try:
            for frame_idx in range(len(self.qpos_data)):
                # 设置关节位置
                qpos = self.qpos_data[frame_idx]
                if len(qpos) <= self.model.nq:
                    self.data.qpos[:len(qpos)] = qpos
                else:
                    self.data.qpos[:] = qpos[:self.model.nq]
                
                # 前向运动学
                mujoco.mj_kinematics(self.model, self.data)
                
                # 设置摄像机参数（通过修改renderer的内部摄像机）
                if hasattr(renderer, 'camera'):
                    camera = renderer.camera
                else:
                    # 如果没有直接的camera属性，创建一个临时的
                    camera = mujoco.MjvCamera()
                    mujoco.mjv_defaultCamera(camera)
                
                # 根据模式设置摄像机参数
                if camera_mode == "follow":
                    # 跟随机器人基座
                    base_pos = self.data.qpos[:3].copy()
                    if hasattr(camera, 'lookat'):
                        camera.lookat[:] = base_pos
                    camera.distance = camera_distance
                    camera.azimuth = camera_azimuth
                    camera.elevation = camera_elevation
                elif camera_mode == "side":
                    # 侧视图
                    if frame_idx == 0:  # 只在第一帧设置目标点
                        base_pos = self.data.qpos[:3].copy()
                        if hasattr(camera, 'lookat'):
                            camera.lookat[:] = base_pos
                    camera.distance = camera_distance
                    camera.azimuth = 90
                    camera.elevation = 0
                elif camera_mode == "front":
                    # 正面视图
                    if frame_idx == 0:
                        base_pos = self.data.qpos[:3].copy()
                        if hasattr(camera, 'lookat'):
                            camera.lookat[:] = base_pos
                    camera.distance = camera_distance
                    camera.azimuth = 0
                    camera.elevation = 0
                elif camera_mode == "top":
                    # 顶视图
                    base_pos = self.data.qpos[:3].copy()
                    if hasattr(camera, 'lookat'):
                        camera.lookat[:] = base_pos
                    camera.distance = camera_distance
                    camera.azimuth = 0
                    camera.elevation = -89
                else:  # free mode
                    # 自由摄像机
                    if hasattr(camera, 'lookat'):
                        camera.lookat[:] = camera_target
                    camera.distance = camera_distance
                    camera.azimuth = camera_azimuth
                    camera.elevation = camera_elevation
                
                # 渲染 - 使用简化的API调用
                try:
                    # 尝试不同的API调用方式
                    if hasattr(renderer, 'camera') and renderer.camera is not None:
                        renderer.update_scene(self.data, camera=camera)
                    else:
                        renderer.update_scene(self.data)
                except TypeError:
                    # 如果上面的调用失败，使用更简单的方式
                    renderer.update_scene(self.data)
                
                pixels = renderer.render()
                
                # 转换颜色格式
                pixels_bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                out.write(pixels_bgr)
                
                # 进度显示
                if frame_idx % 10 == 0:
                    progress = (frame_idx / len(self.qpos_data)) * 100
                    print(f"进度: {frame_idx}/{len(self.qpos_data)} ({progress:.1f}%)")
        
        except Exception as e:
            print(f"录制过程中出错: {e}")
        finally:
            out.release()
            renderer.close()
            print(f"视频保存完成: {output_path}")
    
    def analyze_motion(self):
        """分析运动数据"""
        print("\n=== 运动数据分析 ===")
        
        # 关节名称
        joint_names = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_names.append(name if name else f"joint_{i}")
        
        print(f"关节列表 ({len(joint_names)}):")
        for i, name in enumerate(joint_names):
            if i < self.qpos_data.shape[1]:
                qpos_range = [
                    np.min(self.qpos_data[:, i]),
                    np.max(self.qpos_data[:, i])
                ]
                print(f"  {i}: {name} - 范围: [{qpos_range[0]:.3f}, {qpos_range[1]:.3f}]")
        
        # 基座运动分析（前7维：3位置+4四元数）
        if self.qpos_data.shape[1] >= 7:
            base_pos = self.qpos_data[:, :3]
            base_quat = self.qpos_data[:, 3:7]
            
            print(f"\n基座运动:")
            print(f"  X范围: [{np.min(base_pos[:, 0]):.3f}, {np.max(base_pos[:, 0]):.3f}]")
            print(f"  Y范围: [{np.min(base_pos[:, 1]):.3f}, {np.max(base_pos[:, 1]):.3f}]")
            print(f"  Z范围: [{np.min(base_pos[:, 2]):.3f}, {np.max(base_pos[:, 2]):.3f}]")
            
            print(f"\n运动统计:")
            print(f"  总时长: {len(self.qpos_data) / 30:.2f} 秒 (假设30fps)")
            print(f"  最大基座高度: {np.max(base_pos[:, 2]):.3f} 米")
            print(f"  最小基座高度: {np.min(base_pos[:, 2]):.3f} 米")
            print(f"  水平移动距离: {np.sqrt(np.sum((base_pos[-1,:2] - base_pos[0,:2])**2)):.3f} 米")
    
    def save_high_res_video_info(self):
        """显示高分辨率录制信息"""
        print("=== 高分辨率视频录制说明 ===")
        print("要录制高分辨率视频，需要在XML文件中添加配置:")
        print()
        print("在 <mujoco> 标签内添加:")
        print("<visual>")
        print("  <global offwidth='1920' offheight='1080'/>")
        print("</visual>")
        print()
        print("然后可以使用:")
        print("python mujoco_player.py --xml your_model.xml --npz your_data.npz \\")
        print("  --save-video video.mp4 --video-width 1920 --video-height 1080")
    
    def show_ground_plane_info(self):
        """显示如何在XML中添加地面平面的信息"""
        print("\n=== 添加地面平面到XML模型 ===")
        print("如果你想永久添加地面到XML文件，可以手动添加以下内容:")
        print()
        print("方法1 - 简单灰色地面:")
        print("<worldbody>")
        print("  <geom name='ground' type='plane' size='20 20 0.1' rgba='0.5 0.5 0.5 1'/>")
        print("</worldbody>")
        print()
        print("方法2 - 带棋盘格纹理的地面 (推荐):")
        print()
        print("<!-- 在<asset>标签中添加 -->")
        print("<asset>")
        print("  <texture name='grid' type='2d' builtin='checker' ")
        print("           width='512' height='512' rgb1='0.2 0.3 0.4' rgb2='0.1 0.2 0.3'/>")
        print("  <material name='grid' texture='grid' texrepeat='5 5' texuniform='true'/>")
        print("</asset>")
        print()
        print("<!-- 在<worldbody>标签中添加 -->")
        print("<worldbody>")
        print("  <geom name='ground' type='plane' size='20 20 0.1' material='grid'/>")
        print("</worldbody>")
        print()
        print("注意: 如果不手动添加地面，程序会自动创建临时XML文件添加地面。")


def main():
    parser = argparse.ArgumentParser(description='MuJoCo机器人动画播放器 - 兼容MuJoCo 3.3.6 (带地面显示)')
    parser.add_argument('--xml', type=str, required=True, help='MuJoCo XML文件路径')
    parser.add_argument('--npz', type=str, required=True, help='NPZ数据文件路径')
    parser.add_argument('--fps', type=int, default=30, help='播放帧率')
    parser.add_argument('--no-loop', action='store_true', help='不循环播放')
    parser.add_argument('--analyze', action='store_true', help='分析运动数据')
    parser.add_argument('--save-video', type=str, help='保存视频文件路径')
    parser.add_argument('--video-width', type=int, default=640, help='视频宽度')
    parser.add_argument('--video-height', type=int, default=480, help='视频高度')
    parser.add_argument('--no-ground', action='store_true', help='不显示地面网格')
    
    # 摄像机控制参数
    camera_group = parser.add_argument_group('摄像机控制')
    camera_group.add_argument('--camera-mode', type=str, default='free', 
                            choices=['free', 'follow', 'side', 'front', 'top'],
                            help='摄像机模式')
    camera_group.add_argument('--camera-distance', type=float, default=3.0, help='摄像机距离')
    camera_group.add_argument('--camera-azimuth', type=float, default=45.0, help='方位角(度)')
    camera_group.add_argument('--camera-elevation', type=float, default=-30.0, help='仰角(度)')
    camera_group.add_argument('--camera-target-x', type=float, default=0.0, help='目标点X')
    camera_group.add_argument('--camera-target-y', type=float, default=0.0, help='目标点Y')
    camera_group.add_argument('--camera-target-z', type=float, default=1.0, help='目标点Z')
    
    parser.add_argument('--high-res-info', action='store_true', help='显示高分辨率录制说明')
    parser.add_argument('--ground-plane-info', action='store_true', help='显示如何添加地面平面的说明')
    
    args = parser.parse_args()
    
    # 检查文件
    for file_path, name in [(args.xml, 'XML'), (args.npz, 'NPZ')]:
        if not os.path.exists(file_path):
            print(f"错误: {name}文件不存在: {file_path}")
            return
        

    
    # 创建播放器
    player = MuJoCoAnimationPlayer(args.xml, args.npz, add_ground=not args.no_ground)
    
    # 显示帮助信息
    if args.high_res_info:
        player.save_high_res_video_info()
        return
    
    if args.ground_plane_info:
        player.show_ground_plane_info()
        return
    
    # 分析数据
    if args.analyze:
        player.analyze_motion()
        if not args.save_video:
            return
    
    # 保存视频
    if args.save_video:
        camera_target = (args.camera_target_x, args.camera_target_y, args.camera_target_z)
        player.save_video(
            args.save_video, args.fps, args.video_width, args.video_height,
            args.camera_mode, args.camera_distance, args.camera_azimuth, 
            args.camera_elevation, camera_target, show_ground=not args.no_ground
        )
    else:
        # 交互式播放
        player.play_animation(args.fps, not args.no_loop, show_ground=not args.no_ground)


if __name__ == "__main__":
    # 检查MuJoCo安装
    try:
        import mujoco
        print(f"MuJoCo版本: {mujoco.__version__}")
    except ImportError:
        print("错误: 未安装MuJoCo")
        print("安装命令: pip install mujoco")
        exit(1)
    
    main()