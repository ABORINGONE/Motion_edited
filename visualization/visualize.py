#!/usr/bin/env python3
"""
NPZ/XML/STL 完整3D机器人动画可视化工具
基于qpos数据和STL网格创建完整的机器人动画
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import xml.etree.ElementTree as ET
import os
import sys
import argparse
from pathlib import Path
import glob
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# 可选依赖导入
try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("警告: Trimesh未安装，STL文件加载功能将被禁用")

try:
    from scipy.spatial.transform import Rotation as R
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: SciPy未安装，旋转功能将使用简化版本")


@dataclass
class Joint:
    """关节定义"""
    name: str
    parent: Optional[str]
    joint_type: str = 'revolute'
    axis: np.ndarray = None
    origin: np.ndarray = None  # [x, y, z, roll, pitch, yaw]
    limits: Tuple[float, float] = (-np.pi, np.pi)
    
@dataclass 
class Link:
    """连杆定义"""
    name: str
    mesh_file: Optional[str] = None
    origin: np.ndarray = None
    
@dataclass
class MeshData:
    """网格数据"""
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None


class XMLParser:
    """MuJoCo XML解析器"""
    
    @staticmethod
    def parse_robot_xml(filepath: str) -> Dict[str, Any]:
        """解析MuJoCo格式的机器人XML文件"""
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            print(f"XML根标签: {root.tag}")
            print(f"模型名称: {root.get('model', 'unknown')}")
            
            robot_data = {
                'name': root.get('model', 'robot'),
                'links': {},
                'joints': {},
                'joint_order': [],
                'parent_map': {},
                'child_map': {},
                'mesh_files': {}
            }
            
            # 首先解析asset部分，获取mesh文件映射
            XMLParser._parse_assets(root, robot_data)
            
            # 解析worldbody，获取body和joint结构
            worldbody = root.find('worldbody')
            if worldbody is not None:
                XMLParser._parse_worldbody(worldbody, robot_data)
            
            # 解析actuator部分，获取驱动关节顺序
            XMLParser._parse_actuators(root, robot_data)
            
            print(f"MuJoCo解析成功: {len(robot_data['links'])} 连杆, {len(robot_data['joints'])} 关节")
            print(f"驱动关节顺序: {len(robot_data['joint_order'])} 个")
            
            return robot_data
            
        except Exception as e:
            print(f"MuJoCo XML解析失败: {e}")
            import traceback
            traceback.print_exc()
            return {'links': {}, 'joints': {}, 'joint_order': [], 'parent_map': {}, 'child_map': {}, 'mesh_files': {}}
    
    @staticmethod
    def _parse_assets(root, robot_data):
        """解析asset部分，获取mesh文件信息"""
        asset_elem = root.find('asset')
        if asset_elem is not None:
            for mesh_elem in asset_elem.findall('mesh'):
                mesh_name = mesh_elem.get('name')
                mesh_file = mesh_elem.get('file')
                if mesh_name and mesh_file:
                    robot_data['mesh_files'][mesh_name] = mesh_file
    
    @staticmethod
    def _parse_worldbody(worldbody, robot_data, parent_name=None):
        """递归解析worldbody中的body结构"""
        for body_elem in worldbody.findall('body'):
            body_name = body_elem.get('name')
            if not body_name:
                continue
            
            # 解析位置和姿态
            pos_str = body_elem.get('pos', '0 0 0')
            quat_str = body_elem.get('quat', '1 0 0 0')
            
            pos = np.array([float(x) for x in pos_str.split()])
            quat = np.array([float(x) for x in quat_str.split()])
            
            # 查找对应的mesh
            mesh_name = None
            for geom in body_elem.findall('geom'):
                mesh_attr = geom.get('mesh')
                if mesh_attr:
                    mesh_name = mesh_attr
                    break
            
            # 创建连杆
            robot_data['links'][body_name] = Link(
                name=body_name,
                mesh_file=robot_data['mesh_files'].get(mesh_name) if mesh_name else None,
                origin=np.concatenate([pos, np.zeros(3)])  # [x, y, z, r, p, y]
            )
            
            # 设置父子关系
            if parent_name:
                robot_data['parent_map'][body_name] = parent_name
                if parent_name not in robot_data['child_map']:
                    robot_data['child_map'][parent_name] = []
                robot_data['child_map'][parent_name].append(body_name)
            
            # 解析关节
            for joint_elem in body_elem.findall('joint'):
                joint_name = joint_elem.get('name')
                if not joint_name:
                    continue
                
                joint_type = joint_elem.get('type', 'revolute')
                if joint_type == 'free':
                    continue  # 跳过浮动基座关节
                
                # 获取关节轴
                axis_str = joint_elem.get('axis', '0 0 1')
                axis = np.array([float(x) for x in axis_str.split()])
                
                # 获取关节限制
                range_str = joint_elem.get('range', '-3.14 3.14')
                range_vals = [float(x) for x in range_str.split()]
                limits = (range_vals[0], range_vals[1])
                
                # 创建关节
                robot_data['joints'][joint_name] = Joint(
                    name=joint_name,
                    parent=parent_name,
                    joint_type='revolute' if joint_type != 'prismatic' else 'prismatic',
                    axis=axis,
                    origin=np.concatenate([pos, np.zeros(3)]),
                    limits=limits
                )
            
            # 递归解析子body
            XMLParser._parse_worldbody(body_elem, robot_data, body_name)
    
    @staticmethod
    def _parse_actuators(root, robot_data):
        """解析actuator部分，确定关节驱动顺序"""
        actuator_elem = root.find('actuator')
        if actuator_elem is not None:
            for motor_elem in actuator_elem.findall('motor'):
                joint_name = motor_elem.get('joint')
                if joint_name and joint_name in robot_data['joints']:
                    robot_data['joint_order'].append(joint_name)
        
        print(f"找到 {len(robot_data['joint_order'])} 个驱动关节")
        for i, joint_name in enumerate(robot_data['joint_order']):
            print(f"  {i}: {joint_name}")
    
    @staticmethod
    def _debug_xml_structure(element, level=0):
        """调试XML结构"""
        indent = "  " * level
        print(f"{indent}{element.tag}: {element.attrib}")
        
        if level < 2:  # 限制深度
            for child in element:
                XMLParser._debug_xml_structure(child, level + 1)


class DataLoader:
    """数据加载器"""
    
    @staticmethod
    def load_npz(filepath: str) -> Dict[str, np.ndarray]:
        """加载NPZ文件"""
        try:
            data = np.load(filepath, allow_pickle=True)
            result = {}
            
            for key in data.files:
                result[key] = data[key]
                print(f"加载数据: {key}, 形状: {data[key].shape}")
            
            return result
        except Exception as e:
            print(f"加载NPZ文件失败: {e}")
            return {}
    
    @staticmethod
    def load_stl_meshes(assets_folder: str) -> Dict[str, MeshData]:
        """加载assets文件夹中的所有STL文件，返回字典便于查找"""
        mesh_dict = {}
        
        if not os.path.exists(assets_folder):
            print(f"Assets文件夹不存在: {assets_folder}")
            return mesh_dict
        
        stl_files = glob.glob(os.path.join(assets_folder, "*.stl"))
        print(f"找到 {len(stl_files)} 个STL文件")
        
        for stl_file in stl_files:
            mesh = DataLoader._load_stl(stl_file)
            if mesh:
                mesh_dict[mesh.name] = mesh
                print(f"加载STL: {mesh.name} - {len(mesh.vertices)} 顶点")
        
        return mesh_dict
    
    @staticmethod
    def _load_stl(filepath: str) -> Optional[MeshData]:
        """加载单个STL文件"""
        try:
            if TRIMESH_AVAILABLE:
                mesh = trimesh.load(filepath)
                return MeshData(
                    name=Path(filepath).stem,
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    normals=mesh.vertex_normals if hasattr(mesh, 'vertex_normals') else None
                )
            else:
                return DataLoader._load_stl_ascii(filepath)
        except Exception as e:
            print(f"加载STL文件失败 {filepath}: {e}")
            return None
    
    @staticmethod
    def _load_stl_ascii(filepath: str) -> Optional[MeshData]:
        """简化的ASCII STL加载器"""
        vertices = []
        faces = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                vertex_count = 0
                current_face = []
                
                for line in f:
                    line = line.strip()
                    if line.startswith('vertex'):
                        coords = [float(x) for x in line.split()[1:]]
                        vertices.append(coords)
                        current_face.append(vertex_count)
                        vertex_count += 1
                        
                        if len(current_face) == 3:
                            faces.append(current_face)
                            current_face = []
            
            return MeshData(
                name=Path(filepath).stem,
                vertices=np.array(vertices),
                faces=np.array(faces)
            )
        except:
            return None


class RobotAnimator:
    """机器人3D动画器"""
    
    def __init__(self, figsize=(20, 15)):
        self.fig = plt.figure(figsize=figsize)
        self.setup_subplots()
        self.frame_count = 0
        self.animation = None
        
        # 数据存储
        self.npz_data = None
        self.robot_data = None
        self.mesh_dict = {}
        
        # 动画相关
        self.link_transforms = {}  # 存储每个连杆的变换矩阵
        
    def setup_subplots(self):
        """设置子图"""
        # 主3D视图 (占据大部分空间)
        self.ax_3d = self.fig.add_subplot(2, 2, (1, 3), projection='3d')
        self.ax_3d.set_title('机器人3D动画', fontsize=18, fontweight='bold')
        
        # 关节角度图
        self.ax_joints = self.fig.add_subplot(2, 2, 2)
        self.ax_joints.set_title('关节角度 (qpos)', fontsize=14)
        
        # 关节时序图
        self.ax_timeline = self.fig.add_subplot(2, 2, 4)
        self.ax_timeline.set_title('关节时序变化', fontsize=14)
        
        plt.tight_layout()
    
    def load_data(self, npz_file: str, xml_file: str = None, assets_folder: str = None):
        """加载所有数据"""
        print("开始加载机器人动画数据...")
        
        # 加载NPZ数据
        if npz_file and os.path.exists(npz_file):
            self.npz_data = DataLoader.load_npz(npz_file)
        else:
            print(f"NPZ文件不存在: {npz_file}")
            return False
        
        # 加载机器人XML配置
        if xml_file and os.path.exists(xml_file):
            self.robot_data = XMLParser.parse_robot_xml(xml_file)
        else:
            print("未提供XML文件或文件不存在")
            return False
        
        # 加载STL网格
        if assets_folder:
            self.mesh_dict = DataLoader.load_stl_meshes(assets_folder)
        else:
            print("未提供assets文件夹")
        
        # 验证数据
        return self._validate_data()
    
    def _validate_data(self):
        """验证数据完整性"""
        if not self.npz_data or 'qpos' not in self.npz_data:
            print("错误: 缺少qpos数据")
            return False
        
        if not self.robot_data:
            print("错误: 机器人配置为空")
            return False
        
        # 检查是否有连杆数据
        if not self.robot_data.get('links'):
            print("错误: 缺少机器人连杆配置")
            return False
        
        qpos_shape = self.npz_data['qpos'].shape
        num_actuated_joints = len(self.robot_data.get('joint_order', []))
        num_links = len(self.robot_data.get('links', {}))
        
        print(f"qpos数据: {qpos_shape}")
        print(f"驱动关节: {num_actuated_joints}")
        print(f"机器人连杆: {num_links}")
        print(f"STL网格: {len(self.mesh_dict)} 个")
        
        # MuJoCo格式：qpos = floating_base(7) + actuated_joints(29) = 36
        expected_qpos_dim = 7 + num_actuated_joints  # 7 for floating base (3 pos + 4 quat)
        
        if qpos_shape[1] == expected_qpos_dim:
            print(f"✓ qpos维度匹配: {qpos_shape[1]} = 7(浮动基座) + {num_actuated_joints}(驱动关节)")
        else:
            print(f"警告: qpos维度({qpos_shape[1]}) != 预期维度({expected_qpos_dim})")
        
        return True
    
    def calculate_forward_kinematics(self, qpos: np.ndarray) -> Dict[str, np.ndarray]:
        """计算MuJoCo格式的正向运动学"""
        transforms = {}
        
        # MuJoCo格式：前7维是浮动基座 (x,y,z, qw,qx,qy,qz)
        if len(qpos) >= 7:
            base_pos = qpos[:3]  # [x, y, z]
            base_quat = qpos[3:7]  # [qw, qx, qy, qz]
            actuated_joint_angles = qpos[7:]  # 驱动关节角度
        else:
            print("警告: qpos维度不足，使用零值")
            base_pos = np.zeros(3)
            base_quat = np.array([1, 0, 0, 0])
            actuated_joint_angles = qpos
        
        # 创建基座变换矩阵
        base_transform = np.eye(4)
        base_transform[:3, 3] = base_pos
        
        # 应用基座旋转（四元数转旋转矩阵）
        if SCIPY_AVAILABLE:
            from scipy.spatial.transform import Rotation as R
            base_rotation = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])  # scipy使用[x,y,z,w]
            base_transform[:3, :3] = base_rotation.as_matrix()
        
        # 根连杆（通常是pelvis）使用基座变换
        root_link = 'pelvis'  # 从XML中我们知道根连杆是pelvis
        if root_link in self.robot_data['links']:
            transforms[root_link] = base_transform
        
        # 递归计算所有连杆的变换
        self._calculate_link_transforms_mujoco(root_link, transforms, actuated_joint_angles)
        
        return transforms
    
    def _calculate_link_transforms_mujoco(self, link_name: str, transforms: Dict[str, np.ndarray], joint_angles: np.ndarray):
        """递归计算MuJoCo格式的连杆变换"""
        if link_name not in self.robot_data['child_map']:
            return
        
        for child_link in self.robot_data['child_map'][link_name]:
            # 查找连接到此子连杆的关节
            child_joint = None
            joint_index = -1
            
            # 在子连杆对应的关节中查找
            for joint_name, joint in self.robot_data['joints'].items():
                if joint.parent == link_name:
                    # 检查这个关节是否在驱动关节列表中
                    if joint_name in self.robot_data['joint_order']:
                        child_joint = joint
                        joint_index = self.robot_data['joint_order'].index(joint_name)
                        break
            
            if child_joint and 0 <= joint_index < len(joint_angles):
                # 获取关节角度
                joint_angle = joint_angles[joint_index]
                
                # 计算关节变换
                joint_transform = self._get_joint_transform(child_joint, joint_angle)
                
                # 计算子连杆的变换
                transforms[child_link] = np.dot(transforms[link_name], joint_transform)
            else:
                # 如果没有关节，使用连杆的原始位置
                link_transform = np.eye(4)
                link = self.robot_data['links'].get(child_link)
                if link and link.origin is not None:
                    link_transform[:3, 3] = link.origin[:3]
                
                transforms[child_link] = np.dot(transforms[link_name], link_transform)
            
            # 递归处理子连杆
            self._calculate_link_transforms_mujoco(child_link, transforms, joint_angles)
    
    def _get_joint_transform(self, joint: Joint, angle: float) -> np.ndarray:
        """计算单个关节的变换矩阵"""
        # 创建4x4变换矩阵
        T = np.eye(4)
        
        # 应用关节原点的平移
        T[:3, 3] = joint.origin[:3]
        
        # 应用关节原点的旋转
        if SCIPY_AVAILABLE:
            R_origin = R.from_euler('xyz', joint.origin[3:]).as_matrix()
            T[:3, :3] = R_origin
        
        # 应用关节旋转
        if joint.joint_type == 'revolute':
            if SCIPY_AVAILABLE:
                axis_rotation = R.from_rotvec(joint.axis * angle).as_matrix()
                T[:3, :3] = np.dot(T[:3, :3], axis_rotation)
            else:
                # 简化处理：只处理Z轴旋转
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                if np.abs(joint.axis[2]) > 0.5:  # Z轴旋转
                    rot_z = np.array([[cos_a, -sin_a, 0],
                                      [sin_a, cos_a, 0],
                                      [0, 0, 1]])
                    T[:3, :3] = np.dot(T[:3, :3], rot_z)
        
        return T
    
    def update_frame(self, frame):
        """更新动画帧"""
        self.frame_count = frame
        
        # 清除所有子图
        self.ax_3d.clear()
        self.ax_joints.clear()
        self.ax_timeline.clear()
        
        # 获取当前帧的qpos
        if self.npz_data and 'qpos' in self.npz_data:
            qpos = self.npz_data['qpos'][frame % len(self.npz_data['qpos'])]
            
            # 计算正向运动学
            link_transforms = self.calculate_forward_kinematics(qpos)
            
            # 更新3D视图
            self._update_3d_robot_view(link_transforms, frame)
            
            # 更新关节图
            self._update_joint_plots(qpos, frame)
        
        plt.tight_layout()
    
    def _update_3d_robot_view(self, link_transforms: Dict[str, np.ndarray], frame: int):
        """更新3D机器人视图"""
        self.ax_3d.set_title(f'G1 3D - Frames {frame}', fontsize=18, fontweight='bold')
        
        # 如果有变换矩阵，使用完整的运动学
        if link_transforms:
            self._draw_robot_with_kinematics(link_transforms)
        else:
            # 备用方案：直接显示所有STL文件（静态）
            self._draw_static_robot()
        
        # 设置坐标轴
        self._setup_3d_axes()
    
    def _draw_robot_with_kinematics(self, link_transforms: Dict[str, np.ndarray]):
        """使用运动学绘制机器人"""
        # 绘制每个连杆的STL网格
        for link_name, transform in link_transforms.items():
            link = self.robot_data['links'].get(link_name)
            if link and link.mesh_file:
                # 查找对应的STL文件
                mesh_name = Path(link.mesh_file).stem if link.mesh_file.endswith('.stl') else link.mesh_file
                
                # 尝试多种可能的文件名匹配
                mesh_data = None
                possible_names = [
                    mesh_name,
                    link_name,
                    link_name.replace('_link', ''),
                    mesh_name.replace('_link', '')
                ]
                
                for name in possible_names:
                    if name in self.mesh_dict:
                        mesh_data = self.mesh_dict[name]
                        break
                
                if mesh_data:
                    self._draw_mesh(mesh_data, transform, link_name)
    
    def _draw_static_robot(self):
        """绘制静态机器人（无运动学）"""
        print(f"使用静态显示模式，显示 {len(self.mesh_dict)} 个STL文件")
        
        # 创建一个合理的布局来显示所有STL文件
        mesh_names = list(self.mesh_dict.keys())
        
        # 按类型组织网格
        groups = {
            'torso': [],
            'left_arm': [],
            'right_arm': [],
            'left_leg': [],
            'right_leg': [],
            'head': [],
            'other': []
        }
        
        for name in mesh_names:
            name_lower = name.lower()
            if 'torso' in name_lower or 'pelvis' in name_lower or 'waist' in name_lower:
                groups['torso'].append(name)
            elif 'left' in name_lower and ('arm' in name_lower or 'shoulder' in name_lower or 'elbow' in name_lower or 'wrist' in name_lower or 'hand' in name_lower):
                groups['left_arm'].append(name)
            elif 'right' in name_lower and ('arm' in name_lower or 'shoulder' in name_lower or 'elbow' in name_lower or 'wrist' in name_lower or 'hand' in name_lower):
                groups['right_arm'].append(name)
            elif 'left' in name_lower and ('leg' in name_lower or 'hip' in name_lower or 'knee' in name_lower or 'ankle' in name_lower):
                groups['left_leg'].append(name)
            elif 'right' in name_lower and ('leg' in name_lower or 'hip' in name_lower or 'knee' in name_lower or 'ankle' in name_lower):
                groups['right_leg'].append(name)
            elif 'head' in name_lower:
                groups['head'].append(name)
            else:
                groups['other'].append(name)
        
        # 为每个组创建基本的位置偏移
        group_positions = {
            'torso': np.array([0, 0, 1]),
            'left_arm': np.array([-0.5, 0, 1.2]),
            'right_arm': np.array([0.5, 0, 1.2]),
            'left_leg': np.array([-0.2, 0, 0.5]),
            'right_leg': np.array([0.2, 0, 0.5]),
            'head': np.array([0, 0, 1.7]),
            'other': np.array([0, 0, 0])
        }
        
        # 绘制每个组
        for group_name, mesh_names_in_group in groups.items():
            base_pos = group_positions[group_name]
            
            for i, mesh_name in enumerate(mesh_names_in_group):
                if mesh_name in self.mesh_dict:
                    # 创建简单的变换矩阵
                    transform = np.eye(4)
                    offset = base_pos + np.array([0, i * 0.1, 0])  # 轻微偏移避免重叠
                    transform[:3, 3] = offset
                    
                    mesh_data = self.mesh_dict[mesh_name]
                    self._draw_mesh(mesh_data, transform, mesh_name)
    
    def _draw_mesh(self, mesh_data: MeshData, transform: np.ndarray, link_name: str):
        """绘制变换后的网格"""
        if len(mesh_data.vertices) == 0:
            return
        
        # 应用变换到顶点
        vertices_homo = np.ones((len(mesh_data.vertices), 4))
        vertices_homo[:, :3] = mesh_data.vertices
        transformed_vertices = np.dot(transform, vertices_homo.T).T[:, :3]
        
        # 简化网格以提高性能
        if len(mesh_data.faces) > 1000:
            faces = mesh_data.faces
        else:
            faces = mesh_data.faces
        
        # 创建3D多边形集合
        triangles = []
        for face in faces:
            if len(face) >= 3:
                triangle = transformed_vertices[face[:3]]
                triangles.append(triangle)
        
        if triangles:
            # 创建颜色（根据连杆名称）
            color = plt.cm.Set3(hash(link_name) % 12 / 12.0)
            
            # 添加到图中
            poly_collection = Poly3DCollection(triangles, alpha=0.7, facecolor=color, edgecolor='black', linewidth=0.1)
            self.ax_3d.add_collection3d(poly_collection)
    
    def _setup_3d_axes(self):
        """设置3D坐标轴"""
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        
        # 设置坐标范围
        range_val = 1.5
        self.ax_3d.set_xlim([-range_val, range_val])
        self.ax_3d.set_ylim([-range_val, range_val])
        self.ax_3d.set_zlim([0, range_val * 2])
        
        # 设置视角
        self.ax_3d.view_init(elev=15, azim=45 + self.frame_count * 0.5)
    
    def _update_joint_plots(self, qpos: np.ndarray, frame: int):
        """更新关节相关图表"""
        # 关节角度柱状图
        joint_names = list(self.robot_data['joints'].keys())[:len(qpos)]
        bars = self.ax_joints.bar(range(len(qpos)), qpos, alpha=0.7, color='lightblue')
        
        # 添加数值标签
        for i, (bar, value) in enumerate(zip(bars, qpos)):
            if abs(value) > 0.05:
                self.ax_joints.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                                   f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        self.ax_joints.set_xlabel('joints ID')
        self.ax_joints.set_ylabel('Angle (rad)')
        self.ax_joints.grid(True, alpha=0.3)
        
        # 关节时序图
        if frame > 0:
            window_size = min(50, frame + 1)
            start_frame = max(0, frame - window_size + 1)
            time_steps = range(start_frame, frame + 1)
            
            # 显示前几个关节的时序
            num_joints = min(6, len(qpos))
            for j in range(num_joints):
                joint_history = self.npz_data['qpos'][start_frame:frame+1, j]
                self.ax_timeline.plot(time_steps, joint_history, 
                                    label=f'Joint {j}', linewidth=2, alpha=0.8)
            
            self.ax_timeline.set_xlabel('Frame')
            self.ax_timeline.set_ylabel('Angle (rad)')
            self.ax_timeline.legend(loc='upper right', fontsize=8)
            self.ax_timeline.grid(True, alpha=0.3)
    
    def start_animation(self, interval: int = 50):
        """开始动画播放"""
        if not self.npz_data or 'qpos' not in self.npz_data:
            print("没有可用的qpos数据")
            return
        
        total_frames = len(self.npz_data['qpos'])
        print(f"开始播放机器人动画，总帧数: {total_frames}")
        
        self.animation = FuncAnimation(
            self.fig, 
            self.update_frame, 
            frames=total_frames,
            interval=interval,
            repeat=True,
            blit=False
        )
        
        plt.show()
    
    def save_animation(self, filename: str, fps: int = 20):
        """保存动画"""
        if not self.npz_data or 'qpos' not in self.npz_data:
            print("没有可用的qpos数据")
            return
        
        total_frames = len(self.npz_data['qpos'])
        print(f"正在保存机器人动画到 {filename}...")
        
        animation = FuncAnimation(
            self.fig, 
            self.update_frame, 
            frames=total_frames,
            interval=1000//fps,
            repeat=False,
            blit=False
        )
        
        animation.save(filename, writer='pillow', fps=fps)
        print(f"动画已保存为: {filename}")


def main():
    parser = argparse.ArgumentParser(description='机器人3D动画可视化工具')
    parser.add_argument('--npz', type=str, required=True, help='NPZ数据文件路径')
    parser.add_argument('--xml', type=str, required=True, help='机器人XML配置文件路径')
    parser.add_argument('--assets', type=str, required=True, help='STL文件夹路径')
    parser.add_argument('--save', type=str, help='保存动画文件名')
    parser.add_argument('--fps', type=int, default=20, help='动画帧率')
    parser.add_argument('--interval', type=int, default=100, help='动画间隔（毫秒）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    for file_path, name in [(args.npz, 'NPZ'), (args.xml, 'XML'), (args.assets, 'Assets')]:
        if not os.path.exists(file_path):
            print(f"错误: {name}文件/文件夹不存在: {file_path}")
            return
    
    # 创建机器人动画器
    animator = RobotAnimator()
    
    # 加载数据
    print("加载机器人数据...")
    if not animator.load_data(args.npz, args.xml, args.assets):
        print("数据加载失败，请检查文件路径和格式")
        return
    
    if args.save:
        # 保存动画
        animator.save_animation(args.save, args.fps)
    else:
        # 开始交互式动画
        print("启动机器人3D动画...")
        print("提示:")
        print("  - 关闭窗口以退出")
        print("  - 动画会自动循环播放")
        print("  - 3D视图会自动旋转")
        
        animator.start_animation(args.interval)


if __name__ == "__main__":
    # 检查基本依赖
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:
        print(f"缺少基本依赖: {e}")
        print("请安装: pip install matplotlib numpy")
        sys.exit(1)
    
    main()