import numpy as np
import open3d as o3d
import pyrender
import trimesh
from PIL import Image


# 1. 模型加载函数（纯CPU版本）
# --------------------------------------------------
def load_model_cpu(file_path):
    # 自动检测文件类型
    if file_path.endswith(".ply"):
        # 优先作为点云加载
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            return ("pointcloud", pcd)
        except:
            mesh = o3d.io.read_triangle_mesh(file_path)
            return ("mesh", mesh)
    elif file_path.endswith(".obj") or file_path.endswith(".stl"):
        mesh = o3d.io.read_triangle_mesh(file_path)
        return ("mesh", mesh)
    else:
        raise ValueError("Unsupported file format")


# 2. 模型可视化（阻塞式窗口）
# --------------------------------------------------
def show_raw_model(data_type, model):
    if data_type == "pointcloud":
        o3d.visualization.draw_geometries([model])
    elif data_type == "mesh":
        # 为网格添加默认颜色以便显示
        model.paint_uniform_color([0.6, 0.6, 0.6])
        o3d.visualization.draw_geometries([model])


# 3. 模型转换函数
# --------------------------------------------------
def convert_to_pyrender(data_type, model):
    if data_type == "pointcloud":
        # 将Open3D点云转为PyRender点云（每个点渲染为小立方体）
        points = np.asarray(model.points)
        return pyrender.Mesh.from_points(points, colors=np.ones((len(points), 3)) * 0.5)
    else:
        # 将Open3D网格转为Trimesh再转PyRender
        vertices = np.asarray(model.vertices)
        faces = np.asarray(model.triangles)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return pyrender.Mesh.from_trimesh(mesh)


# 4. 阴影投影核心逻辑
# --------------------------------------------------
def apply_shadow_mask(scene, light_pos, mask_img_path):
    # 加载mask图像并转换为透明度通道
    mask_img = Image.open(mask_img_path).convert("L")
    mask_array = np.array(mask_img) / 255.0  # 归一化到[0,1]

    # 创建带透明度的材质
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1, 1, 1, 1],
        metallicFactor=0,
        alphaMode="BLEND",
        # transparent=True
    )

    # 创建投影平面（位于模型后方）
    projection_plane = pyrender.Mesh.from_trimesh(
        trimesh.creation.box(extents=(10, 10, 0.01)), material=material
    )

    # 将mask作为纹理应用到平面
    plane_node = scene.add(projection_plane)

    # 设置光源与投影矩阵（模拟mask投影）
    light = pyrender.SpotLight(
        intensity=1000.0, innerConeAngle=np.pi / 4, outerConeAngle=np.pi / 3
    )
    light_node = scene.add(light, pose=np.eye(4))
    light_node.matrix = _create_projection_matrix(light_pos, mask_array.shape)


# 辅助函数：创建光源投影矩阵
def _create_projection_matrix(light_pos, mask_shape):
    matrix = np.eye(4)
    matrix[:3, 3] = light_pos
    # 根据mask尺寸调整投影比例
    ratio = mask_shape[1] / mask_shape[0]
    matrix[0, 0] = ratio  # 保持mask宽高比
    return matrix


# 5. 主执行流程
# --------------------------------------------------
if __name__ == "__main__":
    # 参数配置
    model_path = "data/Projects/Project20250124_104153/data/20250124_104153/fuse.ply"  # 可替换为OBJ/STL文件
    mask_path = "mask.png"  # 黑白遮罩图
    light_position = [0, 5, 5]  # 光源位置

    # 1. 加载模型
    data_type, model = load_model_cpu(model_path)

    # 2. 显示原始模型（阻塞窗口）
    print("正在显示原始模型，关闭窗口继续执行...")
    # show_raw_model(data_type, model)

    # 3. 转换为PyRender可用的格式
    pyrender_mesh = convert_to_pyrender(data_type, model)

    # 4. 创建场景
    scene = pyrender.Scene(bg_color=[0, 0, 0, 1])
    scene.add(pyrender_mesh)

    # 5. 添加阴影投影
    apply_shadow_mask(scene, light_position, mask_path)

    # 6. 设置相机
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    camera_pose = np.eye(4)

    # 调整相机位置
    def move_camera(camera_pose, direction, distance):
        if direction == "up":
            camera_pose[1, 3] += distance
        elif direction == "down":
            camera_pose[1, 3] -= distance
        elif direction == "left":
            camera_pose[0, 3] -= distance
        elif direction == "right":
            camera_pose[0, 3] += distance
        return camera_pose

    # 示例：向上移动10个单位
    # camera_pose = move_camera(camera_pose, "down", 150)
    # camera_pose = move_camera(camera_pose, "right", 80)
    camera_pose[:3, 3] += np.array([100, -200, 0])
    print(camera_pose)
    scene.add(camera, pose=camera_pose)

    # 7. 执行渲染
    renderer = pyrender.OffscreenRenderer(800, 600)
    color, depth = renderer.render(scene)

    # 8. 后处理并保存
    # 将深度图转换为阴影效果
    # shadow_mask = (depth > 0).astype(np.uint8) * 255
    print(depth.mean(), depth.max(), depth.min())
    print(color.mean(), color.max(), color.min())
    shadow_mask = (depth / depth.max() * 255).astype(np.uint8)
    final_image = Image.fromarray(shadow_mask)
    final_image.save("shadow_projection.png")
    print("渲染结果已保存为 shadow_projection.png")
