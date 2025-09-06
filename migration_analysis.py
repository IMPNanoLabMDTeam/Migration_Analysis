import MDemon as md
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter
import time
import os
import argparse
import re

def process_model(model_path, massdensity_target_list, massdensity_range_list):
    """处理单个模型文件"""
    print(f"[LOG] 开始处理模型: {model_path}")
    
    # 从模型路径推导data和reaxff文件路径
    if model_path.endswith('.data'):
        datafile = model_path
        bondfile = model_path.replace('.data', '.reaxff')
    else:
        datafile = model_path + '.data'
        bondfile = model_path + '.reaxff'
    
    print(f"[LOG] 读取数据文件: {datafile}")
    print(f"[LOG] 读取键文件: {bondfile}")
    
    if not os.path.exists(datafile):
        raise FileNotFoundError(f"数据文件不存在: {datafile}")
    if not os.path.exists(bondfile):
        raise FileNotFoundError(f"键文件不存在: {bondfile}")
    
    u1 = md.Universe(datafile, bondfile)
    print(f"[LOG] Universe创建完成，原子数: {len(u1.atoms)}")
    atoms = u1.atoms
    
    # 分析分子质量
    print(f"[LOG] 开始分析分子质量，分子数: {len(u1.molecules)}")
    m0 = 0
    for mol in u1.molecules:
        if m0 < mol.mass:
            m0 = mol.mass
    print(f"[LOG] 最大分子质量: {m0}")

    # 确定所有需要分析的质量密度原子分组
    print(f"[LOG] 设置Core质量范围为: ({m0}, {m0+1})")
    massdensity_range_list[massdensity_target_list.index('Core')] = (m0, m0+1)
    
    print(f"[LOG] 开始按质量密度分组原子")
    massdensity_atoms_list = [[] for _ in massdensity_target_list]
    massdensity_masses_list = [[] for _ in massdensity_target_list]
    
    for mol in u1.molecules:
        matching_indices = [i for i, (start, end) in enumerate(massdensity_range_list) if start <= mol.mass < end]
        for index in matching_indices:
            massdensity_atoms_list[index].extend(list(mol.atms.keys()))

    print(f"[LOG] 处理各分组的质量数据")
    for index in range(len(massdensity_target_list)):
        atoms1 = atoms[massdensity_atoms_list[index]]
        massdensity_atoms_list[index] = atoms1
        masses = np.full(len(atoms1.ix), 0, dtype=np.float32)
        for i, ix in enumerate(atoms1.ix):
            masses[i] = u1.atoms[ix].mass
        NA = 6.022140857e23
        massdensity_masses_list[index] = masses/(NA/1e21)
        print(f"[LOG] {massdensity_target_list[index]} 分组: {len(atoms1.ix)} 个原子")

    print(f"[LOG] 模型处理完成")
    return u1, massdensity_atoms_list, massdensity_masses_list

def extract_migration_data(model1_path, model2_path, output_dir="results", file_id=""):
    """提取两个模型之间的迁移数据"""
    print("="*60)
    print("[LOG] 开始提取迁移数据")
    print(f"[LOG] 模型1: {model1_path}")
    print(f"[LOG] 模型2: {model2_path}")
    print(f"[LOG] 输出目录: {output_dir}")
    print(f"[LOG] 文件编号: {file_id}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义质量密度分组
    massdensity_target_list = ['Gas', 'Nogas', 'Core', 'Segments', 'Brokenchains', 'Chains', 'Precursors','Macromolecules']
    massdensity_range_list = [(0,100),(100,np.inf),(None,None),(100,1000),(1000,4000),(4000,4200),(4200,50000),(50000,np.inf)]
    print(f"[LOG] 质量密度分组定义完成，共{len(massdensity_target_list)}个分组")

    print("\n" + "="*60)
    print("[LOG] 第1步：处理第一个模型")
    start_time = time.time()
    u2, atomslist2, masseslist2 = process_model(model1_path, massdensity_target_list, massdensity_range_list.copy())
    print(f"[LOG] 第1步完成，耗时: {time.time()-start_time:.2f}秒")

    print("\n" + "="*60)
    print("[LOG] 第2步：处理第二个模型")
    start_time = time.time()
    u3, atomslist3, masseslist3 = process_model(model2_path, massdensity_target_list, massdensity_range_list.copy())
    print(f"[LOG] 第2步完成，耗时: {time.time()-start_time:.2f}秒")

    print("\n" + "="*60)
    print("[LOG] 第3步：处理盒子参数和坐标")
    L = u2.box[0:3]
    CM = ((u2.box[6:][::2] + u2.box[6:][1::2])/2)[0:3]
    print(f"[LOG] 盒子尺寸: {L}")
    print(f"[LOG] 盒子中心: {CM}")

    # 只考虑最后没有变成气体的那些原子
    print("[LOG] 筛选非气体原子")
    atomsGas3 = atomslist3[massdensity_target_list.index('Gas')]
    ixNogas = np.setdiff1d(u2.atoms.ix, atomsGas3.ix)
    print(f"[LOG] 气体原子数: {len(atomsGas3.ix)}")
    print(f"[LOG] 非气体原子数: {len(ixNogas)}")

    # 处理原子坐标并计算xy平面位移
    print("[LOG] 处理原子坐标并计算xy平面位移")

    # 定义wrap函数，确保坐标在周期性边界内
    def wrap_coordinates(coords, box_bounds):
        """将坐标wrap到周期性边界内"""
        wrapped_coords = coords.copy()
        for i in range(coords.shape[1]):  # 对每个维度
            box_min, box_max = box_bounds[i]
            box_length = box_max - box_min
            # 将坐标移动到[0, box_length)范围内，然后再移回原始位置
            wrapped_coords[:, i] = ((coords[:, i] - box_min) % box_length) + box_min
        return wrapped_coords

    # 获取盒子边界信息
    x0, x1 = u2.box[6], u2.box[7]  # x边界
    y0, y1 = u2.box[8], u2.box[9]  # y边界
    box_bounds_xy = np.array([[x0, x1], [y0, y1]])
    
    # 提取xy坐标并进行wrap处理
    u2_xy_raw = u2.atoms.coordinate[:, :2][ixNogas]  # 只取xy坐标，只考虑非气体原子
    u3_xy_raw = u3.atoms.coordinate[:, :2][ixNogas]  # 只取xy坐标，只考虑非气体原子
    
    print("[LOG] 对坐标进行wrap处理")
    u2_xy = wrap_coordinates(u2_xy_raw, box_bounds_xy)
    u3_xy = wrap_coordinates(u3_xy_raw, box_bounds_xy)

    # 计算原始位移 (单位: Angstrom)
    raw_displacement_xy = u3_xy - u2_xy

    # 处理周期性边界条件
    # 如果位移大于L/2，说明跨越了边界，需要减去一个周期长度
    Lxy = L[:2]  # 取xy方向的盒子长度
    print(f"[LOG] 盒子xy尺寸: {Lxy}")

    # 对x和y方向分别处理周期性边界
    corrected_displacement_xy = raw_displacement_xy.copy()
    for i in range(2):  # x和y方向
        # 找到位移大于L/2的原子
        large_positive = raw_displacement_xy[:, i] > Lxy[i]/2
        large_negative = raw_displacement_xy[:, i] < -Lxy[i]/2
        
        # 修正跨边界的位移
        corrected_displacement_xy[large_positive, i] -= Lxy[i]
        corrected_displacement_xy[large_negative, i] += Lxy[i]
        
        print(f"[LOG] {['x','y'][i]}方向跨边界原子数: +{large_positive.sum()}, -{large_negative.sum()}")

    # 转换为nm
    displacement_xy = corrected_displacement_xy / 10  
    displacement_magnitude = np.linalg.norm(displacement_xy, axis=1)  # 位移大小

    print(f"[LOG] 计算完成，共{len(ixNogas)}个非气体原子")
    print(f"[LOG] xy位移大小范围: {displacement_magnitude.min():.3f} - {displacement_magnitude.max():.3f} nm")
    print(f"[LOG] 平均位移大小: {displacement_magnitude.mean():.3f} nm")
    print(f"[LOG] x方向位移范围: {displacement_xy[:,0].min():.3f} - {displacement_xy[:,0].max():.3f} nm")
    print(f"[LOG] y方向位移范围: {displacement_xy[:,1].min():.3f} - {displacement_xy[:,1].max():.3f} nm")

    print("\n" + "="*60)
    print("[LOG] 第4步：输出数据文件")

    # 在输出前对坐标进行额外处理
    print("[LOG] 对坐标进行最终处理：平移和重新wrap")
    
    # 计算dxyz（这里简化处理，设为零向量）
    dxyz = np.array([0.0, 0.0, 0.0])
    dxyz_xy = dxyz[:2]  # 取xy分量
    CM_xy = CM[:2]      # 取xy分量
    
    print(f"[LOG] dxyz_xy: {dxyz_xy}")
    print(f"[LOG] CM_xy: {CM_xy}")
    
    u2_xy_shifted = u2_xy + dxyz_xy - CM_xy
    print(f"[LOG] 坐标平移完成")
    
    # 计算原始盒子尺寸
    box_size_x = x1 - x0
    box_size_y = y1 - y0
    print(f"[LOG] 原始盒子尺寸: x={box_size_x:.3f} Å, y={box_size_y:.3f} Å")
    
    # 定义关于原点对称的盒子边界
    symmetric_box_bounds = np.array([
        [-box_size_x/2, box_size_x/2],  # x边界，关于原点对称
        [-box_size_y/2, box_size_y/2]   # y边界，关于原点对称
    ])
    print(f"[LOG] 对称盒子边界: x({-box_size_x/2:.3f}, {box_size_x/2:.3f}) y({-box_size_y/2:.3f}, {box_size_y/2:.3f}) Å")
    
    # 将平移后的坐标wrap到对称盒子中
    u2_xy_final = wrap_coordinates(u2_xy_shifted, symmetric_box_bounds)
    print(f"[LOG] 坐标wrap到对称盒子完成")

    # 准备输出数据
    # 使用处理后的对称盒子边界，转换为nm
    symmetric_box_bounds_nm = symmetric_box_bounds / 10  # 转换为nm
    print(f"[LOG] 输出用对称盒子边界 (nm): x({symmetric_box_bounds_nm[0,0]:.3f}, {symmetric_box_bounds_nm[0,1]:.3f}) y({symmetric_box_bounds_nm[1,0]:.3f}, {symmetric_box_bounds_nm[1,1]:.3f})")

    # 原子坐标 (转换为nm，使用处理后的坐标)
    atom_coords_nm = np.column_stack([
        (u2_xy_final[:, 0])/10,  # x坐标 (nm，使用处理后的坐标)
        (u2_xy_final[:, 1])/10,  # y坐标 (nm，使用处理后的坐标)
        displacement_xy[:, 0],   # x方向位移 (nm)
        displacement_xy[:, 1]    # y方向位移 (nm)
    ])

    # 输出到文件
    if file_id:
        output_filename = os.path.join(output_dir, f"migration_data_{file_id}.txt")
    else:
        output_filename = os.path.join(output_dir, "migration_data.txt")
    print(f"[LOG] 正在保存数据到: {output_filename}")

    with open(output_filename, 'w') as f:
        # 写入对称盒子边界信息
        f.write("# Box boundaries (nm) - symmetric around origin\n")
        f.write(f"# x_min x_max: {symmetric_box_bounds_nm[0,0]:.6f} {symmetric_box_bounds_nm[0,1]:.6f}\n")
        f.write(f"# y_min y_max: {symmetric_box_bounds_nm[1,0]:.6f} {symmetric_box_bounds_nm[1,1]:.6f}\n")
        f.write("# Data format: x_coord y_coord x_displacement y_displacement (all in nm)\n")
        f.write("# Note: coordinates have been shifted and wrapped to symmetric box\n")
        
        # 写入原子数据
        for i in range(len(atom_coords_nm)):
            f.write(f"{atom_coords_nm[i,0]:.6f} {atom_coords_nm[i,1]:.6f} {atom_coords_nm[i,2]:.6f} {atom_coords_nm[i,3]:.6f}\n")

    print(f"[LOG] 数据保存完成！")
    print(f"[LOG] 文件包含: {len(atom_coords_nm)} 行原子数据")
    
    return output_filename, symmetric_box_bounds_nm

def read_migration_data(filename):
    """读取migration_data.txt文件"""
    print(f"[LOG] 开始读取数据文件: {filename}")
    start_time = time.time()
    
    # 读取数据，跳过注释行
    data = []
    box_info = {}
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# x_min x_max:'):
                # 解析x边界
                parts = line.split()
                box_info['x_min'] = float(parts[3])
                box_info['x_max'] = float(parts[4])
            elif line.startswith('# y_min y_max:'):
                # 解析y边界
                parts = line.split()
                box_info['y_min'] = float(parts[3])
                box_info['y_max'] = float(parts[4])
            elif not line.startswith('#'):
                # 读取数据行
                parts = line.split()
                if len(parts) == 4:
                    data.append([float(x) for x in parts])
    
    data = np.array(data)
    print(f"[LOG] 数据读取完成，耗时: {time.time()-start_time:.2f}秒")
    print(f"[LOG] 读取了 {len(data)} 个原子的数据")
    print(f"[LOG] 盒子边界: x({box_info['x_min']:.3f}, {box_info['x_max']:.3f}) y({box_info['y_min']:.3f}, {box_info['y_max']:.3f}) nm")
    
    return data, box_info

def create_vector_field(data, box_info, grid_size=100):
    """创建矢量场网格"""
    print(f"[LOG] 开始创建 {grid_size}×{grid_size} 矢量场网格")
    start_time = time.time()
    
    # 提取坐标和位移
    x_coords = data[:, 0]  # x坐标
    y_coords = data[:, 1]  # y坐标
    x_displacements = data[:, 2]  # x方向位移
    y_displacements = data[:, 3]  # y方向位移
    
    # 创建规则网格
    x_min, x_max = box_info['x_min'], box_info['x_max']
    y_min, y_max = box_info['y_min'], box_info['y_max']
    
    xi = np.linspace(x_min, x_max, grid_size)
    yi = np.linspace(y_min, y_max, grid_size)
    Xi, Yi = np.meshgrid(xi, yi)
    
    print(f"[LOG] 开始网格内平均计算...")
    
    # 初始化矢量场数组
    Ui = np.zeros((grid_size, grid_size))
    Vi = np.zeros((grid_size, grid_size))
    counts = np.zeros((grid_size, grid_size))
    
    # 计算网格间距
    dx = (x_max - x_min) / (grid_size - 1)
    dy = (y_max - y_min) / (grid_size - 1)
    
    # 将每个原子分配到对应的网格中并累加位移
    for i in range(len(data)):
        x, y = x_coords[i], y_coords[i]
        
        # 计算原子所属的网格索引
        grid_x = int((x - x_min) / dx)
        grid_y = int((y - y_min) / dy)
        
        # 确保索引在有效范围内
        if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
            Ui[grid_y, grid_x] += x_displacements[i]
            Vi[grid_y, grid_x] += y_displacements[i]
            counts[grid_y, grid_x] += 1
    
    # 计算平均位移（避免除零）
    mask = counts > 0
    Ui[mask] /= counts[mask]
    Vi[mask] /= counts[mask]
    
    # 统计信息
    non_empty_cells = np.sum(mask)
    total_cells = grid_size * grid_size
    print(f"[LOG] 网格统计: {non_empty_cells}/{total_cells} 个网格包含原子")
    print(f"[LOG] 平均每个网格的原子数: {len(data)/non_empty_cells:.1f}")
    
    print(f"[LOG] 矢量场创建完成，耗时: {time.time()-start_time:.2f}秒")
    
    return Xi, Yi, Ui, Vi

def plot_vector_field(Xi, Yi, Ui, Vi, box_info, output_dir="results", output_filename="displacement_field.png", file_id="", colorbar_max=1.0, gif_title="2D Displacement Vector Field - Growing Animation"):
    """绘制矢量场图
    
    参数:
        colorbar_max (float): colorbar的最大值，用于设置颜色条的范围 (默认: 1.0)
        gif_title (str): 图表标题 (默认: "2D Displacement Vector Field - Growing Animation")
    """
    print(f"[LOG] 开始绘制矢量场图")
    start_time = time.time()
    
    # 如果有文件编号，修改输出文件名
    if file_id and output_filename == "displacement_field.png":
        output_filename = f"displacement_field_{file_id}.png"
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 计算矢量的大小用于颜色映射
    magnitude = np.sqrt(Ui**2 + Vi**2)
    
    # 绘制矢量场，使用颜色表示矢量大小
    # 调整箭头大小和样式以提高可见性
    quiver = ax.quiver(Xi, Yi, Ui, Vi, magnitude, 
                      scale=1, scale_units='xy', angles='xy',
                      cmap='plasma', alpha=0.9, width=0.004, 
                      headwidth=3.5, headlength=4)
    
    # 添加颜色条，范围为0到colorbar_max nm
    cbar = plt.colorbar(quiver, ax=ax, shrink=0.8, pad=0.02)
    cbar.mappable.set_clim(vmin=0, vmax=colorbar_max)
    cbar.set_label('Displacement Magnitude (nm)', fontsize=12, labelpad=15)
    cbar.ax.tick_params(labelsize=10)
    
    # 设置颜色条的刻度格式
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    
    # 设置坐标轴（限制显示范围为 -5 到 5 nm）
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('X Position (nm)', fontsize=12)
    ax.set_ylabel('Y Position (nm)', fontsize=12)
    ax.set_title(gif_title, fontsize=18, fontweight='bold')
    
    # 设置等比例坐标轴
    ax.set_aspect('equal')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 保存图片到输出目录
    output_path = os.path.join(output_dir, output_filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[LOG] 矢量场图已保存: {output_path}")
    print(f"[LOG] 绘图完成，耗时: {time.time()-start_time:.2f}秒")
    
    # 显示统计信息
    print(f"[LOG] 矢量场统计:")
    print(f"[LOG] 位移大小范围: {magnitude.min():.4f} - {magnitude.max():.4f} nm")
    print(f"[LOG] 平均位移大小: {magnitude.mean():.4f} nm")
    
    return fig, ax

def create_animated_vector_field(Xi, Yi, Ui, Vi, box_info, output_dir="results", 
                                output_filename="displacement_field_animation.gif", 
                                duration=3.0, fps=30, file_id="", smooth_animation=True, colorbar_max=1.0, gif_title="2D Displacement Vector Field - Growing Animation"):
    """创建动态矢量场动画
    
    参数:
        smooth_animation (bool): 是否使用平滑的缓动效果。
                               True: 使用ease-in-out缓动函数，动画开始和结束时较慢
                               False: 使用线性增长，无润滑效果
        colorbar_max (float): colorbar的最大值，用于设置颜色条的范围 (默认: 1.0)
        gif_title (str): GIF动画的标题内容 (默认: "2D Displacement Vector Field - Growing Animation")
    """
    print(f"[LOG] 开始创建动态矢量场动画")
    start_time = time.time()
    
    # 如果有文件编号，修改输出文件名
    if file_id and output_filename == "displacement_field_animation.gif":
        output_filename = f"displacement_field_animation_{file_id}.gif"
    
    # 计算矢量的大小用于颜色映射
    magnitude = np.sqrt(Ui**2 + Vi**2)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 设置坐标轴（限制显示范围为 -5 到 5 nm）
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel('X Position (nm)', fontsize=12)
    ax.set_ylabel('Y Position (nm)', fontsize=12)
    ax.set_title(gif_title, fontsize=18, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # 计算动画帧数
    total_frames = int(duration * fps)
    
    # 初始化空的quiver plot
    quiver = ax.quiver(Xi, Yi, np.zeros_like(Ui), np.zeros_like(Vi), 
                      np.zeros_like(magnitude),
                      scale=1, scale_units='xy', angles='xy',
                      cmap='plasma', alpha=0.9, width=0.004, 
                      headwidth=3.5, headlength=4)
    
    # 添加颜色条，范围为0到colorbar_max nm
    cbar = plt.colorbar(quiver, ax=ax, shrink=0.8, pad=0.02)
    cbar.mappable.set_clim(vmin=0, vmax=colorbar_max)
    cbar.set_label('Displacement Magnitude (nm)', fontsize=12, labelpad=15)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    
    def animate(frame):
        """动画更新函数"""
        # 计算当前帧的生长比例（从0到1）
        growth_factor = frame / (total_frames - 1) if total_frames > 1 else 1
        
        if smooth_animation:
            # 使用平滑的缓动函数（ease-in-out）
            # 这会让动画开始和结束时较慢，中间较快
            if growth_factor <= 0.5:
                smooth_factor = 2 * growth_factor * growth_factor
            else:
                smooth_factor = 1 - 2 * (1 - growth_factor) * (1 - growth_factor)
        else:
            # 线性增长，无润滑效果
            smooth_factor = growth_factor
        
        # 计算当前帧的矢量大小
        current_U = Ui * smooth_factor
        current_V = Vi * smooth_factor
        current_magnitude = magnitude * smooth_factor
        
        # 更新quiver数据
        quiver.set_UVC(current_U, current_V, current_magnitude)
        
        return [quiver]
    
    print(f"[LOG] 开始渲染动画，总帧数: {total_frames}")
    
    # 创建动画
    anim = animation.FuncAnimation(fig, animate, frames=total_frames, 
                                 interval=1000/fps, blit=True, repeat=True)
    
    # 保存动画
    output_path = os.path.join(output_dir, output_filename)
    print(f"[LOG] 正在保存动画到: {output_path}")
    
    # 保存为GIF
    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer)
    
    print(f"[LOG] 动画创建完成，耗时: {time.time()-start_time:.2f}秒")
    print(f"[LOG] 动画文件已保存: {output_path}")
    print(f"[LOG] 动画参数: 时长={duration}秒, 帧率={fps}fps, 总帧数={total_frames}")
    
    return fig, anim

def main(model1_path, model2_path, output_dir="results", grid_size=100, 
         create_animation=False, animation_duration=3.0, animation_fps=30, file_id="", smooth_animation=True, colorbar_max=1.0, data_dir=None, gif_title="2D Displacement Vector Field - Growing Animation"):
    """
    主函数
    
    参数:
        model1_path: 第一个模型文件路径
        model2_path: 第二个模型文件路径
        output_dir: 输出目录
        grid_size: 网格大小
        create_animation: 是否创建动画
        animation_duration: 动画时长
        animation_fps: 动画帧率
        file_id: 文件标识符
        smooth_animation: 是否使用平滑动画效果
        colorbar_max: colorbar最大值
        data_dir: 数据目录，如果提供则直接读取已有的migration_data文件
        gif_title: GIF动画的标题内容
    """
    print("="*60)
    print("[LOG] 迁移场分析程序开始运行")
    print(f"[LOG] 模型1路径: {model1_path}")
    print(f"[LOG] 模型2路径: {model2_path}")
    print(f"[LOG] 输出目录: {output_dir}")
    print(f"[LOG] 网格大小: {grid_size}×{grid_size}")
    if file_id:
        print(f"[LOG] 文件编号: {file_id}")
    if create_animation:
        print(f"[LOG] 动画模式: 开启 (时长={animation_duration}s, 帧率={animation_fps}fps)")
    
    # 检查是否直接从数据目录读取已有文件
    if data_dir and file_id:
        data_filename = f"migration_data_{file_id}.txt"
        data_file = os.path.join(data_dir, data_filename)
        
        if os.path.exists(data_file):
            print(f"[LOG] 从数据目录读取已有文件: {data_file}")
            print("[LOG] 跳过数据提取步骤，直接读取数据")
        else:
            print(f"[ERROR] 指定的数据文件不存在: {data_file}")
            exit(1)
    else:
        # 第一步：提取迁移数据
        data_file, box_info = extract_migration_data(model1_path, model2_path, output_dir, file_id)
    
    print("\n" + "="*60)
    print("[LOG] 第5步：创建矢量场网格")
    
    # 读取数据
    data, box_info_dict = read_migration_data(data_file)
    
    # 创建矢量场
    Xi, Yi, Ui, Vi = create_vector_field(data, box_info_dict, grid_size=grid_size)
    
    print("\n" + "="*60)
    print("[LOG] 第6步：绘制矢量场图")
    
    # 绘制静态矢量场
    fig, ax = plot_vector_field(Xi, Yi, Ui, Vi, box_info_dict, output_dir, file_id=file_id, colorbar_max=colorbar_max, gif_title=gif_title)
    
    if create_animation:
        print("\n" + "="*60)
        print("[LOG] 第7步：创建动态矢量场动画")
        
        # 创建动画
        fig_anim, anim = create_animated_vector_field(Xi, Yi, Ui, Vi, box_info_dict, output_dir,
                                                     duration=animation_duration, fps=animation_fps, file_id=file_id, smooth_animation=smooth_animation, colorbar_max=colorbar_max, gif_title=gif_title)
    
    print("\n" + "="*60)
    print("[LOG] 所有步骤完成！")
    print(f"[LOG] 输出文件保存在: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Analyze migration field between two MD models')
    parser.add_argument('model1', type=str, nargs='?', help='Path to first model file (without .data extension)')
    parser.add_argument('model2', type=str, nargs='?', help='Path to second model file (without .data extension)')
    parser.add_argument('--output-dir', '-o', type=str, default="results", 
                        help='Output directory for results (default: results)')
    parser.add_argument('--grid-size', '-g', type=int, default=100,
                        help='Grid size for vector field (default: 100)')
    parser.add_argument('--animation', '-a', action='store_true',
                        help='Create animated version of vector field (default: False)')
    parser.add_argument('--duration', '-d', type=float, default=3.0,
                        help='Animation duration in seconds (default: 3.0)')
    parser.add_argument('--fps', '-f', type=int, default=30,
                        help='Animation frame rate (default: 30)')
    parser.add_argument('--file-id', '-i', type=str, default="",
                        help='File ID for output files (default: empty)')
    parser.add_argument('--no-smooth', action='store_true',
                        help='Disable smooth animation effect (use linear growth instead)')
    parser.add_argument('--colorbar-max', '-c', type=float, default=1.0,
                        help='Maximum value for colorbar scale (default: 1.0)')
    parser.add_argument('--data-dir', '-D', type=str, default=None,
                        help='Directory containing existing migration_data_{id}.txt files (skip data extraction)')
    parser.add_argument('--gif-title', '-t', type=str, default="2D Displacement Vector Field - Growing Animation",
                        help='Title for the GIF animation (default: "2D Displacement Vector Field - Growing Animation")')

    args = parser.parse_args()
    
    # 检查参数有效性
    if args.data_dir:
        # 使用数据目录模式，检查数据目录是否存在
        if not os.path.exists(args.data_dir):
            print(f"[ERROR] 数据目录不存在: {args.data_dir}")
            exit(1)
        
        # 检查是否提供了file_id
        if not args.file_id:
            print(f"[ERROR] 使用--data-dir参数时必须提供--file-id参数")
            exit(1)
    else:
        # 常规模式，检查是否提供了必需的模型参数
        if not args.model1 or not args.model2:
            print(f"[ERROR] 常规模式下必须提供model1和model2参数")
            exit(1)
            
        # 检查模型文件是否存在
        model1_data = args.model1 + '.data' if not args.model1.endswith('.data') else args.model1
        model2_data = args.model2 + '.data' if not args.model2.endswith('.data') else args.model2
        
        if not os.path.exists(model1_data):
            print(f"[ERROR] 模型1数据文件不存在: {model1_data}")
            exit(1)
        
        if not os.path.exists(model2_data):
            print(f"[ERROR] 模型2数据文件不存在: {model2_data}")
            exit(1)
    
    # 运行主程序
    main(args.model1, args.model2, output_dir=args.output_dir, grid_size=args.grid_size, 
         create_animation=args.animation, animation_duration=args.duration, 
         animation_fps=args.fps, file_id=args.file_id, smooth_animation=not args.no_smooth, 
         colorbar_max=args.colorbar_max, data_dir=args.data_dir, gif_title=args.gif_title)