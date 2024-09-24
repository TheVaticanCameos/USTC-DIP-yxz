import cv2
import numpy as np
import gradio as gr
from annoy import AnnoyIndex


# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img: np.array) -> np.array:
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData) -> np.array:
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image


# 执行仿射变换
def point_guided_deformation(image: np.array, 
                             source_pts: np.array, 
                             target_pts: np.array, 
                             alpha: float = 1.0, 
                             eps: float = 1e-8) -> np.array:
    """ 
    Params
    ------
        - image: 待变换的图像
        - source_pts: 控制点的坐标列表，nx2
        - target_pts: 目标点的坐标列表，nx2
        - alpha: 插值权重，默认为1.0
        - eps: 浮点数精度，默认为1e-8

    Return
    ------
        A deformed image.
    """

    original_image = np.array(image)
    warped_image = np.zeros_like(original_image)
    
    ### FILL: 基于 MLS 实现 image warping
    
    def get_bot(v: np.array) -> np.array:
        return np.column_stack((-v[:, 1], v[:, 0]))
        
    def rigid_transform(v: np.array) -> np.array:
        weights = 1.0 / (np.power(np.linalg.norm(source_pts - v, axis=1), 2*alpha) + eps)
        pStar = np.dot(weights, source_pts) / np.sum(weights)
        qStar = np.dot(weights, target_pts) / np.sum(weights)
        p_hat = source_pts - pStar
        q_hat = target_pts - qStar
        p_hat_bot = get_bot(p_hat)
        q_hat_bot = get_bot(q_hat)

        part1 = np.dot(weights, np.sum(q_hat * p_hat, axis=1))
        part2 = np.dot(weights, np.sum(q_hat * p_hat_bot, axis=1))

        mu = np.sqrt(part1**2 + part2**2)

        m11 = np.dot(weights, np.sum(p_hat * q_hat, axis=1))
        m12 = np.dot(weights, np.sum(-p_hat * q_hat_bot, axis=1))
        m21 = np.dot(weights, np.sum(-p_hat_bot * q_hat, axis=1))
        m22 = np.dot(weights, np.sum(p_hat_bot * q_hat_bot, axis=1))
        
        M = np.array([[m11, m12], [m21, m22]]) / mu
        
        return (v - pStar) @ M + qStar


    mask = np.ones(shape=(original_image.shape[0], original_image.shape[1]))
    index = AnnoyIndex(2, 'euclidean')
    
    for i in range(warped_image.shape[0]):
        for j in range(warped_image.shape[1]):
            idx = rigid_transform(np.array([i, j])).astype(int)
            if not (idx[0] < 0 or idx[0] >= original_image.shape[0] or idx[1] < 0 or idx[1] >= original_image.shape[1]):
                warped_image[idx[0], idx[1]] = original_image[i, j]
                mask[idx[0], idx[1]] = 0
                index.add_item(i * original_image.shape[1] + j, idx)

    index.build(8, n_jobs=-1)

    for needFill in np.argwhere(mask == 1):
        i, j = needFill[0], needFill[1]
        idx = index.get_nns_by_vector(np.array([i, j]), 1)[0]
        warped_image[i, j] = original_image[idx // original_image.shape[1], idx % original_image.shape[1]]

    return warped_image


def run_warping() -> np.array:
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image


# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图


# 使用 Gradio 构建界面
if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
                point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
                
            with gr.Column():
                result_image = gr.Image(label="变换结果", width=800, height=400)
        
        # 按钮
        run_button = gr.Button("Run Warping")
        clear_button = gr.Button("Clear Points")  # 添加清除按钮
        
        # 上传图像的交互
        input_image.upload(upload_image, input_image, point_select)
        # 选择点的交互，点选后刷新图像
        point_select.select(record_points, None, point_select)
        # 点击运行 warping 按钮，计算并显示变换后的图像
        run_button.click(run_warping, None, result_image)
        # 点击清除按钮，清空所有已选择的点
        clear_button.click(clear_points, None, point_select)

        
    # 启动 Gradio 应用
    demo.launch()
