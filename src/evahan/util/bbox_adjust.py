"""
bbox调整

如果遇到cv2导入错误，尝试安装依赖：
```shell
apt-get install -y libgl1
```
"""

# import cv2
# import numpy as np


# def is_background_pixel(pixel, bg_threshold=240):
#     """
#     判断单个像素是否为背景色（接近白色）
#     :param pixel: 像素值 (B, G, R) 或灰度值
#     :param bg_threshold: 背景色阈值（0-255，越大越严格，255为纯白）
#     :return: True=背景，False=非背景
#     """
#     # 处理彩色/灰度图像
#     if isinstance(pixel, (list, np.ndarray)) and len(pixel) >= 3:
#         # 彩色转灰度（简化背景判断）
#         gray = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2GRAY)[0][0]
#     else:
#         gray = pixel if isinstance(pixel, int) else pixel[0]

#     # 灰度值越接近255（白色），越判定为背景
#     return gray >= bg_threshold


# def get_min_rect_from_bbox(bbox_points):
#     """
#     将不规则/倾斜的bbox坐标转换为最小外接矩形（旋转矩形）
#     :param bbox_points: 原始bbox的4个坐标点，格式为 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
#     :return: 最小外接矩形的参数 (中心(x,y), 宽高(w,h), 旋转角度)
#     """
#     pts = np.array(bbox_points, dtype=np.float32)
#     min_rect = cv2.minAreaRect(pts)
#     return min_rect


# def adjust_bbox_to_content(
#     image, bbox_points, bg_threshold=240, step=1, padding=2
# ):
#     """
#     调整bbox，框入所有连续的非背景色区域，剔除空白，并添加padding保证框体不会太小
#     :param image: 原始图像（cv2.imread读取）
#     :param bbox_points: 初始bbox的4个坐标点 [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
#     :param bg_threshold: 背景色灰度阈值（越大越严格）
#     :param step: 扩展/收缩的步长（像素）
#     :param padding: 调整后添加的内边距（像素），保证框体不会太小
#     :return: 调整后的bbox最小外接矩形参数、调整后的bbox四个顶点坐标
#     """
#     # 1. 获取初始bbox的最小外接矩形
#     min_rect = get_min_rect_from_bbox(bbox_points)
#     (cx, cy), (w, h), angle = min_rect

#     # 2. 提取初始矩形的所有像素坐标（生成掩码）
#     # 转换为矩形的四个顶点
#     box_pts = cv2.boxPoints(min_rect).astype(np.int32)
#     # 获取初始矩形的边界范围
#     x_min, y_min = np.min(box_pts, axis=0)
#     x_max, y_max = np.max(box_pts, axis=0)

#     # 3. 扩展阶段：向外逐像素检测，直到遇到连续非背景色
#     expand_x_min = x_min
#     expand_x_max = x_max
#     expand_y_min = y_min
#     expand_y_max = y_max

#     # 图像边界（防止越界）
#     img_h, img_w = image.shape[:2]

#     # 向左扩展
#     while expand_x_min - step >= 0:
#         # 检测当前列是否有非背景色
#         has_content = False
#         for y in range(expand_y_min, expand_y_max + 1):
#             if y >= img_h:
#                 continue
#             pixel = image[y, expand_x_min - step]
#             if not is_background_pixel(pixel, bg_threshold):
#                 has_content = True
#                 break
#         if has_content:
#             expand_x_min -= step
#         else:
#             break

#     # 向右扩展
#     while expand_x_max + step < img_w:
#         has_content = False
#         for y in range(expand_y_min, expand_y_max + 1):
#             if y >= img_h:
#                 continue
#             pixel = image[y, expand_x_max + step]
#             if not is_background_pixel(pixel, bg_threshold):
#                 has_content = True
#                 break
#         if has_content:
#             expand_x_max += step
#         else:
#             break

#     # 向上扩展
#     while expand_y_min - step >= 0:
#         has_content = False
#         for x in range(expand_x_min, expand_x_max + 1):
#             if x >= img_w:
#                 continue
#             pixel = image[expand_y_min - step, x]
#             if not is_background_pixel(pixel, bg_threshold):
#                 has_content = True
#                 break
#         if has_content:
#             expand_y_min -= step
#         else:
#             break

#     # 向下扩展
#     while expand_y_max + step < img_h:
#         has_content = False
#         for x in range(expand_x_min, expand_x_max + 1):
#             if x >= img_w:
#                 continue
#             pixel = image[expand_y_max + step, x]
#             if not is_background_pixel(pixel, bg_threshold):
#                 has_content = True
#                 break
#         if has_content:
#             expand_y_max += step
#         else:
#             break

#     # 4. 收缩阶段：向内逐像素剔除空白
#     # 向左收缩
#     while expand_x_min + step <= expand_x_max:
#         all_background = True
#         for y in range(expand_y_min, expand_y_max + 1):
#             if y >= img_h:
#                 continue
#             pixel = image[y, expand_x_min]
#             if not is_background_pixel(pixel, bg_threshold):
#                 all_background = False
#                 break
#         if all_background:
#             expand_x_min += step
#         else:
#             break

#     # 向右收缩
#     while expand_x_max - step >= expand_x_min:
#         all_background = True
#         for y in range(expand_y_min, expand_y_max + 1):
#             if y >= img_h:
#                 continue
#             pixel = image[y, expand_x_max]
#             if not is_background_pixel(pixel, bg_threshold):
#                 all_background = False
#                 break
#         if all_background:
#             expand_x_max -= step
#         else:
#             break

#     # 向上收缩
#     while expand_y_min + step <= expand_y_max:
#         all_background = True
#         for x in range(expand_x_min, expand_x_max + 1):
#             if x >= img_w:
#                 continue
#             pixel = image[expand_y_min, x]
#             if not is_background_pixel(pixel, bg_threshold):
#                 all_background = False
#                 break
#         if all_background:
#             expand_y_min += step
#         else:
#             break

#     # 向下收缩
#     while expand_y_max - step >= expand_y_min:
#         all_background = True
#         for x in range(expand_x_min, expand_x_max + 1):
#             if x >= img_w:
#                 continue
#             pixel = image[expand_y_max, x]
#             if not is_background_pixel(pixel, bg_threshold):
#                 all_background = False
#                 break
#         if all_background:
#             expand_y_max -= step
#         else:
#             break

#     # ========== 新增：添加padding，保证框体不会太小 ==========
#     # 向外扩展padding像素（可根据需求改为向内收缩，只需把+/-反过来）
#     expand_x_min = max(0, expand_x_min - padding)  # 左边界向左扩展，不小于0
#     expand_x_max = min(
#         img_w - 1, expand_x_max + padding
#     )  # 右边界向右扩展，不超过图像宽度
#     expand_y_min = max(0, expand_y_min - padding)  # 上边界向上扩展，不小于0
#     expand_y_max = min(
#         img_h - 1, expand_y_max + padding
#     )  # 下边界向下扩展，不超过图像高度

#     # 5. 生成调整后的矩形（兼容倾斜场景）
#     adjusted_pts = np.array(
#         [
#             [expand_x_min, expand_y_min],
#             [expand_x_max, expand_y_min],
#             [expand_x_max, expand_y_max],
#             [expand_x_min, expand_y_max],
#         ],
#         dtype=np.float32,
#     )
#     adjusted_min_rect = cv2.minAreaRect(adjusted_pts)
#     adjusted_box_pts = cv2.boxPoints(adjusted_min_rect).astype(np.int32)

#     return adjusted_min_rect, adjusted_box_pts


# # ------------------- 测试示例 -------------------
# if __name__ == "__main__":
#     # 读取图像（替换为你的图片路径）
#     img_path = "b_0001.jpg"
#     image = cv2.imread(img_path)
#     if image is None:
#         raise ValueError("无法读取图像，请检查路径是否正确")

#     # 模拟人工标记的倾斜bbox（4个点，示例坐标）
#     # 格式：[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
#     original_bbox = [(363, 160), (316, 160), (316, 2), (363, 2)]

#     # 调整bbox（新增padding参数，设为10像素，可根据需求调整）
#     adjusted_rect, adjusted_pts = adjust_bbox_to_content(
#         image=image,
#         bbox_points=original_bbox,
#         bg_threshold=240,  # 背景阈值，可根据实际调整
#         step=1,  # 逐像素调整
#         padding=10,  # 新增：调整后添加的padding，避免框体太小
#     )

#     # 可视化结果（绘制原始和调整后的bbox）
#     # 绘制原始bbox（红色）
#     cv2.polylines(
#         image,
#         [np.array(original_bbox, np.int32)],
#         isClosed=True,
#         color=(0, 0, 255),
#         thickness=2,
#     )
#     # 绘制调整后的bbox（绿色）
#     cv2.polylines(
#         image, [adjusted_pts], isClosed=True, color=(0, 255, 0), thickness=2
#     )

#     # 保存/显示结果
#     cv2.imwrite("adjusted_bbox.jpg", image)

#     # 输出调整后的bbox参数
#     print("调整前的外接矩形参数：", original_bbox)
#     print("调整后的bbox最小外接矩形参数：", adjusted_rect)
#     print("调整后的bbox四个顶点坐标：", adjusted_pts)
