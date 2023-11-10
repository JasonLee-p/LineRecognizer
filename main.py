# -*- coding: utf-8 -*-
"""
LSD检测直线
"""
import os
from utils import print_time, color_print, get_images_from_video

import numpy as np
from matplotlib import pyplot as plt
import cv2


plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
OWN_PATH = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(OWN_PATH, 'sources', 'images')
OUTPUT_PATH = os.path.join(OWN_PATH, 'output')


BG_COLOUR = "#ffffff"
FONT_SIZE10 = 10
FONT_SIZE16 = 16


@print_time
def find_lines(img_path, output_folder):
    """
    检测直线并绘制
    :param img_path: 图片路径
    :param output_folder: 保存文件夹
    :return:
    """
    img_name = os.path.basename(img_path)
    suffix = img_name.split('.')[-1]
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 创建一个LSD对象
    lsd = cv2.createLineSegmentDetector(0)
    # 执行检测结果，过滤长度小于50的直线
    dlines = lsd.detect(gray)
    # 绘制检测结果
    hor_lines = []
    ver_lines = []
    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        line_len = np.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        if x1 == x0:
            k = 1000
        else:
            k = (y1 - y0) / (x1 - x0)
        if line_len > 5 and (k > 1 or k < -1):
            ver_lines.append(dline)
            cv2.line(image, (x0, y0), (x1, y1), (255, 0, 0), 2)
        elif line_len > 25 and -1 < k < 1:
            hor_lines.append(dline)
            cv2.line(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
    # 保存结果
    cv2.imwrite(os.path.join(output_folder, f"{img_name}"), image)
    return ver_lines, hor_lines


def store_images_from_videos():
    # 将视频中的图片保存到文件夹
    for video_name in os.listdir(os.path.join(OWN_PATH, 'sources', 'videos')):
        get_images_from_video(os.path.join(OWN_PATH, 'sources', 'videos', video_name), INPUT_PATH)


def extract_lines():
    # 遍历文件夹路径下的所有图片（不包括子文件夹）
    total = 0
    for _ in os.listdir(INPUT_PATH):
        total += 1
    # 清空输出文件夹
    try:
        os.mkdir(OUTPUT_PATH)
    except FileExistsError:
        for file in os.listdir(OUTPUT_PATH):
            os.remove(os.path.join(OUTPUT_PATH, file))
    except PermissionError:
        color_print(f"[ERROR] permission denied when creating {OUTPUT_PATH}", color='red')
        exit(-1)
    all_ver_lines = []
    all_hor_lines = []
    for i, image_name in enumerate(os.listdir(INPUT_PATH)):
        image_path = os.path.join(INPUT_PATH, image_name)
        ver_lines, hor_lines = find_lines(image_path, OUTPUT_PATH)
        # all_ver_lines.extend(ver_lines)
        # all_hor_lines.extend(hor_lines)
        all_hor_lines = hor_lines
        all_ver_lines = ver_lines
        print(f"[INFO] processed {i + 1}/{total}")
    return all_ver_lines, all_hor_lines


class ThreeDPlot:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        # 在右侧添加一个2d对照子图，用于显示其他数据
        # self.ax2d = self.fig.add_subplot(122)
        # plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    def scatter_plot(self, x, y, z, xlabel, ylabel, zlabel, color='r', marker='o'):
        self.ax.scatter(x, y, z, c=color, marker=marker)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.set_zlabel(zlabel)

    def plot_lines_in_2d(self, ks, bs, x_min, x_max, color='r'):
        """
        在2d图中绘制直线
        :param ks: 斜率
        :param bs: 截距
        :param x_min: x的最小值
        :param x_max: x的最大值
        :param color: 颜色
        :return:
        """
        ...

    def surface_plot(self, x, y, z):
        self.ax.plot_surface(x, y, z, cmap='viridis')
        self.ax.set_xlabel('X Axis')
        self.ax.set_ylabel('Y Axis')
        self.ax.set_zlabel('Z Axis')


def plot_line_data(lines):
    """
    绘制直线数据
    :param lines: 直线数据
    :return:
    """
    # 绘制可交互的三维图像
    three_d_plot = ThreeDPlot()
    xlabel = '截距'
    ylabel = '斜率'
    zlabel = '长度'
    xs = []
    ys = []
    zs = []
    for line in lines:
        if line[0][2] == line[0][0]:
            continue
        k = (line[0][3] - line[0][1]) / (line[0][2] - line[0][0])
        b = line[0][1] - k * line[0][0]
        line_len = np.sqrt((line[0][0] - line[0][2]) ** 2 + (line[0][1] - line[0][3]) ** 2)
        xs.append(b)
        ys.append(k)
        zs.append(line_len)
    three_d_plot.scatter_plot(xs, ys, zs, xlabel, ylabel, zlabel)
    plt.show()


if __name__ == '__main__':
    # store_images_from_videos()
    verL, horL = extract_lines()
    plot_line_data(horL)
