# -*- coding: utf-8 -*-
"""
工具函数
"""
import os
import time
from typing import Literal

import cv2


def color_print(text, color: Literal['red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white'] = 'green'):
    """
    彩色打印
    :param text: 文本
    :param color: 颜色
    :return:
    """
    color_dict = {
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m'
    }
    print(f"{color_dict[color]}{text}\033[0m")


def print_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        color_print(f"[INFO] {func.__name__} took {end - start} seconds", color='yellow')
        return result

    return wrapper


# 从视频中截取图片
def get_images_from_video(video_path, output_folder, start_time=None, end_time=None, interval=50):
    """
    从视频中截取图片
    :param video_path: 视频路径
    :param output_folder: 输出文件夹
    :param start_time: 开始时间
    :param end_time: 结束时间
    :param interval: 间隔帧数
    :return:
    """
    print(f"[INFO] extracting images from {video_path}")
    p0 = (40, 150)
    p1 = (35 + 640, 140 + 480)
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    # 获取帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / interval)

    count = 0
    saved_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0 and ((
            start_time is None or start_time * fps <= count) or (
            end_time is None or end_time * fps >= count
        )):
            # 截取大小到p0至p1的区域
            frame = frame[p0[1]:p1[1], p0[0]:p1[0]]
            cv2.imwrite(f"{output_folder}/{video_name.split('.')[0]}_{count}.jpg", frame)
            saved_count += 1
            color_print(f"[INFO] processed {saved_count}/{total}", color='yellow')
        count += 1
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] {saved_count} images extracted from {video_path}!")
