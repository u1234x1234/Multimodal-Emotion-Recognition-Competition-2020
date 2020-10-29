import cv2
import numpy as np
import multiprocessing
from uxils.file_system import glob_videos
from uxils.image.draw import draw_points
from uxils.image.vis import show_image
from uxils.profiling import Profiler
from uxils.timer import Timer
from uxils.video.io import read_video_av
from uxils.db.pickle_db import read_pkll_generator
from uxils.video.vis import show_frames


in_filename = 'data/2020-1/val/48960-4-085-w-67-036-dis-sad-dis.mp4'
paths = glob_videos("data/2020-1/val")


def func(path):
    frames = read_video_av(path)
    return len(frames)


with multiprocessing.Pool(40) as pool:
    pool.map(func, paths)

qwe

# landmarks = read_pkll_generator("landmarks_val.pkll")
# print(len(list(landmarks)))

for path, predictions in landmarks:

    with Timer():
        frames = read_video_av(path)

    print(len(frames), len(predictions))
    for l_per_i, image in zip(predictions, frames):
        for l_per_box in l_per_i:
            draw_points(image, l_per_box)

    show_frames(frames)
        # show_image(image, wait=False)
