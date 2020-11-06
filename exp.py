import numpy as np
import multiprocessing
from uxils.file_system import glob_videos, glob_audio
from uxils.image.draw import draw_points
from uxils.image.vis import show_image, show_images_hstack
from uxils.timer import Timer
from uxils.video.io import read_video_av, read_video_cv2, read_video_av_generator, read_video_cv2_generator
from uxils.db.pickle_db import read_pkll_generator
# from uxils.video.vis import show_frames
from app import VideoLandmarkExtractor
from uxils.functools_ext import ignore_exceptions


in_filename = 'data/2020-1/val/48960-4-085-w-67-036-dis-sad-dis.mp4'
# paths = glob_videos("data/2020-1/val")
# paths = glob_videos("face_images")
for path in glob_audio("data/audio"):
    print(path)
    qwe


ex = VideoLandmarkExtractor("face_images", max_size=400)

for path in paths:
    with Timer():
        r = read_video_av(path)

    show_images_hstack(r)


# landmarks = read_pkll_generator("landmarks_val.pkll")
print(len(list(landmarks)))

for path, predictions in landmarks:

    with Timer():
        frames = read_video_av(path)

    print(len(frames), len(predictions))
    for l_per_i, image in zip(predictions, frames):
        for l_per_box in l_per_i:
            draw_points(image, l_per_box)

    show_frames(frames)
        # show_image(image, wait=False)
