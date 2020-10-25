from uxils.video.io import read_video_av, read_video_cv2
from uxils.timer import Timer
from uxils.file_system import glob_videos
from uxils.image.vis import show_image


for path in glob_videos("data/2020-1/val"):
    print(path)
    with Timer():
        frames = read_video_av(path)
    for frame in frames:
        show_image(frame)
