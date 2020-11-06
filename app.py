import os

import fire
import numpy as np
from uxils.file_system import glob_videos
from uxils.image.face.alignment import crop_face_from_landmarks
from uxils.timer import Timer
from uxils.video.io import read_video_av_generator, write_frames_to_file


class VideoLandmarkExtractor:
    def __init__(self, out_dir, max_size=None):
        from uxils.image.face.alignment import DLibFaceLandmarksExtractor

        self.ex = DLibFaceLandmarksExtractor(max_size=max_size)
        self.out_dir = out_dir

    def extract(self, path):
        try:
            face_images = []
            for frame in read_video_av_generator(path, threads=False, sparse=10):
                landmarks = self.ex.extract_landmarks(frame)

                if landmarks:
                    face_image, _ = crop_face_from_landmarks(frame, landmarks[0])
                    face_images.append(face_image)

            write_frames_to_file(
                f"{self.out_dir}/{os.path.basename(path)}", face_images
            )
        except Exception as e:
            print(path, e)


def extract_landmarks():
    from uxils.multiprocessing_ext.map import map_class

    paths = glob_videos("data/2020-1/test1")
    np.random.shuffle(paths)
    print(len(paths))

    map_class(
        (VideoLandmarkExtractor, "extract", dict(out_dir="face_images", max_size=400)),
        paths,
        n_workers=40,
    )


if __name__ == "__main__":
    fire.Fire()
