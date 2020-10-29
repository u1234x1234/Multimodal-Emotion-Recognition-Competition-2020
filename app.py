import fire
from uxils.video.io import read_video_av


class VideoLandmarkExtractor:
    def __init__(self, max_size=None):
        from uxils.image.face.alignment import DLibFaceLandmarksExtractor

        self.ex = DLibFaceLandmarksExtractor(max_size=max_size)

    def extract(self, path):
        frames = read_video_av(path)
        print(len(frames))
        return [1] * len(frames)
        # return self.ex.extract_landmarks(frames)


def extract_landmarks():
    from uxils.file_system import glob_videos
    from uxils.multiprocessing_ext.map import map_class_pickle_db

    paths = glob_videos("data/2020-1/val")

    map_class_pickle_db(
        (VideoLandmarkExtractor, "extract", dict(max_size=400)),
        paths,
        "landmarks_val.pkll",
        batch=False,
        n_workers=40
    )


if __name__ == "__main__":
    fire.Fire()
