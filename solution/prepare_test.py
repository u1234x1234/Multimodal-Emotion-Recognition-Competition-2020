import contextlib
import multiprocessing
import os
from functools import partial
from glob import glob
from pathlib import Path

import av
import cv2
import dlib
import librosa
import numpy as np
from scipy.io.wavfile import write


def provide_dir_tree(filepath: str) -> None:
    """Create directory tree if not exists"""
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


@contextlib.contextmanager
def provide_file(path: str, *args, **kwargs):
    """Create a directory tree to make sure the path is accessible"""
    provide_dir_tree(path)

    # It is faster to remove file before overwriting
    if (args and args[0] == "wb") or kwargs.get("mode") == "wb":
        if os.path.exists(path):
            os.remove(path)

    with open(path, *args, **kwargs) as out_file:
        yield out_file


class Audio:
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate

    def __str__(self):
        return (
            f"Audio: {self.data.shape} SR: {self.sample_rate}, Sec: {self.n_seconds()}"
        )

    def n_seconds(self):
        return self.data.shape[0] / self.sample_rate


def read_audio(path, sr=None, resampling_type=0) -> Audio:
    """
    Args:
        resampling: int 0=best, 3=fast
    """
    RESAMPLING_TYPES = {
        0: "kaiser_best",
        1: "kaiser_fast",
        2: "scipy",  # Same as fft ?
        3: "polyphase",
    }
    audio, sr = librosa.load(path, sr=sr, res_type=RESAMPLING_TYPES[resampling_type])
    return Audio(audio, sr)


def write_audio(path, data, sr=None, out_format="wav"):
    assert out_format == "wav"
    if isinstance(data, Audio):
        data, sr = data.data, data.sample_rate
    else:
        assert sr is not None

    with provide_file(path, "wb") as out_file:
        write(out_file, sr, data)


def resample_file(path, out_path, sr=16000):
    a = read_audio(path, sr=sr)
    write_audio(out_path, a)


def extract_audio(path, out_path, sample_rate=16000):

    input_container = av.open(path)
    input_stream = input_container.streams.get(audio=0)[0]

    output_container = av.open(out_path, "w")
    output_stream = output_container.add_stream("pcm_s16le")

    for frame in input_container.decode(input_stream):
        frame.pts = None
        for packet in output_stream.encode(frame):
            output_container.mux(packet)

    for packet in output_stream.encode(None):
        output_container.mux(packet)

    output_container.close()

    if sample_rate is not None:
        resample_file(out_path, out_path, sr=sample_rate)


def rebase_path(path, root_dir, out_dir, ext=None):
    new_path = f"{out_dir}/{os.path.relpath(path, root_dir)}"
    if ext is not None:
        bn, prev_ext = os.path.splitext(new_path)
        assert prev_ext
        new_path = f"{bn}.{ext}"

    return new_path


def make_identical_dir_tree(input_path, output_path):
    input_path = str(input_path)
    output_path = str(output_path)

    dirs = [p for p in Path(input_path).rglob("**/*") if p.is_dir()]

    to_create = []
    for p in dirs:
        sub = p.relative_to(input_path)
        out = os.path.join(output_path, sub)
        to_create.append(out)

    for p in to_create:
        os.makedirs(p, exist_ok=True)


def extract_audio_dir(root_dir, out_dir, sample_rate=16000):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    make_identical_dir_tree(root_dir, out_dir)
    paths = glob(root_dir + "/**/*.mp4", recursive=True)
    tasks = [
        (p, rebase_path(p, root_dir, out_dir, ext="wav"), sample_rate) for p in paths
    ]
    print(f"number of tasks: {len(tasks)}")

    with multiprocessing.Pool(10) as pool:
        pool.starmap(extract_audio, tasks)


class FACE_LANDMARKS_68:
    LEFT_EYE_INDICES = list(range(36, 42))
    RIGHT_EYE_INDICES = list(range(42, 48))
    MOUTH_INDICES = list(range(48, 60))
    LIPS_INDICES = list(range(61, 68))


def mean_position(landmarks, indices):
    return tuple(np.mean([landmarks[idx] for idx in indices], axis=0).astype(np.int))


def crop_face_from_landmarks(image, landmarks, face_size=200):
    landmarks = np.array(landmarks)
    assert landmarks.shape == (68, 2)

    left_eye_dst = (face_size * 0.25, face_size * 0.25)
    right_eye_dst = (face_size * 0.75, face_size * 0.25)
    mouth_dst = (face_size * 0.5, face_size * 0.75)

    left_eye_src = mean_position(landmarks, FACE_LANDMARKS_68.LEFT_EYE_INDICES)
    right_eye_src = mean_position(landmarks, FACE_LANDMARKS_68.RIGHT_EYE_INDICES)
    mouth_src = mean_position(landmarks, FACE_LANDMARKS_68.MOUTH_INDICES)

    src_pts = np.float32([left_eye_src, right_eye_src, mouth_src])
    dst_pts = np.float32([left_eye_dst, right_eye_dst, mouth_dst])

    M = cv2.getAffineTransform(src_pts, dst_pts)
    face_image = cv2.warpAffine(image, M, (face_size, face_size))

    return face_image, M


def read_video_av_generator(
    path, fmt="bgr24", postprocessing=None, threads=False, sparse=None
):
    """

    Args:
        sparse: int
            10 - linspace(0, n_frames, 10)
            0.5 - a half of frames
    """
    container = av.open(path)
    if threads:
        container.streams.video[0].thread_type = "AUTO"
    n_frames = container.streams.video[0].frames

    indices_to_read = []
    if sparse is not None:
        if isinstance(sparse, float):
            assert 0 < sparse < 1
            sparse = int(sparse * n_frames)

        assert 1 <= sparse <= n_frames
        indices_to_read = np.linspace(0, n_frames - 1, sparse).astype(np.int)

    for idx, frame in enumerate(container.decode(video=0)):
        if sparse is not None:
            if idx not in indices_to_read:
                continue

        frame = frame.to_ndarray(format=fmt)
        if postprocessing is not None:
            frame = postprocessing(frame)

        yield frame


def write_frames_to_file(path, frames, frame_rate=30):
    provide_dir_tree(path)

    fc = cv2.VideoWriter_fourcc(*"mp4v")
    frame = frames[0]
    writer = cv2.VideoWriter(path, fc, frame_rate, (frame.shape[1], frame.shape[0]))

    for frame in frames:
        writer.write(frame)

    writer.release()


def get_transform_rigid(
    image_w: int,
    image_h: int,
    w: int,
    h: int,
    scale="max",
    min_scale_ratio: float = 1.0,
    max_scale_ratio: float = 1.0,
    rotation_angle=0,
    min_shift_ratio: float = 0.0,
    max_shift_ratio: float = 0.0,
):
    if isinstance(scale, str):
        if scale == "max":
            scale = min(h / image_h, w / image_w)
        elif scale == "min":
            scale = max(h / image_h, w / image_w)
        else:
            raise ValueError(f"Unkwnown scale parameter: {scale}")
    scale *= np.random.uniform(min_scale_ratio, max_scale_ratio)

    if rotation_angle is None:
        min_rotate_angle, max_rotate_angle = 0, 0
    elif isinstance(rotation_angle, (tuple, list, np.ndarray)):
        min_rotate_angle, max_rotate_angle = rotation_angle
    elif isinstance(rotation_angle, int):
        min_rotate_angle, max_rotate_angle = rotation_angle, rotation_angle
    else:
        raise ValueError(f"Unknown rotation parameter: {rotation_angle}")

    angle = np.random.randint(min_rotate_angle, max_rotate_angle + 1)
    center = (image_w / 2, image_h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale=scale)

    sx = (w - image_w) / 2
    sy = (h - image_h) / 2
    sx += sx * np.random.uniform(-min_shift_ratio, max_shift_ratio)
    sy += sy * np.random.uniform(-min_shift_ratio, max_shift_ratio)
    M[0][2] += sx
    M[1][2] += sy
    return M, (w, h)


def scale_max(w, h, out_w, out_h):
    "scale to (w, h) with padding & preserving aspect ratio"
    return get_transform_rigid(w, h, w=out_w, h=out_h, scale="max")


def scale_min(w, h, out_w, out_h):
    return get_transform_rigid(w, h, w=out_w, h=out_h, scale="min")


def scale_max_factory(out_size):
    if isinstance(out_size, int):
        out_size = (out_size, out_size)
    assert isinstance(out_size, tuple) and len(out_size) == 2
    return partial(scale_max, out_w=out_size[0], out_h=out_size[1])


def pad_affine(m):
    return np.vstack((m, [0, 0, 1]))


def combine_affine(*matrices):
    """List of 2x3 to 2x3"""

    final = pad_affine(matrices[-1])
    for m in list(reversed(matrices))[1:]:
        final = np.dot(final, pad_affine(m))

    return final[:2]


class PointTransformer:
    def __init__(self, M):
        self.M = M
        self.M_inv = None  # lazy eval

    def transform(self, points):
        points = cv2.transform(np.array([points]), self.M)[0]
        return points

    def inverse_transform(self, points):
        if self.M_inv is None:
            self.M_inv = np.linalg.inv(pad_affine(self.M))[:2]

        points = cv2.transform(np.array([points]), self.M_inv)[0]
        return points


class AffinePipeline:
    def __init__(self, *ops, interpolation=cv2.INTER_LINEAR):
        """
        Example:
            p = AffinePipeline(scale_max_factory(300))
            t_image, p_transformer = p.transform(image)
            bboxes = face_detector.detect(t_image)
            p_transformer.inverse_transform(bboxes)
        """

        self.interpolation = interpolation

        self._operations = []
        for op in ops:
            if callable(op):
                self._operations.append(op)
            else:
                raise ValueError(f"unkwown op: {op}")

    def get_transformed_meta(self, w, h):
        matrices = []
        for op in self._operations:
            M, (w, h) = op(w, h)
            matrices.append(M)

        M = combine_affine(*matrices)
        return M, (w, h)

    def transform(self, image):
        M, (w, h) = self.get_transformed_meta(*image.shape[:2][::-1])
        image = cv2.warpAffine(image, M, (w, h), flags=self.interpolation)
        return image, PointTransformer(M)


class DLibFaceLandmarksExtractor:
    def __init__(
        self,
        landmark_extractor_path="shape_predictor_68_face_landmarks.dat",
        max_size=None,
    ):
        self.detector = dlib.get_frontal_face_detector()

        self.landmark_extactor = dlib.shape_predictor(landmark_extractor_path)
        self.preprocessor = (
            AffinePipeline(scale_max_factory(max_size)) if max_size else None
        )

    def extract_landmarks(self, images):
        """
        for landmarks_per_image in extract_landmarks(images):
            for landmarks in landmarks_per_image:
                assert landmarks.shape == (68, 2)
        """
        detections = []
        for image in images:
            if self.preprocessor is not None:
                image, point_transformer = self.preprocessor.transform(image)

            landmarks_per_image = []
            for box in self.detector(image, 0):  # 0 == no upsampling
                landmarks = self.landmark_extactor(image, box)
                landmarks = np.array(
                    [
                        (landmarks.part(i).x, landmarks.part(i).y)
                        for i in range(landmarks.num_parts)
                    ]
                )

                if self.preprocessor is not None:
                    landmarks = point_transformer.inverse_transform(landmarks)

                landmarks_per_image.append(landmarks)

            detections.append(landmarks_per_image)

        return detections


class VideoLandmarkExtractor:
    def __init__(self, out_dir, max_size=None):

        self.ex = DLibFaceLandmarksExtractor(max_size=max_size)
        self.out_dir = out_dir

    def extract(self, path):
        try:
            face_images = []
            for frame in read_video_av_generator(path, threads=False, sparse=10):
                landmarks = self.ex.extract_landmarks([frame])[0]

                if landmarks:
                    face_image, _ = crop_face_from_landmarks(frame, landmarks[0])
                    face_images.append(face_image)

            write_frames_to_file(
                f"{self.out_dir}/{os.path.basename(path)}", face_images
            )
        except Exception as e:
            import traceback
            print(e, path)
            traceback.print_exc()


def _worker(paths, out_dir):
    ex = VideoLandmarkExtractor(out_dir=out_dir, max_size=400)
    for idx, p in enumerate(paths):
        ex.extract(p)


def extract_faces(root_dir, out_dir, n_workers=20):
    os.makedirs(out_dir, exist_ok=True)

    paths = glob(f"{root_dir}/**/*.mp4", recursive=True)
    np.random.shuffle(paths)

    worker_pool = []
    for chunk in np.array_split(paths, n_workers):
        p = multiprocessing.Process(target=_worker, args=(chunk, out_dir))
        p.start()
        worker_pool.append(p)

    for p in worker_pool:
        p.join()


if __name__ == "__main__":
    import fire

    fire.Fire()
