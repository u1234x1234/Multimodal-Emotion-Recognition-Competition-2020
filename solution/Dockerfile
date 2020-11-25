FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

LABEL maintainer="u1234x1234@gmail.com"

RUN apt-get update && \
    apt-get install unzip aria2 -y

WORKDIR /home/

ENV CONDA_VERSION Miniconda3-py38_4.8.3-Linux-x86_64.sh
RUN aria2c -x8 https://repo.anaconda.com/miniconda/${CONDA_VERSION} && \
    bash ${CONDA_VERSION} -b && \
    rm ${CONDA_VERSION}
ENV PATH /root/miniconda3/bin:$PATH

RUN pip install \
    scikit-learn==0.22.0 \
    torch==1.6.0 \
    pandas==1.0.0 \
    timm==0.3.1 \
    tqdm==4.46.0 \
    opencv-python==4.4.0.42 \
    fire==0.3.1 \
    librosa==0.7.2 \
    av==8.0.2

RUN apt-get update && apt-get install cmake gcc g++ -y
RUN pip install dlib==19.21.0

RUN aria2c -x8 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
RUN bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2

RUN apt-get update && apt-get install -y libsndfile1-dev libgl1-mesa-dev libglib2.0-0
RUN pip install numba==0.48 sndfile==0.2.0 torchaudio==0.6.0

COPY baseline_lite_ap.model 0b624d_22.pt 1837a9_38.pt f3456b_18.pt /home/
COPY *.py /home/
COPY *.csv /home/

CMD ["/bin/bash"]
