# MERC 2020

# Description

The following files contains the final predictions on each of test sets and was submitted to EvalAI.

* sub_test1.csv - 0.6140161725067386 EvalAI score
* sub_test2.csv - 0.6418230563002681 EvalAI score
* sub_test3.csv - 0.6039763567974208 EvalAI score

By the end of this guide, these files will reproduced.

# Sequence of steps to reproduce the prediction results

All the source codes are packaged in Docker to make it reproducible in different environments. Versions of libraries/packages are specified in Dockerfile. Please refer to the [Dockerfile](./Dockerfile) for details.

Hardware requirements: 1 GPU with cuda support; RAM: 24GB. N_CPUS: more is better.

1. Install docker with [cuda support](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

2. cd to the directory with this README

3. Build the docker image:
```bash
docker build -t merc
```
During this steps all required libraries will be downloaded.

4. Run the built docker image with mounting of directories with test images:
```bash
docker run --gpus=0 --shm-size=24G -it -v "$(pwd)/data/2020-1/test1:/test1" -v "$(pwd)/data/2020-2/test2:/test2" -v "$(pwd)/data/2020-3/test3:/test3" merc bash
```

All the following steps are needed to be run in Docker.

5. Extract audio from videos. Takes up to 10 minutes on the machine with 10 CPUS.

```bash
python prepare_test.py extract_audio_dir /test1/ audio_test1
python prepare_test.py extract_audio_dir /test2/ audio_test2
python prepare_test.py extract_audio_dir /test3/ audio_test3
```

6. Extract face images from videos. Takes up to 1 hour.

```bash
python prepare_test.py extract_faces /test1/ faces_test1
python prepare_test.py extract_faces /test2/ faces_test2
python prepare_test.py extract_faces /test3/ faces_test3
```

7. Given the extracted faces and audio, make predictions:

```bash
python predict_test.py /test1 audio_test1 faces_test1 predictions_test1.csv
python predict_test.py /test2 audio_test2 faces_test2 predictions_test2.csv
python predict_test.py /test3 audio_test3 faces_test3 predictions_test3.csv
```

The predictions will be in files:

* predictions_test1.csv
* predictions_test2.csv
* predictions_test3.csv


8. Make sure the generated predictions match the predictions submitted to the EvalAI platform:

```bash
python check_is_same.py sub_test1.csv predictions_test1.csv
python check_is_same.py sub_test2.csv predictions_test2.csv
python check_is_same.py sub_test3.csv predictions_test3.csv
```
