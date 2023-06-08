# README

## Python3 Dependencies

```bash
apt install python3-dev python3-pip
python3 -m pip install -U pip
python3 -m pip install numpy matplotlib seaborn colorama pyyaml

# GPU
python3 -m pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102
# CPU
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu
```

### Create docker container
```bash
export CONTAINER_NAME=cw-event-tools && 
export HOST_DATA_DIR=/data && 
export HOST_SHARED_DIR=~/shared-${CONTAINER_NAME} && 
mkdir -p ${HOST_SHARED_DIR} &&
docker run --runtime=nvidia --gpus all -it \
--env DISPLAY \
--env "QT_X11_NO_MITSHM=1" \
--volume /tmp/.X11-unix:/tmp/.X11-unix \
--volume ${DATA_DIR}:/data \
--volume ${HOST_SHARED_DIR}:/root/shared-${CONTAINER_NAME} \
--workdir=/root/shared-${CONTAINER_NAME} \
--name ${CONTAINER_NAME} \
leechanwoo25/cw-event-tools:v1.0
```
