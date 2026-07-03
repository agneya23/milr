set -Eeo pipefail
trap 'echo "ERROR: Script failed at line $LINENO"; exit 1' ERR

ENV_PATH="/ivi/zfs/s0/original_homes/ydu/miniconda3/envs/milr_latentseek"

if [ -d "$ENV_PATH" ]; then
    conda env remove -y -p "$ENV_PATH"
fi

source /ivi/zfs/s0/original_homes/ydu/miniconda3/bin/activate
conda create -y -p /ivi/zfs/s0/original_homes/ydu/miniconda3/envs/milr_latentseek python=3.10
conda activate /ivi/zfs/s0/original_homes/ydu/miniconda3/envs/milr_latentseek/
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r /ivi/zfs/s0/original_homes/ydu/PythonWorkSpace/agneya/milr/requirements2.txt

#install Geneval configs
#You may meet package counters, it doesn't matter
pip install -U openmim
mim install mmengine
mim install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

cd /ivi/zfs/s0/original_homes/ydu/PythonWorkSpace/agneya/milr/src/geneval
./evaluation/download_models.sh "object_detector/"

if [ -d "/ivi/zfs/s0/original_homes/ydu/PythonWorkSpace/agneya/milr/src/geneval/mmdetection" ]; then
    rm -rf /ivi/zfs/s0/original_homes/ydu/PythonWorkSpace/agneya/milr/src/geneval/mmdetection
fi

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x

python -m pip install pip==24.0
python -m pip install -v -e . --no-build-isolation

cd /ivi/zfs/s0/original_homes/ydu/PythonWorkSpace/agneya/milr/src/rewards
./evaluation/download_models.sh "object_detector/"