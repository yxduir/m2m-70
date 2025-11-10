
This project is a subproject of https://github.com/X-LANCE/SLAM-LLM.  
/mgData2/yxdu/github/README.md
# SRT-Large

## 70 Languages
<img src="SLAM-LLM/examples/st_covost2/image/70_language.png" alt="Photo" style="width:75%;">


## Installation
```
conda create -n m2m-70 python=3.10
conda activate m2m-70

git clone https://github.com/yxduir/m2m-70
cd m2m-70/SLAM-LLM

sudo apt update
sudo apt install ffmpeg
sudo apt install git-lfs

pip install -r requirements.txt
pip install -e .
```



## Infer Demo
This is a demo inference script for the FLEURS dataset, covering translation between 70 languages, with a total of 70×69=4,830 directions.
This demo will automatically download the model and dataset from Hugging Face, which require approximately 200 GB of storage space. GPUs requires 80 GB of VRAM.
```
bash scripts/infer_demo.sh
```
