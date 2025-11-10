
This project is a subproject of https://github.com/X-LANCE/SLAM-LLM.  

# SRT-Large

## 70 Languages
<img src="SLAM-LLM/examples/st_covost2/image/70_language.png" alt="Photo" style="width:75%;">


## Installation
```
conda create -n m2m-70 python=3.10
conda activate m2m-70

git clone https://github.com/yxduir/m2m-70
cd m2m-70/SLAM-LLM

pip install -e .
sudo apt install ffmpeg
pip install -r requirements.txt


## Infer Demo
This is an automatic inference script for the fleurs dataset from 70 to 69, total 4,830 directions.
```
bash scripts/infer_demo.sh
```
