# MCAT Models

## 70 Languages
<img src="SLAM-LLM/examples/st_covost2/image/70_language.png" alt="Photo" style="width:50%;">


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
cd ..
```

## Download Model 
Encoder | Adapter | LLM 
|---|---|---
[whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | [Adapter](https://huggingface.co/yxdu/mcat-large) | [Gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) 
```
cd models/
# Total 150G of storage space for models
git lfs clone https://huggingface.co/openai/whisper-large-v3
git lfs clone https://huggingface.co/yxdu/mcat-large
# Access to the Gemma models is required before using git lfs.
git lfs clone https://huggingface.co/google/gemma-3-27b-it
cd ..
```


## Infer Demo
This is a demo inference script, covering translation between 70 languages, with a total of 70×69=4,830 directions.
This demo downloads the 9 GB dataset from HuggingFace.
It requires GPUs with 80GB VRAM, with support for BF16 only.
```
bash scripts/infer_demo.sh
```
