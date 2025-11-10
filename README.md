
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

sudo apt update
sudo apt install ffmpeg
sudo apt install git-lfs

pip install -r requirements.txt
pip install -e .
```

## Download Model 
Encoder | Adapter | LLM 
|---|---|---
[whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | [Adapter](https://huggingface.co/yxdu/srt-large) | [Gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) 
```
cd models/
# Total 200G of storage space for models
git lfs clone https://huggingface.co/openai/whisper-large-v3
git lfs clone https://huggingface.co/yxdu/srt-large
# Access to the Gemma models is required before using git lfs.
git lfs clone https://huggingface.co/google/gemma-3-27b-it
```


## Infer Demo
This is a demo inference script for the FLEURS dataset, covering translation between 70 languages, with a total of 70×69=4,830 directions.
This demo automatically downloads the 5 GB dataset from Hugging Face.
GPUs requires 80 GB of VRAM.
```
bash scripts/infer_demo.sh
```
