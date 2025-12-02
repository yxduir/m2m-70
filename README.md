# MCAT Models

## Language Support
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
[Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | [Adapter](https://huggingface.co/yxdu/mcat-large) | [Gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) 
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

## Train
Please refer to [ours previous work](https://github.com/yxduir/LLM-SRT).

##  Citation
You can refer to the paper for more results. 
```
@misc{du2025mcatscalingmanytomanyspeechtotext,
      title={MCAT: Scaling Many-to-Many Speech-to-Text Translation with MLLMs to 70 Languages}, 
      author={Yexing Du and Kaiyuan Liu and Youcheng Pan and Bo Yang and Keqi Deng and Xie Chen and Yang Xiang and Ming Liu and Bin Qin and YaoWei Wang},
      year={2025},
      eprint={2512.01512},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.01512}, 
}

@article{du2025speech2text,  
  title     = {Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning},
  author    = {Du, Yexing and Pan, Youcheng and Ma, Ziyang and Yang, Bo and Yang, Yifang and Deng, Keqi and Chen, Xie and Xiang, Yang and Liu, Ming and Qin, Bing},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)},
  year      = {2025},
}

```