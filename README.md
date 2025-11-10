
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

```
## Download Model 
Encoder | Adapter | LLM 
|---|---|---
[whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | [Adapter](https://huggingface.co/yxdu/srt-large) | [Gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) 
```
cd m2m-70/models/
git lfs clone https://huggingface.co/openai/whisper-large-v3
git lfs clone https://huggingface.co/yxdu/srt-large
git lfs clone https://huggingface.co/google/gemma-3-27b-it
```


## Infer Demo
This is an automatic inference script for the fleurs dataset from English (eng) to Chinese (cmn).
```
bash examples/st_covost2/scripts/infer_hf.sh
```

## Train Dataset
If you want to train your own model, you can download the following datasets.
```
[Common Voice](https://commonvoice.mozilla.org/en/datasets)
[Fleurs](https://huggingface.co/datasets/google/fleurs)
```



## Data preparation
You need to prepare the data jsonl in this format.  
| audio      | source           | prompt                     | gt            |
|------------|------------------|----------------------------|---------------|
| audio_path | `{dataset}_{src}_{tgt}` | `<\|{src}\|><\|{tgt}\|>`| `transcription{prompt}translation` |
```
{"audio": "eng/test/139.wav", "source": "fleurs_eng_cmn", "prompt": "<|eng|><|cmn|>", "gt": "They have feet with scales and claws, they lay eggs, and they walk on their two back legs like a T-Rex.<|eng|><|cmn|>它们脚上有鳞片和爪子，会产卵，还像霸王龙一样用两条后腿走路。"}
{"audio": "deu/test/0.wav", "source": "fleurs_deu_ara", "prompt": "<|deu|><|ara|>", "gt": "Für die besten Aussichten auf Hongkong sollten Sie die Insel verlassen und zum gegenüberliegenden Ufer von Kowloon fahren.<|deu|><|ara|>لكي تحظى بأفضل المشاهد لهونج كونج، غادر الجزيرة واتجه إلى واجهة كولون البحرية في الجهة المقابلة."}
{"audio": "jpn/test/485.wav", "source": "fleurs_jpn_ita", "prompt": "<|jpn|><|ita|>", "gt": "これらの結晶の組成は、赤外分光法（FTIR）で比較すると、患部のペットの尿中に見られるものと一致します。<|jpn|><|ita|>Al confronto mediante spettroscopia infrarossa (FT-IR), la composizione di questi cristalli corrisponde a quella individuata nell'urina degli animali da compagnia che ne sono colpiti."}
```
## Training and Inference
You can use the following scripts to perform training and inference separately. 
For all.sh, you can modify the training task based on the 'mode' keyword: asr, smt, srt.
```
#train
bash examples/st_covost2/scripts/all.sh


#infer
bash examples/st_covost2/scripts/infer_all.sh
bash examples/st_covost2/scripts/infer_hf.sh
```



##  Citation
You can refer to the paper for more results. 
```
@article{du2025speech2text,  
  title     = {Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning},
  author    = {Du, Yexing and Pan, Youcheng and Ma, Ziyang and Yang, Bo and Yang, Yifang and Deng, Keqi and Chen, Xie and Xiang, Yang and Liu, Ming and Qin, Bing},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)},
  year      = {2025},
}
```