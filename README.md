# MCAT Models


**LLM-SRT (v1.0) paper**: [https://arxiv.org/abs/2409.19510](https://arxiv.org/abs/2409.19510) [ACL 2025 Main]; 

**MCAT (v2.0) paper**: [https://arxiv.org/abs/2512.01512v1](https://arxiv.org/abs/2512.01512v1); 

✅ **Current Version MCAT (v2.0)**  
- **Supported 70 Languages**: Afrikaans (afr), Amharic (amh), Arabic (ara), Assamese (asm), Azerbaijani (azj), Belarusian (bel), Bengali (ben), Bosnian (bos), Bulgarian (bul), Catalan (cat), Czech (ces), Chinese (cmn), Welsh (cym), Danish (dan), German (deu), Greek (ell), English (eng), Estonian (est), Persian (fas), Finnish (fin), French (fra), Galician (glg), Gujarati (guj), Hebrew (heb), Hindi (hin), Croatian (hrv), Hungarian (hun), Armenian (hye), Indonesian (ind), Icelandic (isl), Italian (ita), Javanese (jav), Japanese (jpn), Kannada (kan), Georgian (kat), Kazakh (kaz), Khmer (khm), Kyrgyz (kir), Korean (kor), Lao (lao), Latvian (lav), Lithuanian (lit), Malayalam (mal), Macedonian (mkd), Malay (msa), Burmese (mya), Dutch (nld), Norwegian (nob), Nepali (npi), Punjabi (pan), Polish (pol), Portuguese (por), Romanian (ron), Russian (rus), Slovak (slk), Slovenian (slv), Spanish (spa), Serbian (srp), Swedish (swe), Swahili (swh), Tamil (tam), Telugu (tel), Tagalog (tgl), Thai (tha), Turkish (tur), Ukrainian (ukr), Urdu (urd), Uzbek (uzb), Vietnamese (vie), Cantonese (yue)
- **4830 Translation Directions** - Supports all 4830 possible translation directions (70×69 language pairs)

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
Access to the Gemma models is required before downloading.

```
cd models/

# Total 75G of storage space for models
hf download yxdu/mcat-large --local-dir mcat-large
hf download openai/whisper-large-v3 --local-dir whisper-large-v3
hf download google/gemma-3-27b-it --local-dir gemma-3-27b-it

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