# MCAT Models


**MCAT (v2.0) paper**: [https://arxiv.org/abs/2512.01512v1](https://arxiv.org/abs/2512.01512v1) [IEEE TASLP 2026]; 

✅ **Current Version MCAT (v2.0)**  
- **Task**: Speech-to-Text Translation (S2TT)
- **Supported 70 Languages**: Afrikaans (afr), Amharic (amh), Arabic (ara), Assamese (asm), Azerbaijani (azj), Belarusian (bel), Bengali (ben), Bosnian (bos), Bulgarian (bul), Catalan (cat), Czech (ces), Chinese (cmn), Welsh (cym), Danish (dan), German (deu), Greek (ell), English (eng), Estonian (est), Persian (fas), Finnish (fin), French (fra), Galician (glg), Gujarati (guj), Hebrew (heb), Hindi (hin), Croatian (hrv), Hungarian (hun), Armenian (hye), Indonesian (ind), Icelandic (isl), Italian (ita), Javanese (jav), Japanese (jpn), Kannada (kan), Georgian (kat), Kazakh (kaz), Khmer (khm), Kyrgyz (kir), Korean (kor), Lao (lao), Latvian (lav), Lithuanian (lit), Malayalam (mal), Macedonian (mkd), Malay (msa), Burmese (mya), Dutch (nld), Norwegian (nob), Nepali (npi), Punjabi (pan), Polish (pol), Portuguese (por), Romanian (ron), Russian (rus), Slovak (slk), Slovenian (slv), Spanish (spa), Serbian (srp), Swedish (swe), Swahili (swh), Tamil (tam), Telugu (tel), Tagalog (tgl), Thai (tha), Turkish (tur), Ukrainian (ukr), Urdu (urd), Uzbek (uzb), Vietnamese (vie), Cantonese (yue)
- **4830 Translation Directions** - Supports all 4830 possible translation directions (70×69 language pairs)
- **License**: CC BY-NC-SA-4.0

## Installation
```
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/yxduir/m2m-70
cd m2m-70
uv venv --python 3.10
source .venv/bin/activate

cd SLAM-LLM
sudo apt update
sudo apt install ffmpeg
uv pip install -r requirements.txt
uv pip install -e .
cd ..
```

## Download Model 
Encoder | Adapter | LLM 
|---|---|---
[Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | [Adapter](https://huggingface.co/yxdu/mcat-large) | [Gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) 


```
# Total 96G of storage space for all models
cd models/

# Total 75G of storage space for 27B models
hf download yxdu/mcat-large --local-dir mcat-large
hf download openai/whisper-large-v3 --local-dir whisper-large-v3
# Access to the Gemma-3 models is required before downloading.
hf download google/gemma-3-27b-it --local-dir gemma-3-27b-it

# Total 43G of storage space for 9B models
hf download yxdu/mcat-small --local-dir mcat-small
hf download openai/whisper-large-v3 --local-dir whisper-large-v3
hf download ModelSpace/GemmaX2-28-9B-v0.1 --local-dir GemmaX2-28-9B-v0.1

#Total 2G of storage space for eval model
hf download Unbabel/wmt22-comet-da --local-dir wmt22-comet-da

cd ..
```

## Demo Data Download
```
# Total 6G of storage space for demo data
cd data
bash download_demo_data.sh
cd ..
```

## Infer Demo
```
Modify train_config.val_batch_size to change the batch size. The default value is kept low to avoid OOM.

This is a demo for 70 languages, with a total of 4,830 directions.
It requires GPUs with 80GB VRAM (only support BF16).
bash scripts/infer_demo_large_27b.sh

This is a demo for 28 languages, with a total of 756 directions.
It requires GPUs with 24GB VRAM (only support BF16).
bash scripts/infer_demo_small_9b.sh
```

## COMET Eval
```
cd eval
python test_metric_n.py
```

## Train
Please refer to [ours previous work](https://github.com/yxduir/LLM-SRT).

##  Citation
```
@ARTICLE{11481964,
  author={Du, Yexing and Liu, Kaiyuan and Pan, Youcheng and Yang, Bo and Deng, Keqi and Chen, Xie and Xiang, Yang and Liu, Ming and Qin, Bing and Wang, YaoWei},
  journal={IEEE Transactions on Audio, Speech and Language Processing}, 
  title={MCAT: Scaling Many-to-Many Speech-to-Text Translation With MLLMs to 70 Languages}, 
  year={2026},
  volume={34},
  number={},
  pages={2876-2887},
  keywords={Feeds;Radio broadcasting;Frequency modulation;LoRa;Electronic mail;Video games;Videos;Internet;Video equipment;Modulation;Speech-to-text translation;multimodal large language models;curriculum learning},
  doi={10.1109/TASLPRO.2026.3684396}}

@inproceedings{du2025making,
  title={Making llms better many-to-many speech-to-text translators with curriculum learning},
  author={Du, Yexing and Pan, Youcheng and Ma, Ziyang and Yang, Bo and Yang, Yifan and Deng, Keqi and Chen, Xie and Xiang, Yang and Liu, Ming and Qin, Bing},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={12466--12478},
  year={2025}
}
```