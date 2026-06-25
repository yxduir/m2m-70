# MCAT: Speech-to-Text Translation for 70 Languages

[![arXiv](https://img.shields.io/badge/arXiv-2512.01512-b31b1b)](https://arxiv.org/abs/2512.01512v1)
[![License](https://img.shields.io/badge/License-CC_BY--NC--SA--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**MCAT (v2.0)** — Speech-to-Text Translation (S2TT) supporting **70 languages** and **4,830 translation directions** (70 × 69 language pairs).

> 📄 [MCAT: Scaling Many-to-Many Speech-to-Text Translation With MLLMs to 70 Languages](https://arxiv.org/abs/2512.01512v1) — IEEE TASLP 2026

<details>
<summary><b>Supported 70 languages</b></summary>

Afrikaans (afr), Amharic (amh), Arabic (ara), Assamese (asm), Azerbaijani (azj), Belarusian (bel), Bengali (ben), Bosnian (bos), Bulgarian (bul), Catalan (cat), Czech (ces), Chinese (cmn), Welsh (cym), Danish (dan), German (deu), Greek (ell), English (eng), Estonian (est), Persian (fas), Finnish (fin), French (fra), Galician (glg), Gujarati (guj), Hebrew (heb), Hindi (hin), Croatian (hrv), Hungarian (hun), Armenian (hye), Indonesian (ind), Icelandic (isl), Italian (ita), Javanese (jav), Japanese (jpn), Kannada (kan), Georgian (kat), Kazakh (kaz), Khmer (khm), Kyrgyz (kir), Korean (kor), Lao (lao), Latvian (lav), Lithuanian (lit), Malayalam (mal), Macedonian (mkd), Malay (msa), Burmese (mya), Dutch (nld), Norwegian (nob), Nepali (npi), Punjabi (pan), Polish (pol), Portuguese (por), Romanian (ron), Russian (rus), Slovak (slk), Slovenian (slv), Spanish (spa), Serbian (srp), Swedish (swe), Swahili (swh), Tamil (tam), Telugu (tel), Tagalog (tgl), Thai (tha), Turkish (tur), Ukrainian (ukr), Urdu (urd), Uzbek (uzb), Vietnamese (vie), Cantonese (yue)

</details>

## Installation
```
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/yxduir/m2m-70
cd m2m-70
sudo apt update && sudo apt install ffmpeg -y
uv venv --python 3.10
source .venv/bin/activate

cd SLAM-LLM
uv pip install -r requirements.txt
uv pip install -e .
cd ..
```

## Download Model

### Option A: MCAT-Large (27B) — full 70 languages

| Component | Model |
|-----------|-------|
| Encoder | [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) |
| Adapter | [yxdu/mcat-large](https://huggingface.co/yxdu/mcat-large) |
| LLM | [Gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) |

Requires GPU with 80 GB VRAM (BF16) — Storage: ~96 GB in total.

```
cd models/

# Total 75G of storage space for 27B models
hf download yxdu/mcat-large --local-dir mcat-large
hf download openai/whisper-large-v3 --local-dir whisper-large-v3
# Access to the Gemma-3 models is required before downloading.
hf download google/gemma-3-27b-it --local-dir gemma-3-27b-it

cd ..
```

### Option B: MCAT-Small (9B) — 28 languages

| Component | Model |
|-----------|-------|
| Encoder | [Whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) |
| Adapter | [yxdu/mcat-small](https://huggingface.co/yxdu/mcat-small) |
| LLM | [GemmaX2-28-9B-v0.1](https://huggingface.co/ModelSpace/GemmaX2-28-9B-v0.1) |

Requires GPU with 24 GB VRAM (BF16) — Storage: ~43 GB in total.

```
cd models/

# Total 43G of storage space for 9B models
hf download yxdu/mcat-small --local-dir mcat-small
hf download openai/whisper-large-v3 --local-dir whisper-large-v3
hf download ModelSpace/GemmaX2-28-9B-v0.1 --local-dir GemmaX2-28-9B-v0.1

cd ..
```

### Evaluation Model

```
# Total 2G of storage space for eval model
cd models/
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


**Option A** — 70 languages, 4,830 directions (requires GPU with 80 GB VRAM, BF16):
```bash
bash scripts/infer_demo_large_27b.sh
```

**Option B** — 28 languages, 756 directions (requires GPU with 24 GB VRAM, BF16):
```bash
bash scripts/infer_demo_small_9b.sh
```

## COMET Eval
```
cd eval
python test_metric_n.py
```

## Train
Please refer to [our previous work](https://github.com/yxduir/LLM-SRT).

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