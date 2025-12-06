
## Download Model 
Encoder | Adapter | LLM 
|---|---|---
[whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | [Adapter](https://huggingface.co/yxdu/mcat-large) | [Gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) 
Access to the Gemma models is required before downloading.

```
cd models/
hf download yxdu/mcat-large --local-dir mcat-large
hf download openai/whisper-large-v3 --local-dir whisper-large-v3
hf download google/gemma-3-27b-it --local-dir gemma-3-27b-it
```


## Evaluation Model 
```
hf download google/Unbabel/wmt22-cometkiwi-da --local-dir wmt22-cometkiwi-da
```