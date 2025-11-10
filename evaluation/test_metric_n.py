import os
import json
import csv
from collections import defaultdict
from multiprocessing import Process, Queue, current_process

import torch
from sacrebleu.metrics import BLEU
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from comet import load_from_checkpoint

# -------------------- 参数 ----------------------
file_path = "../output/srt_large_27b_fleurs_eng_70.jsonl"
gpus = [0,1,2,3,4,5,6,7]  # 可见 GPU 列表
batch_size = 64

normalizer = BasicTextNormalizer()

taslp_70 = ['afr', 'amh', 'ara', 'asm', 'azj', 'bel', 'ben', 'bos', 'bul', 'cat', 'ces', 'cmn', 'cym', 'dan', 'deu', 'ell', 'eng', 'est', 'fas', 'fin', 'fra', 'glg', 'guj', 'heb', 'hin', 'hrv', 'hun', 'hye', 'ind', 'isl', 'ita', 'jav', 'jpn', 'kan', 'kat', 'kaz', 'khm', 'kir', 'kor', 'lao', 'lav', 'lit', 'mal', 'mkd', 'msa', 'mya', 'nld', 'nob', 'npi', 'pan', 'pol', 'por', 'ron', 'rus', 'slk', 'slv', 'spa', 'srp', 'swe', 'swh', 'tam', 'tel', 'tgl', 'tha', 'tur', 'ukr', 'urd', 'uzb', 'vie', 'yue']

src_langs = taslp_70
tgt_langs = taslp_70
test_metrics = ["idx","iso3","spbleu","comet"]

# -------------------- 读取 jsonl ----------------------
lang_groups = defaultdict(lambda: defaultdict(lambda: {
    "asr_gt": [], "asr_re": [], "st_gt": [], "st_re": []
}))

with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        gt = data.get("gt","")
        prompt = data.get("prompt","")
        response = data.get("response","") or data.get("s2tt","")

        src_lang = prompt.split("|>")[0].split("<|")[-1]
        tgt_lang = prompt.split("<|")[-1].split("|>")[0]
        if src_lang == tgt_lang: continue
        if src_lang not in src_langs or tgt_lang not in tgt_langs: continue

        prompt_tag = f"<|{src_lang}|><|{tgt_lang}|>"
        split_res = response.split(prompt_tag)
        split_gt = gt.split(prompt_tag)

        asr_gt = split_gt[0] if len(split_gt)==2 else gt
        st_gt = split_gt[1] if len(split_gt)==2 else gt
        asr_re = split_res[0] if len(split_res)==2 else response
        st_re = split_res[1] if len(split_res)==2 else response.split("|>")[-1]

        lang_groups[src_lang][tgt_lang]["asr_gt"].append(asr_gt.strip())
        lang_groups[src_lang][tgt_lang]["asr_re"].append(asr_re.strip())
        lang_groups[src_lang][tgt_lang]["st_gt"].append(st_gt.strip())
        lang_groups[src_lang][tgt_lang]["st_re"].append(st_re.strip())

# -------------------- 任务拆分 ----------------------
tasks = []
for src_lang in sorted(lang_groups.keys()):
    for tgt_lang in sorted(lang_groups[src_lang].keys()):
        d = lang_groups[src_lang][tgt_lang]
        tasks.append((src_lang, tgt_lang, d["asr_gt"], d["asr_re"], d["st_re"], d["st_gt"]))

# 将任务均分给每个 GPU
chunk_size = len(tasks) // len(gpus)
task_chunks = [tasks[i*chunk_size:(i+1)*chunk_size] for i in range(len(gpus)-1)]
task_chunks.append(tasks[(len(gpus)-1)*chunk_size:])  # 最后一块包含剩余任务

# -------------------- 子进程执行函数 ----------------------
def worker(gpu, task_chunk, output_queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    device = torch.device("cuda:0")
    print(f"[{current_process().name}] Loading COMET model on GPU {gpu}")
    comet_model = load_from_checkpoint("../models/wmt22-comet-da/checkpoints/model.ckpt").half().to(device)

    results = []
    for src_lang, tgt_lang, asr_gt, asr_re, st_re, st_gt in task_chunk:

        spbleu = BLEU(tokenize="flores200")
        spbleu_score = spbleu.corpus_score(st_re, [st_gt]).score

        # COMET
        comet_data = [{'src': s, 'mt': p, 'ref': r} for s,p,r in zip(asr_gt, st_re, st_gt)]
        comet_score = comet_model.predict(comet_data, batch_size=batch_size, devices=[0])['system_score']*100

        print_result = {
            "iso3": f"{src_lang}_{tgt_lang}",
            "spbleu": round(spbleu_score,2),
            "comet": round(comet_score,2)
        }
        results.append(print_result)
        print(print_result)
        print(f"[GPU{gpu}] {src_lang}->{tgt_lang} done")

    output_queue.put(results)

# -------------------- 启动多进程 ----------------------
if __name__ == "__main__":
    from multiprocessing import Queue

    output_queue = Queue()
    processes = []

    for i, gpu in enumerate(gpus):
        p = Process(target=worker, args=(gpu, task_chunks[i], output_queue))
        p.start()
        processes.append(p)

    # 收集结果
    all_results = []
    for _ in processes:
        all_results.extend(output_queue.get())

    for p in processes:
        p.join()

    # -------------------- 写 CSV ----------------------
    output_csv = f"{file_path.split('/')[-1].split('.')[0]}_multiGPU.csv"
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(test_metrics)
        for idx, r in enumerate(all_results):
            writer.writerow([idx+1, r["iso3"], r["spbleu"], r["comet"]])

    print(f"✅ Results saved: {output_csv}")
