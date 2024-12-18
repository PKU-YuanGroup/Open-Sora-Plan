
The original code is from [MPS](https://github.com/Kwai-Kolors/MPS).


## Requirements and Installation

下载[laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)到$CACHE_DIR
下载[MPS_overall_checkpoint.pth](https://github.com/Kwai-Kolors/MPS?tab=readme-ov-file#download-the-mps-checkpoint)到$CACHE_DIR


## Eval

### Step 1

修改$PROMPT, 目前支持['DrawBench', 'PartiPrompts', 'GenAI']
```
bash opensora/eval/mps/step1_gen_samples.sh
```

```
📦 opensora/
├── 📂 eval/
│   ├── 📂 gen_img_for_human_pref/
│       ├── 📂 GenAI/
│       |   ├── 📄 $ID.jpg
│       |   ├── 📄 $ID.jpg
│       |   └── 📄 .....
│       ├── 📂 DrawBench/
│       |   ├── 📄 $ID.jpg
│       |   ├── 📄 $ID.jpg
│       |   └── 📄 .....
│       |── 📂 .....
```

### Step 2

```
bash opensora/eval/mps/step2_run_model.sh
```