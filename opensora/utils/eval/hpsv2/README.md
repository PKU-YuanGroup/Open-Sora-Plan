
The original code is from [HPSv2](https://github.com/tgxs002/HPSv2).


## Requirements and Installation

```
pip install open_clip_torch
```

下载[xswu/HPSv2](https://huggingface.co/xswu/HPSv2)到$CACHE_DIR
下载[laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)到$CACHE_DIR

## Eval

### Step 1

修改$PROMPT, 目前支持['GenAI527', 'GenAI1600', 'DALLE3', 'DOCCI-Test-Pivots', 'DrawBench', 'Gecko-Rel', 'PartiPrompts']
```
bash opensora/eval/step1_gen_samples.sh
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
bash opensora/eval/hpsv2/step2_run_model.sh
```