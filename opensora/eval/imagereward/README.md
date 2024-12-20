
The original code is from [ImageReward](https://github.com/THUDM/ImageReward).


## Requirements and Installation

pip install git+https://github.com/openai/CLIP.git
pip install fairscale==0.4.13

下载[THUDM/ImageReward](https://huggingface.co/THUDM/ImageReward)到$CACHE_DIR
下载[google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)到$CACHE_DIR

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
bash opensora/eval/imagereward/step2_run_model.sh
```