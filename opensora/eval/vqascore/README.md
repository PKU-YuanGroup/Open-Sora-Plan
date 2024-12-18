
The original code is from [VQAScore](https://github.com/linzhiqiu/t2v_metrics).


## Requirements and Installation

```
pip install git+https://github.com/openai/CLIP.git
pip install open-clip-torch
```

下载[zhiqiulin/clip-flant5-xxl](https://huggingface.co/zhiqiulin/clip-flant5-xxl)到$CACHE_DIR
下载[openai/clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)到$CACHE_DIR

## Eval

### Step 1

修改$PROMPT, 目前支持['DrawBench', 'PartiPrompts', 'GenAI']
```
bash opensora/eval/vqascore/step1_gen_samples.sh
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

#### GenAI-Bench

只有$PROMPT=='GenAI'才支持
```
bash opensora/eval/vqascore/step2_genai_image_eval.sh
```


#### VQAScore

```
bash opensora/eval/vqascore/step2_run_model.sh
```

