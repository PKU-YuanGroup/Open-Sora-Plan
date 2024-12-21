
The original code is from [improved-aesthetic-predictor-v2](https://github.com/christophschuhmann/improved-aesthetic-predictor).


## Requirements and Installation

下载[ViT-L/14](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt)到$CACHE_DIR
下载[sac+logos+ava1-l14-linearMSE.pth](https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/ava%2Blogos-l14-linearMSE.pth)到$CACHE_DIR


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
bash opensora/eval/laionaesv2/step2_run_model.sh
```