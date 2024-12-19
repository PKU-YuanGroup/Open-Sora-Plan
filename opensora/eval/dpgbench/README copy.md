
The original code is from [DPG-Bench](https://github.com/TencentQQGYLab/ELLA).


## Requirements and Installation

recording how to install the dependencies


## Eval

### Step 1

配置环境，运行生成t2i的采样脚本
```
bash opensora/eval/dpgbench/step1_sample_t2v_v1_5_384.sh
```


### Step 2

运行pip install -r requirements-for-dpg_bench.txt

将mplug模型下载到本地
```
modelscope download --model 'iic/mplug_visual-question-answering_coco_large_en' --local_dir 'iic/mplug_visual-question-answering_coco_large_en'
```
local_dir为本地存放权重路径

运行脚本
```
bash dpgbench/step2_compute_dpg_bench.sh $YOUR_IMAGE_PATH $RESOLUTION
```

