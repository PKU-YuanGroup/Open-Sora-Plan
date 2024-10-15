# MindSpeed-MM
We have adapted OpenSoraPlan1.3 based on the MindSpeed-MM suite, but currently only the inference of the T2V model on npu is supported. We will add support for the I2V model and the corresponding training part as soon as possible.

## Requirements and Installation

1. Clone this repository and navigate to mindspeed-mm brach
```
git clone https://github.com/PKU-YuanGroup/Open-Sora-Plan
git checkout mindspeed-mm
cd Open-Sora-Plan
```
2. Install required packages
    1. We recommend the requirements as follows.
       * Python >= 3.8
       * Pytorch == 2.1.0
       * torch_npu == 2.1.0
    ```
    conda create -n opensora python=3.8 -y
    conda activate opensora
    pip install torch==2.1.0
    pip install torch_npu==2.1.0.post6
    ```
   2. decord
    ```
    # ref https://github.com/dmlc/decord
    git clone --recursive https://github.com/dmlc/decord
    mkdir build && cd build 
    cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR=/usr/local/ffmpeg 
    make 
    cd ../python 
    pwd=$PWD 
    echo "PYTHONPATH=$PYTHONPATH:$pwd" >> ~/.bashrc 
    source ~/.bashrc 
    python3 setup.py install --user
    ```
   3. apex
    ```
    # ref https://gitee.com/ascend/apex 
    git clone -b master https://gitee.com/ascend/apex.git
    cd apex/
    bash scripts/build.sh --python=3.8
    cd apex/dist/
    pip3 uninstall apex
    pip3 install --upgrade apex-0.1_ascend*-cp38-cp38m-linux_aarch64.whl
    cd ..
    ```
   4. Megatron-LM
    ```
    git clone https://github.com/NVIDIA/Megatron-LM.git
    cd Megatron-LM
    git checkout core_r0.6.0
    cp -r megatron ../MindSpeed-MM/
    cd ..
    ```
   5. MindSpeed
    ```
    git clone https://gitee.com/ascend/MindSpeed.git
    cd MindSpeed
    git checkout 5dc1e83b
    pip install -r requirements.txt 
    pip3 install -e .
    cd ..
    ```
   6. MindSpeed-MM
    ```
    cd MindSpeed-MM
    pip install -e .
    ```
## Inference
```shell
bash examples/opensoraplan1.3/inference_t2v_1_3.sh
```

We support both tensor parallelism and sequence parallelism to accelerate the inference.

| Argparse                              | Usage                                                                                                                                                                                                                                                                                                                                                 |
|:--------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| _Tensor Parallelism_                  | TP=2                                                                                                                                                                                                                                                                                                                                                  |
| _Sequence Parallelism_                | CP=4                                                                                                                                                                                                                                                                                                                                                  |

Please modify the configuration information in `examples/opensoraplan1.3/inference_t2v_model1_3.json.`

| Argparse          | Usage                                                                                    |
|:------------------|:-----------------------------------------------------------------------------------------|
| _ae_              |                                                                                          |
| `use_tiling`      | Use tiling to deal with videos of high resolution and long time.                         |
| _Load weights_    |                                                                                          |
| `from_pretrained` | /path/to/model_dir. A directory containing the checkpoint of model is used for inference |
| _pipeline_config_ |                                                                                          |
| `input_size`      | The number of frames and the resolution of generated videos                              |
| _Other_           |                                                                                          |
| `save_path`       | The output path of the generated videos.                                                                 |
