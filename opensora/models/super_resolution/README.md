
## Environment Preparation

For video super resolution, please prepare your own python envirment from [RGT](https://github.com/zhengchen1999/RGT)  and down the [ckpt](https://drive.google.com/drive/folders/1zxrr31Kp2D_N9a-OUAPaJEn_yTaSXTfZ) into the folder like 
```bash
./experiments/pretrained_models/RGT_x2.pth
```

## Video Super Resolution
The inferencing instruction is in [run.py](run.py).
```bash
python run.py --SR x4 --root_path /path_to_root --input_dir /path_to_input_dir --output_dir /path_to_video_output
```
You can configure some more detailed parameters in [run.py](run.py) such as . 
```bash
--mul_numwork 16 --use_chop False
```
We recommend using `` --use_chop = False `` when memory allows. 
Note that in our tests.

A single frame of 256x256 requires about 3G RAM-Usage, and a single 4090 card can process about one frame per second.

A single frame of 512x512 takes about 19G RAM-Usage, and a single 4090 takes about 5 seconds to process a frame.


