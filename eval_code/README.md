# Metrics_on_Video_Quality

You can easily calculate the following video quality metrics, which supports the batch-wise process.
- **CLIP-SCORE**: It uses the pretrained CLIP model to measure the cosine similarity between two modalities.
- **FVD**: Frechét Video Distance
- **SSIM**: structural similarity index measure
- **LPIPS**: learned perceptual image patch similarity
- **PSNR**: peak-signal-to-noise ratio

# Requrirement
## Environemnt
- install Pytorch (torch>=1.7.1)
- install CLIP
    ```
        pip install git+https://github.com/openai/CLIP.git
    ```
- install clip-cose from PyPi
    ```
        pip install clip-score
    ```
- Other package
    ```
        pip install lpips
        pip install scipy (scipy==1.7.3/1.9.3, if you use 1.11.3, **you will calculate a WRONG FVD VALUE!!!**)
        pip install numpy
        pip install pillow
        pip install torchvision>=0.8.2
        pip install ftfy
        pip install regex
        pip install tqdm
    ```
## Pretrain model
- FVD
    Before you cacluate FVD, you should first download the FVD pre-trained model. You can manually download any of the following and put it into FVD folder. 
    - `i3d_torchscript.pt` from [here](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt) 
    - `i3d_pretrained_400.pt` from [here](https://onedrive.live.com/download?cid=78EEF3EB6AE7DBCB&resid=78EEF3EB6AE7DBCB%21199&authkey=AApKdFHPXzWLNyI)


## Other Notices
1. Make sure the pixel value of videos should be in [0, 1].
2. We average SSIM when images have 3 channels, ssim is the only metric extremely sensitive to gray being compared to b/w.
3. Because the i3d model downsamples in the time dimension, `frames_num` should > 10 when calculating FVD, so FVD calculation begins from 10-th frame, like upper example.
4. For grayscale videos, we multiply to 3 channels
5. data input specifications for clip_score
> - Image Files:All images should be stored in a single directory. The image files can be in either .png or .jpg format.
> 
> - Text Files: All text data should be contained in plain text files in a separate directory. These text files should have the extension .txt.    
>
> Note: The number of files in the image directory should be exactly equal to the number of files in the text directory. Additionally, the files in the image directory and text directory should be paired by file name. For instance, if there is a cat.png in the image directory, there should be a corresponding cat.txt in the text directory.
>
> Directory Structure Example:
> ```
>   ├── path/to/image
>   │   ├── cat.png
>   │   ├── dog.png
>   │   └── bird.jpg
>   └── path/to/text
>       ├── cat.txt
>       ├── dog.txt
>       └── bird.txt
> ```

# Usage



```
# clip_score cross modality
python clip_score.py \
    --real_path path/to/image \
    --generated_path path/to/text \
    --batch-size 50 \
    --device "cuda"

# clip_score within the same modality
python clip_score.py \
    --real_path path/to/textA \
    --generated_path path/to/textB \
    --real_flag txt \
    --generated_flag txt \
    --batch-size 50 \
    --device "cuda"

python clip_score.py \
    --real_path path/to/imageA \
    --generated_path path/to/imageB \
    --real_flag img \
    --generated_flag img \
    --batch-size 50 \
    --device "cuda"


# fvd, psnr, ssim, lpips, 8 videos of a batch, 10 frames, 3 channels, 64x64 size.
import torch
from cal_fvd import calculate_fvd
from cal_psnr import calculate_psnr
from cal_ssim import calculate_ssim
from cal_lpips import calculate_lpips

NUMBER_OF_VIDEOS = 8
VIDEO_LENGTH = 30
CHANNEL = 3
SIZE = 64
videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE, SIZE, requires_grad=False)
device = torch.device("cuda")
device = torch.device("cpu")

import json
result = {}
result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv')
# result['fvd'] = calculate_fvd(videos1, videos2, device, method='videogpt')
result['ssim'] = calculate_ssim(videos1, videos2)
result['psnr'] = calculate_psnr(videos1, videos2)
result['lpips'] = calculate_lpips(videos1, videos2, device)
print(json.dumps(result, indent=4))
```

It means we calculate:

- `FVD-frames[:10]`, `FVD-frames[:11]`, ..., `FVD-frames[:30]` 
- `avg-PSNR/SSIM/LPIPS-frame[0]`, `avg-PSNR/SSIM/LPIPS-frame[1]`, ..., `avg-PSNR/SSIM/LPIPS-frame[:30]`, and their std.

We cannot calculate `FVD-frames[:8]`, and it will pass when calculating, see ps.6.

The result shows: a all-zero matrix and a all-one matrix, their FVD-30 (FVD[:30]) is 151.17 (styleganv method). We also calculate their standard deviation. Other metrics are the same. And we use the calculation method of styleganv.

```
{
    "fvd": {
        "value": {
            "10": 570.07320378183,
            "11": 486.1906542471159,
            "12": 552.3373915075898,
            "13": 146.6242330185728,
            "14": 172.57268402948895,
            "15": 133.88932632116126,
            "16": 153.11023578170108,
            "17": 357.56400892781204,
            "18": 382.1335612721498,
            "19": 306.7100176942531,
            "20": 338.18221898178774,
            "21": 77.95587603163293,
            "22": 82.49997632357349,
            "23": 64.41624523513073,
            "24": 66.08097153313875,
            "25": 314.4341061962642,
            "26": 316.8616746151064,
            "27": 288.884418528541,
            "28": 287.8192683223724,
            "29": 152.15076552354864,
            "30": 151.16806952692093
        },
        "video_setting": [
            8,
            3,
            30,
            64,
            64
        ],
        "video_setting_name": "batch_size, channel, time, heigth, width"
    },
        "video_setting": [
            8,
            3,
            30,
            64,
            64
        ],
        "video_setting_name": "batch_size, channel, time, heigth, width"
    },
    "ssim": {
        "value": {
            "0": 9.999000099990664e-05,
            ...,
            "29": 9.999000099990664e-05
        },
        "value_std": {
            "0": 0.0,
            ...,
            "29": 0.0
        },
        "video_setting": [
            30,
            3,
            64,
            64
        ],
        "video_setting_name": "time, channel, heigth, width"
    },
    "psnr": {
        "value": {
            "0": 0.0,
            ...,
            "29": 0.0
        },
        "value_std": {
            "0": 0.0,
            ...,
            "29": 0.0
        },
        "video_setting": [
            30,
            3,
            64,
            64
        ],
        "video_setting_name": "time, channel, heigth, width"
    },
    "lpips": {
        "value": {
            "0": 0.8140146732330322,
            ...,
            "29": 0.8140146732330322
        },
        "value_std": {
            "0": 0.0,
            ...,
            "29": 0.0
        },
        "video_setting": [
            30,
            3,
            64,
            64
        ],
        "video_setting_name": "time, channel, heigth, width"
    }
}
```

# Acknowledgement
The evaluation codebase refers to [clip-score](https://github.com/Taited/clip-score) and [common_metrics](https://github.com/JunyaoHu/common_metrics_on_video_quality).