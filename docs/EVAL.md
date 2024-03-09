# Evaluate the generated videos quality

You can easily calculate the following video quality metrics, which supports the batch-wise process.
- **CLIP-SCORE**: It uses the pretrained CLIP model to measure the cosine similarity between two modalities.
- **FVD**: Frechét Video Distance
- **SSIM**: structural similarity index measure
- **LPIPS**: learned perceptual image patch similarity
- **PSNR**: peak-signal-to-noise ratio

# Requirement 
## Environment
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

6. data input specifications for fvd, psnr, ssim, lpips

> Directory Structure Example:
> ```
>   ├── path/to/generated_image
>   │   ├── cat.mp4
>   │   ├── dog.mp4
>   │   └── bird.mp4
>   └── path/to/real_image
>       ├── cat.mp4
>       ├── dog.mp4
>       └── bird.mp4
> ```



# Usage

```
# you change the file path and need to set the frame_num, resolution etc...

# clip_score cross modality
cd opensora/eval
bash script/cal_clip_score.sh



# fvd 
cd opensora/eval
bash script/cal_fvd.sh

# psnr
cd opensora/eval
bash eval/script/cal_psnr.sh


# ssim
cd opensora/eval
bash eval/script/cal_ssim.sh


# lpips
cd opensora/eval
bash eval/script/cal_lpips.sh
```

# Acknowledgement
The evaluation codebase refers to [clip-score](https://github.com/Taited/clip-score) and [common_metrics](https://github.com/JunyaoHu/common_metrics_on_video_quality).