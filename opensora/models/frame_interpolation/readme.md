#### Frame Interpolation

We use AMT as our frame interpolation model. (Thanks [AMT](https://github.com/MCG-NKU/AMT)) After sampling, you can use frame interpolation model to interpolate your video smoothly.

1. Download the pretrained weights from [AMT](https://github.com/MCG-NKU/AMT), we recommend using the largest model AMT-G to achieve the best performance. 
2. Run the script of frame interpolation.
```
python opensora/models/frame_interpolation/interpolation.py --ckpt /path/to/ckpt --niters 1 --input /path/to/input/video.mp4 --output_path /path/to/output/floder --frame_rate 30
```
3. The output video will be stored at output_path and its duration time is equal `the total number of frames after frame interpolation / the frame rate`
##### Frame Interpolation Specific Settings

* `--ckpt`: Pretrained model of [AMT](https://github.com/MCG-NKU/AMT). We use AMT-G as our frame interpolation model. 
* `--niter`: Iterations of interpolation. With $m$ input frames, `[N_ITER]` $=n$ corresponds to $2^n\times (m-1)+1$ output frames.
* `--input`: Path of the input video.
* `--output_path`: Folder Path of the output video.
* `--frame_rate"`: Frame rate of the output video. 
