
**We need more dataset**, please refer to the [Open-Sora-Dataset](https://github.com/PKU-YuanGroup/Open-Sora-Dataset) for details.

## v1.0.0

### Text-to-Video

We open source v1.0.0 all the training data, the annotations and the original video can be found [here](https://huggingface.co/datasets/LanguageBind/Open-Sora-Plan-v1.0.0).

These data consist of segmented video clips, with each clip obtained through center cropping. The resolution of each clip is 512×512. There are 64 frames in each clip, and their corresponding captions can be found in the annotation files.

We present additional details in [report](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.0.0.md#data-construction) and [Open-Sora-Dataset](https://github.com/PKU-YuanGroup/Open-Sora-Dataset).

### Class-condition

In order to download UCF-101 dataset, you can download the necessary files in [here](https://www.crcv.ucf.edu/data/UCF101.php). The code assumes a `ucf101` directory with the following structure
```
UCF-101/
    ApplyEyeMakeup/
        v1.avi
        ...
    ...
    YoYo/
        v1.avi
        ...
```

### Un-condition

We use [sky_timelapse](https://drive.google.com/open?id=1xWLiU-MBGN7MrsFHQm4_yXmfHBsMbJQo), which is an un-condition datasets.

```
sky_timelapse
├── readme
├── sky_test
├── sky_train
├── test_videofolder.py
└── video_folder.py
```
