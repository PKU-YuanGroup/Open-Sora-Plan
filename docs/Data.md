
## Sky


This is an un-condition datasets. [Link](https://drive.google.com/open?id=1xWLiU-MBGN7MrsFHQm4_yXmfHBsMbJQo)

```
sky_timelapse
├── readme
├── sky_test
├── sky_train
├── test_videofolder.py
└── video_folder.py
```

## UCF101

We test the code with UCF-101 dataset. In order to download UCF-101 dataset, you can download the necessary files in [here](https://www.crcv.ucf.edu/data/UCF101.php). The code assumes a `ucf101` directory with the following structure
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

1. Download videos from [UCF101.rar](https://www.crcv.ucf.edu/data/UCF101/UCF101.rar)
2. Download train/test splits from [UCF101TrainTestSplits-RecognitionTask.zip](https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip)
3. Uncompress `UCF101.rar` and `UCF101TrainTestSplits-RecognitionTask.zip`

## Kinetics-400

```
kinetics-400/
├── videos_train/
├── videos_val/
├── kinetics400_train_list_videos.txt
└── kinetics400_val_list_videos.txt
```

1. download kinetics-400 from [opendatalab site](https://opendatalab.com/OpenMMLab/Kinetics-400)
2. merge multi-part gz files to a single gz file: `cat Kinetics-400.tar.gz.* > Kinetics-400.tar.gz`
3. Uncompressing gz file: `tar -xvzf Kinetics-400.tar.gz`


## Offline feature extraction
Coming soon...
