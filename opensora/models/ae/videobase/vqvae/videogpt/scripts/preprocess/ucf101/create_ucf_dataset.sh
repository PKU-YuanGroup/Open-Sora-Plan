#! /bin/sh

mkdir -p ${1}

# Download UCF-101 video files
wget --no-check-certificate -P ${1} https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar x ${1}/UCF101.rar ${1}
rm ${1}/UCF101.rar

# Download UCF-101 train/test splits
wget --no-check-certificate -P ${1} https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
unzip ${1}/UCF101TrainTestSplits-RecognitionTask.zip -d ${1}
rm ${1}/UCF101TrainTestSplits-RecognitionTask.zip

# Move video files into train / test directories based on train/test split
python scripts/preprocess/ucf101/ucf_split_train_test.py ${1} 1

# Delete leftover files
rm -r ${1}/UCF-101
