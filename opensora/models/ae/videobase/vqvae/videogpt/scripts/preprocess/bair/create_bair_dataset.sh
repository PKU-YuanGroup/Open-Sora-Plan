#! /bin/sh

mkdir -p ${1}
mkdir -p ~/.cache/bair
wget -P ~/.cache/bair/ http://rail.eecs.berkeley.edu/datasets/bair_robot_pushing_dataset_v0.tar
tar -xf ~/.cache/bair/bair_robot_pushing_dataset_v0.tar -C ~/.cache/bair/

python scripts/preprocess/bair/bair_extract_images.py --data_dir ~/.cache/bair
python scripts/preprocess/bair/bair_image_to_hdf5.py --data_dir ~/.cache/bair --output_dir ${1}

rm -r ~/.cache/bair
