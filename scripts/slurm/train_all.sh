#!/bin/bash
#SBATCH -J open-sora
#SBATCH -p sora                     ### Partition name , depends on the name on cluster, for example: sora
#SBATCH -N 1                        ### Number of nodes
#SBATCH --cpus-per-task=128         ### Number of CPUs, normally 16 for one gpu.
#SBATCH --gres=gpu:hgx:8            ### Number of GPUs, xxx means name of GPU depends on the cluster you are using for example hgx usuallys refers to a100 or h100
#SBATCH --mem 1024GB                ### Memory allocation
#SBATCH -o train_all_debug.out      ### File to store standard output


###### add or change the training scripts for your needs, after submit to hpc, the following tasks will run one by one
bash /Users/luoyaxin/Desktop/Open-Sora-Plan/scripts/videogpt/train_videogpt.sh 
bash /Users/luoyaxin/Desktop/Open-Sora-Plan/scripts/un_condition/train_imgae_with_img.sh\
bash /Users/luoyaxin/Desktop/Open-Sora-Plan/scripts/text_condition/train_feature_vidae_with_img.sh
bash /Users/luoyaxin/Desktop/Open-Sora-Plan/scripts/class_condition/train_imgae_with_img.sh 


###### run sbatch train_all.sh to submit the job to hpc######