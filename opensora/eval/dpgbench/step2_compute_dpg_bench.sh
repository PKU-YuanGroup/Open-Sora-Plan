# IMAGE_ROOT_PATH=$1
# RESOLUTION=$2
PIC_NUM=${PIC_NUM:-4}
PROCESSES=${PROCESSES:-8}
PORT=${PORT:-1234}
RESULTPATH="/storage/hxy/t2i/results/dpgbench/results/mmdit_last_results.txt" #保存结果的txt路径
VQACKPT="/storage/hxy/t2i/weight/dpgbench" #保存vqamodel的本地权重路径

num_layers=(12 1 12)

# Outer loop for tr_stage
for tr_stage in "${!num_layers[@]}"; do
    tr_layers=${num_layers[$tr_stage]}
    for ((tr_layer=0; tr_layer<tr_layers; tr_layer++)); do
        echo "/storage/ongoing/12.13/t2i/Open-Sora-Plan/t2i_ablation_arch_del/sandwich/del_s${tr_stage}_l${tr_layer}"
        # Add your commands here
        accelerate launch --num_machines 1 --num_processes $PROCESSES --multi_gpu --mixed_precision "fp16" --main_process_port $PORT \
          step2_compute_dpg_bench.py \
          --image_root_path /storage/hxy/t2i/results/dpgbench/results/sandwich/del_s${tr_stage}_l${tr_layer} \
          --resolution 256 \
          --pic_num $PIC_NUM \
          --res_path /storage/hxy/t2i/results/dpgbench/results/sandwich/del_s${tr_stage}_l${tr_layer}.txt \
          --vqa_model mplug \
          # --vqa_model_ckpt $VQACKPT \ 
    done
done


# accelerate launch --num_machines 1 --num_processes $PROCESSES --multi_gpu --mixed_precision "fp16" --main_process_port $PORT \
#   step2_compute_dpg_bench.py \
#   --image_root_path $IMAGE_ROOT_PATH \
#   --resolution $RESOLUTION \
#   --pic_num $PIC_NUM \
#   --res_path $RESULTPATH \
#   --vqa_model mplug \
#   # --vqa_model_ckpt $VQACKPT \ 

