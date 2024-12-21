IMAGE_ROOT_PATH=$1
RESOLUTION=$2
PIC_NUM=${PIC_NUM:-4}
PROCESSES=${PROCESSES:-8}
PORT=${PORT:-1234}
RESULTPATH="/storage/hxy/t2i/results/dpgbench/results/result_6B.txt" #保存结果的txt路径
VQACKPT="/storage/hxy/t2i/weight/dpgbench" #保存vqamodel的本地权重路径

accelerate launch --num_machines 1 --num_processes $PROCESSES --multi_gpu --mixed_precision "fp16" --main_process_port $PORT \
  step2_compute_dpg_bench.py \
  --image_root_path $IMAGE_ROOT_PATH \
  --resolution $RESOLUTION \
  --pic_num $PIC_NUM \
  --res_path $RESULTPATH \
  --vqa_model mplug \
  # --vqa_model_ckpt $VQACKPT \ 

