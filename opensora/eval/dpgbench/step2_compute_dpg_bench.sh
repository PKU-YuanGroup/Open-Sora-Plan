IMAGE_ROOT_PATH=$1
RESOLUTION=$2
PIC_NUM=${PIC_NUM:-4}
PROCESSES=${PROCESSES:-8}
PORT=${PORT:-1234}
RESULTPATH="/storage/hxy/t2i/opensora/Open-Sora-Plan/opensora/eval/dpgbench_test/score_result/result.txt" #保存结果的txt路径
VQACKPT="/storage/hxy/t2i/opensora/Open-Sora-Plan/opensora/eval/dpgbench_test/mplug" #保存vqamodel的本地权重路径

accelerate launch --num_machines 1 --num_processes $PROCESSES --multi_gpu --mixed_precision "fp16" --main_process_port $PORT \
  step2_compute_dpg_bench.py \
  --image-root-path $IMAGE_ROOT_PATH \
  --resolution $RESOLUTION \
  --pic-num $PIC_NUM \
  --res-path $$RESULTPATH \
  --vqa-model mplug \
  --vqa-model-ckpt $VQACKPT \ 

