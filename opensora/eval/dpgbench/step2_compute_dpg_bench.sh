

PROMPT="DPGBench"
export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7
IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_shrink/${PROMPT}"
accelerate launch --num_machines 1 --num_processes 6 --multi_gpu --mixed_precision "fp16" --main_process_port 1234 \
    opensora/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 256 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug 

IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/deepnorm_skip/${PROMPT}"
accelerate launch --num_machines 1 --num_processes 6 --multi_gpu --mixed_precision "fp16" --main_process_port 1234 \
    opensora/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 256 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug 


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4/${PROMPT}"
accelerate launch --num_machines 1 --num_processes 6 --multi_gpu --mixed_precision "fp16" --main_process_port 1234 \
    opensora/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 256 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug 


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/glm4v/${PROMPT}"
accelerate launch --num_machines 1 --num_processes 6 --multi_gpu --mixed_precision "fp16" --main_process_port 1234 \
    opensora/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 256 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug 


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/mixnorm/${PROMPT}"
accelerate launch --num_machines 1 --num_processes 6 --multi_gpu --mixed_precision "fp16" --main_process_port 1234 \
    opensora/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 256 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug 


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm/${PROMPT}"
accelerate launch --num_machines 1 --num_processes 6 --multi_gpu --mixed_precision "fp16" --main_process_port 1234 \
    opensora/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 256 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug 


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/postnorm_skip/${PROMPT}"
accelerate launch --num_machines 1 --num_processes 6 --multi_gpu --mixed_precision "fp16" --main_process_port 1234 \
    opensora/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 256 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug 


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm/${PROMPT}"
accelerate launch --num_machines 1 --num_processes 6 --multi_gpu --mixed_precision "fp16" --main_process_port 1234 \
    opensora/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 256 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug 


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_normskip/${PROMPT}"
accelerate launch --num_machines 1 --num_processes 6 --multi_gpu --mixed_precision "fp16" --main_process_port 1234 \
    opensora/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 256 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug 


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/prenorm_skip/${PROMPT}"
accelerate launch --num_machines 1 --num_processes 6 --multi_gpu --mixed_precision "fp16" --main_process_port 1234 \
    opensora/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 256 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug 


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/qwenvl2/${PROMPT}"
accelerate launch --num_machines 1 --num_processes 6 --multi_gpu --mixed_precision "fp16" --main_process_port 1234 \
    opensora/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 256 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug 


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/sandwich/${PROMPT}"
accelerate launch --num_machines 1 --num_processes 6 --multi_gpu --mixed_precision "fp16" --main_process_port 1234 \
    opensora/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 256 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug 


IMAGE_DIR="/storage/ongoing/12.29/eval/t2i_ablation_arch/umt5/${PROMPT}"
accelerate launch --num_machines 1 --num_processes 6 --multi_gpu --mixed_precision "fp16" --main_process_port 1234 \
    opensora/eval/dpgbench/step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 256 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug 
