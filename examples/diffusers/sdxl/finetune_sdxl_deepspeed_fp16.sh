Network="StableDiffusionXLFinetuneDeepspeed"

model_name="stabilityai/stable-diffusion-xl-base-1.0"
dataset_name="pokemon-blip-captions"
batch_size=24
max_train_steps=2000
checkpointing_steps=2000
validation_epochs=2000
mixed_precision="fp16"
resolution=512

source /usr/local/Ascend/ascend-toolkit/set_env.sh

for para in $*; do
  if [[ $para == --batch_size* ]]; then
    batch_size=$(echo ${para#*=})
  elif [[ $para == --max_train_steps* ]]; then
    max_train_steps=$(echo ${para#*=})
  elif [[ $para == --mixed_precision* ]]; then
    mixed_precision=$(echo ${para#*=})
  elif [[ $para == --resolution* ]]; then
    resolution=$(echo ${para#*=})
  elif [[ $para == --checkpointing_steps* ]]; then
    checkpointing_steps=$(echo ${para#*=})
  elif [[ $para == --validation_epochs* ]]; then
    validation_epochs=$(echo ${para#*=})
  fi
done

cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}
fi

output_path=${cur_path}/output/${ASCEND_DEVICE_ID}

echo ${output_path}
mkdir -p ${output_path}


start_time=$(date +%s)
echo "start_time: ${start_time}"

accelerate launch --config_file ./sdxl/accelerate_deepspeed_config.yaml \
  ./examples/text_to_image/train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$model_name \
  --caption_column="text" \
  --dataset_name=$dataset_name \
  --resolution=$resolution  \
  --train_batch_size=$batch_size \
  --checkpointing_steps=$checkpointing_steps \
  --learning_rate=1e-06 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision=$mixed_precision \
  --max_train_steps=$max_train_steps \
  --dataloader_num_workers=8 \
  --seed=1234 \
  --enable_npu_flash_attention \
  --output_dir=${output_path} > ${output_path}train_${mixed_precision}_sdxl_finetune_deepspeed.log 2>&1 &
wait


#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出性能FPS，需要模型审视修改
FPS=$(grep "FPS: " ${output_path}/train_${mixed_precision}_sdxl_finetune_deepspeed.log | awk '{print $NF}' | sed -n '100,199p' | awk '{a+=$1}END{print a/NR}')

#获取性能数据，不需要修改
#吞吐量
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#loss值，不需要修改
ActualLoss=$(grep -o "step_loss=[0-9.]*" ${output_path}/train_${mixed_precision}_sdxl_finetune_deepspeed.log | awk 'END {print $NF}')

#打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
#训练用例信息，不需要修改
BatchSize=${batch_size}
DeviceType=$(uname -m)
CaseName=${Network}_bs${BatchSize}_'8p'_'acc'

#单迭代训练时长
TrainingTime=$(awk 'BEGIN{printf "%.2f\n", '${batch_size}'*8/'${FPS}'}')

#关键信息打印到${CaseName}.log中，不需要修改
echo "Network = ${Network}" >${output_path}/${CaseName}.log
echo "BatchSize = ${BatchSize}" >>${output_path}/${CaseName}.log
echo "DeviceType = ${DeviceType}" >>${output_path}/${CaseName}.log
echo "CaseName = ${CaseName}" >>${output_path}/${CaseName}.log
echo "ActualFPS = ${ActualFPS}" >>${output_path}/${CaseName}.log
echo "TrainingTime = ${TrainingTime}" >>${output_path}/${CaseName}.log
echo "ActualLoss = ${ActualLoss}" >>${output_path}/${CaseName}.log
echo "E2ETrainingTime = ${e2e_time}" >>${output_path}/${CaseName}.log
