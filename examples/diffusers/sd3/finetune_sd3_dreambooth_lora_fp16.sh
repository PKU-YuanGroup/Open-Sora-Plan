Network="StableDiffusion3DreamboothLoRA"

model_name="stabilityai/stable-diffusion-3-medium-diffusers"
input_dir="dog"
batch_size=8
max_train_steps=5000
mixed_precision="fp16"
resolution=512
gradient_accumulation_steps=1

source /usr/local/Ascend/ascend-toolkit/set_env.sh

cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}
fi

output_path=${cur_path}/test/output/${ASCEND_DEVICE_ID}

echo ${output_path}
mkdir -p ${output_path}


start_time=$(date +%s)
echo "start_time: ${start_time}"


accelerate launch ./examples/dreambooth/train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path=$model_name  \
  --instance_data_dir=$input_dir \
  --output_dir=$output_path \
  --instance_prompt="a photo of sks dog" \
  --resolution=$resolution \
  --train_batch_size=$batch_size \
  --mixed_precision=$mixed_precision \
  --gradient_accumulation_steps=$gradient_accumulation_steps \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=$max_train_steps \
  --seed="0" > ${output_path}/train_${mixed_precision}_sd3_dreambooth_lora.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出性能FPS，需要模型审视修改
FPS=$(grep "FPS: " ${output_path}/train_${mixed_precision}_sd3_dreambooth_lora.log | awk '{print $NF}' | sed -n '100,199p' | awk '{a+=$1}END{print a/NR}')

#获取性能数据，不需要修改
#吞吐量
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

#打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

#loss值，不需要修改
ActualLoss=$(grep -o "loss=[0-9.]*" ${output_path}/train_${mixed_precision}_sd3_dreambooth_lora.log | awk 'END {print $NF}')

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
