
# 网络名称,同目录名称,需要模型审视修改
Network="StableDiffusion3Dreambooth"

scripts_path="./sd3"

# 预训练模型
model_name="stabilityai/stable-diffusion-3-medium-diffusers"
dataset_name="pokemon-blip-captions"
# input_dir="dog"
batch_size=4
max_train_steps=2000
mixed_precision="bf16"
resolution=1024
config_file="${scripts_path}/${mixed_precision}_accelerate_config.yaml"

source /usr/local/Ascend/ascend-toolkit/set_env.

#如果使用 input_dir="dog"，请修改dataset_name为input_dir
for para in $*; do
  if [[ $para == --model_name* ]]; then
    model_name=$(echo ${para#*=})
  elif [[ $para == --batch_size* ]]; then
    batch_size=$(echo ${para#*=})
  elif [[ $para == --max_train_steps* ]]; then
    max_train_steps=$(echo ${para#*=})
  elif [[ $para == --mixed_precision* ]]; then
    mixed_precision=$(echo ${para#*=})
  elif [[ $para == --resolution* ]]; then
    resolution=$(echo ${para#*=})
  elif [[ $para == --dataset_name* ]]; then
    dataset_name=$(echo ${para#*=})
  elif [[ $para == --config_file* ]]; then
    config_file=$(echo ${para#*=})
  fi
done

# cd到与test文件夹同层级目录下执行脚本，提高兼容性；test_path_dir为包含test文件夹的路径
cur_path=$(pwd)
cur_path_last_dirname=${cur_path##*/}
if [ x"${cur_path_last_dirname}" == x"test" ]; then
  test_path_dir=${cur_path}
  cd ..
  cur_path=$(pwd)
else
  test_path_dir=${cur_path}/test
fi

echo ${test_path_dir}

#创建DeviceID输出目录，不需要修改
output_path=${cur_path}/test/output/${ASCEND_DEVICE_ID}

mkdir -p ${output_path}

#训练开始时间，不需要修改
start_time=$(date +%s)
echo "start_time: ${start_time}"

#如果数据集为pokemon或其他，需要把 sks dog 修改为pokemon或其他
accelerate launch --config_file ${config_file} \
  ./examples/dreambooth/train_dreambooth_sd3.py \
  --pretrained_model_name_or_path=$model_name \
  --dataset_name=$dataset_name --caption_column="text" \
  --instance_prompt="A photo of sks dog" \
  --train_batch_size=$batch_size \
  --resolution=$resolution --random_flip \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=$max_train_steps \
  --learning_rate=1e-05 --lr_scheduler="constant_with_warmup" --lr_warmup_steps=0 \
  --max_grad_norm=1 \
  --validation_prompt="A photo of sks dog in a bucket" \
  --validation_epochs=25 \
  --mixed_precision=$mixed_precision \
  --checkpointing_steps=500 \
  --output_dir=${output_path} > ${output_path}train_${mixed_precision}.log 2>&1 &
wait

#训练结束时间，不需要修改
end_time=$(date +%s)
e2e_time=$(($end_time - $start_time))

#结果打印，不需要修改
echo "------------------ Final result ------------------"

#输出性能FPS，需要模型审视修改
FPS=$(grep "FPS: " ${output_path}/train_${mixed_precision}_sd3_dreambooth.log | awk '{print $NF}' | sed -n '100,199p' | awk '{a+=$1}END{print a/NR}')

#获取性能数据，不需要修改
# - 吞吐量
ActualFPS=$(awk 'BEGIN{printf "%.2f\n", '${FPS}'}')

# - 打印，不需要修改
echo "Final Performance images/sec : $ActualFPS"

# - loss值，不需要修改
ActualLoss=$(grep -o "loss=[0-9.]*" ${output_path}/train_${mixed_precision}_sd3_dreambooth.log | awk 'END {print $NF}')

# - 打印，不需要修改
echo "Final Train Loss : ${ActualLoss}"
echo "E2E Training Duration sec : $e2e_time"

#性能看护结果汇总
# - 训练用例信息，不需要修改
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