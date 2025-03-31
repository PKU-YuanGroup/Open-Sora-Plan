MINDSPEED_PATH="./MindSpeed/"
export PYTHONPATH=${MINDSPEED_PATH}:$PYTHONPATH

export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD

python convert_ckpt_from_megatron_to_pt.py