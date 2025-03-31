MINDSPEED_PATH="./MindSpeed/"
export PYTHONPATH=${MINDSPEED_PATH}:$PYTHONPATH

export LD_PRELOAD=/lib/aarch64-linux-gnu/libGLdispatch.so.0
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1:$LD_PRELOAD

python examples/opensoraplan1.5/convert_mm_to_ckpt.py