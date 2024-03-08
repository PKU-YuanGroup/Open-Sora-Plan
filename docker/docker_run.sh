#!/usr/bin/env bash

WORK_DIR=$(dirname "$(readlink -f "$0")")
source $WORK_DIR/setup_env.sh

RUNNING_IDS="$(docker ps --filter ancestor=$TAG --format "{{.ID}}")"

if [ -n "$RUNNING_IDS" ]; then
    # Initialize an array to hold the container IDs
    declare -a container_ids=($RUNNING_IDS)

    # Get the first container ID using array indexing
    ID=${container_ids[0]}

    # Print the first container ID
    echo ' '
    echo "The running container ID is: $ID, enter it!"
else
    echo ' '
    echo "Not found running containers, run it!"

    # Run a new docker container instance
    ID=$(docker run \
        --rm \
        --gpus all \
        -itd \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
        -v $PWD:/home/$USER_NAME/workspace \
        -w /home/$USER_NAME/workspace \
        $(cat $WORK_DIR/ports.txt) \
        $TAG)
fi

docker logs $ID

echo ' '
echo ' '
echo '========================================='
echo ' '

docker exec -it $ID bash
