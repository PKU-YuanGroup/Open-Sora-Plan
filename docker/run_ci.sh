#!/usr/bin/env bash

WORK_DIR=$(dirname "$(readlink -f "$0")")
source $WORK_DIR/conf/setup_env.sh

RUNNING_IDS="$(docker ps --filter ancestor=$CI_TAG --format "{{.ID}}")"

if [ -n "$RUNNING_IDS" ]; then
    echo ' '
    echo "The running container ID is: $RUNNING_IDS, kill them!"
    docker kill $RUNNING_IDS
fi

# Run a new docker container instance
docker run \
    --rm \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    $CI_TAG $@
