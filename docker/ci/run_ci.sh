#!/usr/bin/env bash

WORK_DIR=$(dirname "$(readlink -f "$0")")
source $WORK_DIR/../conf/setup_env.sh
DOCKER_USER=
DOCKER_GPU=

# help info
usage() {
    echo ""
    echo "Usage: $0 [-h] [-u <docker_username>] <positional-arg>"
    echo ""
    echo " -h: show help about usage"
    echo " -u: docker username"
    echo ""
    exit 1
}

# Use getopt to parse command-line options
OPTSTRING=":u:h"
while getopts ${OPTSTRING} opt; do
  case ${opt} in
    u)
      echo "use docker user: ${OPTARG}"
      DOCKER_USER=${OPTARG}/
      ;;
    h)
      usage
      ;;
    :)
      echo "Option -${OPTARG} requires an argument."
      usage
      ;;
    ?)
      echo "Invalid option: -${OPTARG}."
      usage
      ;;
  esac
done
shift $((OPTIND - 1))

# Run nvidia-smi to check for NVIDIA GPUs and their CUDA capabilities
nvidia-smi
if [ $? -eq 0 ]; then
    echo "This device supports CUDA."
    DOCKER_GPU="--gpus all"
fi

# Check running docker containers, and kill them
RUNNING_IDS="$(docker ps --filter ancestor=$DOCKER_USER$CI_TAG --format "{{.ID}}")"
if [ -n "$RUNNING_IDS" ]; then
    echo ' '
    echo "The running container ID is: $RUNNING_IDS, kill them!"
    docker kill $RUNNING_IDS
fi

# Run a new docker container instance
docker run \
    --rm \
    $DOCKER_GPU \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    $DOCKER_USER$CI_TAG $@
