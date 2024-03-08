#!/usr/bin/env bash

WORK_DIR=$(dirname "$(readlink -f "$0")")
cd $WORK_DIR

source setup_env.sh

docker build -t $TAG --build-arg BASE_TAG=$BASE_TAG --build-arg USER_NAME=$USER_NAME --build-arg USER_PASSWD=$USER_PASSWD . -f dockerfile.base
