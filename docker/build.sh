#!/usr/bin/env bash

WORK_DIR=$(dirname "$(readlink -f "$0")")

# include config
source $WORK_DIR/conf/setup_env.sh

# build developement docker image
pushd $WORK_DIR/conf
docker build -t $TAG --build-arg BASE_TAG=$BASE_TAG --build-arg USER_NAME=$USER_NAME --build-arg USER_PASSWD=$USER_PASSWD . -f $WORK_DIR/dockerfile/dockerfile.base
popd
