#!/usr/bin/env bash

WORK_DIR=$(dirname "$(readlink -f "$0")")
DOCKER_USER=
USER_ID=$(id -u $USER)
GROUP_ID=$(id -g $USER)

# include config
source $WORK_DIR/conf/setup_env.sh

# help info
usage() {
    echo ""
    echo "Usage: $0 [-h] [-u <docker_username>]"
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

# build developement docker image
pushd $WORK_DIR/conf
docker build -t $DOCKER_USER$TAG --build-arg BASE_IMG=$BASE_IMG --build-arg USER_NAME=$USER --build-arg USER_PASSWD=$USER_PASSWD --build-arg USER_ID=$USER_ID --build-arg GROUP_ID=$GROUP_ID . -f $WORK_DIR/dockerfile/dockerfile.base
popd
