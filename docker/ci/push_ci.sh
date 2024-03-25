#!/usr/bin/env bash

WORK_DIR=$(dirname "$(readlink -f "$0")")
DOCKER_USER=
source $WORK_DIR/../conf/setup_env.sh

# Note:
#   Need to docker login first with your dockerhub username & password

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

# Push to docker hub
docker push $DOCKER_USER$CI_TAG
