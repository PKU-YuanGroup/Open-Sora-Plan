#!/usr/bin/env bash

WORK_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_DIR=$WORK_DIR/..
DOCKER_USER=
USER_ID=$(id -u $USER)
GROUP_ID=$(id -g $USER)

# include config
source $WORK_DIR/../conf/setup_env.sh

# help info
usage() {
    echo ""
    echo "Usage: $0 [-h] [-u <docker_username>] [-d <project_dir>]"
    echo ""
    echo " -h: show help about usage"
    echo " -u: docker username"
    echo " -d: project root dir"
    echo ""
    exit 1
}

# Use getopt to parse command-line options
OPTSTRING=":u:d:h"
while getopts ${OPTSTRING} opt; do
  case ${opt} in
    u)
      echo "use docker user: ${OPTARG}"
      DOCKER_USER=${OPTARG}/
      ;;
    d)
      echo "use project root dir: ${OPTARG}"
      PROJECT_DIR=${OPTARG}
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

CONTEXT_DIR=$PROJECT_DIR/Context

# build base docker image
pushd $WORK_DIR/../conf
docker build --no-cache -t $DOCKER_USER$TAG --build-arg BASE_IMG=$BASE_IMG --build-arg USER_NAME=$USER --build-arg USER_PASSWD=$USER_PASSWD --build-arg USER_ID=$USER_ID --build-arg GROUP_ID=$GROUP_ID . -f $WORK_DIR/../dockerfile/dockerfile.base
popd

# cp git repo files to context
pushd $PROJECT_DIR
rm -rf $CONTEXT_DIR && mkdir -p $CONTEXT_DIR
find "." -mindepth 1 -maxdepth 1 ! -name "$(basename $CONTEXT_DIR -d)" -exec cp -r {} "$CONTEXT_DIR" \;
popd

# generate entrypoint
cat > $CONTEXT_DIR/entrypoint.sh <<EOF
#!/usr/bin/env bash
echo "Starting the application..."
exec "\$@"
EOF

# build ci docker image
pushd $CONTEXT_DIR
docker build --no-cache -t $DOCKER_USER$CI_TAG --build-arg BASE_IMG=$DOCKER_USER$TAG . -f $WORK_DIR/../dockerfile/dockerfile.ci
popd

# clean download cache files
rm -rf $CONTEXT_DIR