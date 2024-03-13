#!/usr/bin/env bash

WORK_DIR=$(dirname "$(readlink -f "$0")")
CONTEXT_DIR=$WORK_DIR/Context

# include config
source $WORK_DIR/conf/setup_env.sh

# remove old docker images
docker rmi $TAG
docker rmi $CI_TAG

# build base docker image
pushd $WORK_DIR/conf
docker build --no-cache -t $TAG --build-arg BASE_TAG=$BASE_TAG --build-arg USER_NAME=$USER_NAME --build-arg USER_PASSWD=$USER_PASSWD . -f $WORK_DIR/dockerfile/dockerfile.base
popd

# download context files from git repo
rm -rf $CONTEXT_DIR
git clone -b $GIT_BRANCH $GIT_REPO $CONTEXT_DIR

# generate entrypoint
cat > $CONTEXT_DIR/entrypoint.sh <<EOF
#!/usr/bin/env bash
echo "Starting the application..."
exec "\$@"
EOF

# build ci docker image
pushd $CONTEXT_DIR
docker build --no-cache -t $CI_TAG --build-arg TAG=$TAG --build-arg USER_NAME=$USER_NAME . -f $WORK_DIR/dockerfile/dockerfile.ci
popd

# clean download cache files
rm -rf $CONTEXT_DIR