#!/bin/bash
# 用于开发者自测 UT/ST
set -e
BASE_DIR=$(dirname "$(readlink -f "$0")")
WORKSPACE=$(cd $BASE_DIR; cd ../; pwd)

TEST_TYPE="all"
SKIP_BUILD=1
for para in $*; do
  if [[ $para == --type* ]]; then
    TEST_TYPE=$(echo ${para#*=})
  elif [[ $para == --skip_build* ]]; then
    SKIP_BUILD=$(echo ${para#*=})
  fi
done

source /usr/local/Ascend/ascend-toolkit/set_env.sh

echo "init mindspeed-mm"
cd "${WORKSPACE}"
if [ $SKIP_BUILD -eq 1 ]
then
    echo "skip build environments"
else
    pip install -e .
    pip install -e .[test]
    if [ "${type}" == "ut" ]; then
        echo "build when ut test"
        pip install --upgrade build
        python -m build
    fi
fi
echo "start test"
cd "${WORKSPACE}/ci"
export PYTHONPATH=$PYTHONPATH:${WORKSPACE}
python access_control_test.py --type=${TEST_TYPE}