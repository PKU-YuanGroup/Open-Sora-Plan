basepath=$(cd `dirname $0`; cd ../../; pwd)
export PYTHONPATH=${basepath}:$PYTHONPATH
python3.8 ${basepath}/tests/st/st_demo.py
exit $?