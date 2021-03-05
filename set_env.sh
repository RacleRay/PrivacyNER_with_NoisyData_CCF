#!/bin/bash

# ###########################################
echo $PYTHONPATH
echo "## 请输入绝对路径： "
read aa

# 当前shell退出失效
# export PYTHONPATH=$aa:$PYTHONPATH

cat >> ~/.bashrc << EOF
export PYTHONPATH=$aa:$PYTHONPATH
EOF
# 记得source，此处source不成功

echo "## After..."
echo $PYTHONPATH
echo "## 请在当前 terminal 中运行：source ~/.bashrc, 以更新配置"