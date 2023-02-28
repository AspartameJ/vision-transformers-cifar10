#!/bin/bash
export RANK_SIZE=4
export RANK_ID=0
export HCCL_WHITELIST_DISABLE=1
device_id=0
test_path_dir=`pwd`
if [ $ASCEND_DEVICE_ID ];then
    echo "device id is ${ASCEND_DEVICE_ID}"
    # 平台运行软链数据集路径
elif [ ${device_id} ];then
    export ASCEND_DEVICE_ID=${device_id}
    echo "device id is ${ASCEND_DEVICE_ID}"
else
    "[Error] device id must be config"
    exit 1
fi

#################创建日志输出目录，不需要修改#################



#################启动训练脚本#################
# 训练开始时间，不需要修改
start_time=$(date +%s)

for((RANK_ID=0;RANK_ID<$RANK_SIZE;RANK_ID++));
do
echo "DEVICE_ID=$RANK_ID"
if [ -d ${test_path_dir}/output/${RANK_ID} ];then
    rm -rf ${test_path_dir}/output/${RANK_ID}
    mkdir -p ${test_path_dir}/output/$RANK_ID
else
    mkdir -p ${test_path_dir}/output/$RANK_ID
fi
##其他参数正常设置
nohup python3 train_cifar10_npu_distribute.py --net swin --n_epochs 400 --bs 2048 --distributed --rank-size=$RANK_SIZE --device_id=$RANK_ID\
   > ${test_path_dir}/output/${RANK_ID}/train_${RANK_ID}.log 2>&1 &
  
done
