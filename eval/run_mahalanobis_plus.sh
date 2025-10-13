#!/bin/bash

# Mahalanobis的rate参数列表
RATES=(0.01 0.05 0.1 0.15 0.2)
echo "删除旧的tensors和results目录"
rm -rf /root/wzhdesign/Igeood/tensors /root/wzhdesign/Igeood/results
# 循环执行三次（0,1,2）
for i in {0..2}; do
    echo "-------- 开始执行任务 $i --------"
    
    # 遍历所有rate值
    for RATE in "${RATES[@]}"; do
        echo "执行任务 $i, rate=$RATE"

        # 构建命令
        cmd="python /root/wzhdesign/Igeood/eval.py"
        cmd+=" mahalanobis_plus"
        cmd+=" --nn icarl"
        cmd+=" --in-dataset ustc_task_${i}_in"
        cmd+=" --out-dataset ustc_task_${i}_out"
        cmd+=" --temperature 1"
        cmd+=" -gpu 0 "
        cmd+=" -rate $RATE"
        cmd+=" >> eval/mahalanobis_plus_task_${i}_rate_${RATE}.log"
        
        # 打印并执行命令
        echo "执行命令: $cmd"
        eval $cmd
        
        # 检查执行结果
        if [ $? -eq 0 ]; then
            echo "✅ Mahalanobis Plus (rate=$RATE) 任务 $i 成功完成"
        else
            echo "❌ Mahalanobis Plus (rate=$RATE) 任务 $i 执行失败！"
            exit 1
        fi
    done
    
    echo "-------- 任务 $i 完成 --------"
    echo
done