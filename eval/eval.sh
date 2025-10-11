#!/bin/bash

# 默认方法列表
DEFAULT_METHODS=("mahalanobis" "mahalanobis_plus")

# 检查是否提供了方法参数
if [ $# -gt 0 ]; then
    METHODS=("$@")
else
    METHODS=("${DEFAULT_METHODS[@]}")
fi

# 删除结果和tensor目录
rm -rf results tensors

# 遍历所有方法
for METHOD in "${METHODS[@]}"; do
    echo "========== 开始执行方法: $METHOD =========="
    
    # 循环执行三次（0,1,2）
    for i in {0..2}; do
        echo "-------- 开始执行任务 $i --------"
        
        # 构建命令
        cmd="python /root/wzhdesign/Igeood/eval.py"
        cmd+=" $METHOD"
        cmd+=" --nn icarl"
        cmd+=" --in-dataset ustc_task_${i}_in"
        cmd+=" --out-dataset ustc_task_${i}_out"
        cmd+=" --temperature 1"
        cmd+=" -gpu 0 "
        cmd+=" >> eval/${METHOD}_task_${i}.log"
        
        # 打印并执行命令
        echo "执行命令: $cmd"
        eval $cmd
        
        # 检查执行结果
        if [ $? -eq 0 ]; then
            echo "✅ 方法 $METHOD 任务 $i 成功完成"
        else
            echo "❌ 方法 $METHOD 任务 $i 执行失败！"
            exit 1
        fi
        
        echo "-------- 任务 $i 完成 --------"
        echo
    done
    
    echo "========== 方法 $METHOD 所有任务完成 =========="
    echo
done

echo "🎉 所有方法和任务执行完毕！"