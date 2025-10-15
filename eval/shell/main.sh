#!/bin/bash
ROOT=$(readlink -f $(dirname $0))
cd $ROOT
# 默认方法列表
DEFAULT_METHODS=("mahalanobis" "mahalanobis_multi"  "mahalanobis_ood" "mahalanobis_plus" "igeood_plus") 

# 检查是否提供了方法参数
if [ $# -gt 0 ]; then
    METHODS=("$@")
else
    METHODS=("${DEFAULT_METHODS[@]}")
fi

# 删除结果和tensor目录
rm -rf results tensors
mkdir -p eval  # 确保eval目录存在

# 遍历所有方法
for METHOD in "${METHODS[@]}"; do
    echo "========== 开始执行方法: $METHOD =========="
    
    if [ "$METHOD" == "mahalanobis" ]; then
        ./run_mahalanobis.sh
    elif [ "$METHOD" == "mahalanobis_multi" ]; then
        ./run_mahalanobis_multi.sh
    elif [ "$METHOD" == "mahalanobis_ood" ]; then
        ./run_mahalanobis_ood.sh
    elif [ "$METHOD" == "mahalanobis_plus" ]; then
        ./run_mahalanobis_plus.sh
    elif [ "$METHOD" == "igeood_plus" ]; then
        ./run_igeood_plus.sh
    else
        echo "未知方法: $METHOD, 跳过"
    fi
    
    echo "========== 方法 $METHOD 所有任务完成 =========="
    echo
done

echo "🎉 所有方法和任务执行完毕！"