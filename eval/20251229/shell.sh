 rm -rf results/scores/ tensors && python /root/wzhdesign/Igeood/eval.py igeood_plus --nn resnet --in-dataset ustc_task_2_in --out-dataset ustc_task_2_out --temperature 1.0 --epsilon 0 -gpu 0 > eval/20251229/igeood_task_2.log

  rm -rf results/scores/ tensors && python /root/wzhdesign/Igeood/eval.py mahalanobis --nn resnet --in-dataset ustc_task_2_in --out-dataset ustc_task_2_out --temperature 1.0 --epsilon 0 -gpu 0 > eval/20251229/mahalanobis_task_2.log

rm -rf results/scores/ tensors && python /root/wzhdesign/Igeood/eval.py mahalanobis_multi --nn resnet --in-dataset ustc_task_2_in --out-dataset ustc_task_2_out --temperature 1.0 --epsilon 0 -gpu 0 >> eval/20251229/mahalanobis_multi_task_2.log