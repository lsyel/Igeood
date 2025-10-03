import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import utils.data_and_nn_loader as dl
import utils.evaluation_metrics as em
import utils.file_manager as fm
from torch.autograd import Variable
from utils.logger import logger, timing

from src.ensemble_method import *
from src.estimators import (hidden_feature_estimator,
                            run_logits_centroid_estimator)
from src.measures import *

cudnn.benchmark = True


def get_prefix(
    cov_mat_ood,
    means_ood,
    logits_flag,
    per_class=False,
    distance=fr_distance_multivariate_gaussian,
):
    # File naming
    prefix = "igeoodfeatures"
    if cov_mat_ood is not None:
        prefix += "cov" + cov_mat_ood
    if logits_flag:
        prefix += "combinelogits"
    if per_class:
        prefix += "_per_class"
    if means_ood:
        prefix += "_means_ood"
    return prefix


def get_filename(prefix, temperature, eps):
    return "{}{:.1f}_{:.4f}.txt".format(prefix, temperature, eps)


@timing
def main(
    ensemble_method,
    nn_name,
    in_dataset_name,
    out_dataset_name,
    cov_mat_ood,
    means_ood,
    temperature,
    eps,
    batch_size,
    gpu,
    rewrite=False,
    logits_flag=True,
    per_class=False,
    distance=fr_distance_multivariate_gaussian,
):
    if cov_mat_ood == "same":
        cov_mat_ood = out_dataset_name
    elif cov_mat_ood == "ADV":
        cov_mat_ood += nn_name
    if out_dataset_name == "ADV":
        out_dataset_name += nn_name

    prefix = get_prefix(cov_mat_ood, means_ood, logits_flag, per_class, distance)
    filename = get_filename(prefix, temperature, eps)

    # in_dataset_name = dl.get_in_dataset_name(nn_name)
    fm.make_output_folders(nn_name, in_dataset_name)
    fm.make_output_folders(nn_name, out_dataset_name)

    # In scores
    f = fm.find_score_file(nn_name, in_dataset_name, filename)
    if rewrite is True or f is None:
        in_scores = igeoodwb_score(
            nn_name,
            in_dataset_name,
            batch_size,
            gpu,
            rewrite,
            cov_mat_ood,
            means_ood,
            logits_flag,
            temperature,
            eps,
            per_class=per_class,
            distance=distance,
        )
        fw = fm.make_score_file(nn_name, in_dataset_name, filename)
        fm.write_score_file(fw, in_scores)
        fw.close()
    else:
        in_scores = fm.load_score_file(nn_name, in_dataset_name, filename)

    # Out scores
    f = fm.find_score_file(nn_name, out_dataset_name, filename)
    if rewrite is True or f is None:
        out_scores = igeoodwb_score(
            nn_name,
            out_dataset_name,
            batch_size,
            gpu,
            rewrite,
            cov_mat_ood,
            means_ood,
            logits_flag,
            temperature,
            eps,
            per_class=per_class,
            distance=distance,
        )
        fw = fm.make_score_file(nn_name, out_dataset_name, filename)
        fm.write_score_file(fw, out_scores)
        fw.close()
    else:
        out_scores = fm.load_score_file(nn_name, out_dataset_name, filename)

    # Validation data
    if "val" in ensemble_method.__name__:
        val_dataset_name = ensemble_method.val_dataset_name
    elif "adv" in ensemble_method.__name__:
        val_dataset_name = "ADV" + nn_name
    else:
        val_dataset_name = out_dataset_name
    val_scores = None
    if "val" in ensemble_method.__name__ or "adv" in ensemble_method.__name__:
        val_filename = "{}{:.1f}_{:.4f}.txt".format(prefix, temperature, eps)
        fm.make_output_folders(nn_name, val_dataset_name)
        f = fm.find_score_file(nn_name, val_dataset_name, val_filename)
        if rewrite is True or f is None:
            val_scores = igeoodwb_score(
                nn_name,
                val_dataset_name,
                batch_size,
                gpu,
                rewrite,
                cov_mat_ood,
                means_ood,
                logits_flag,
                temperature,
                eps,
                per_class=per_class,
                distance=distance,
            )
            fw = fm.make_score_file(nn_name, val_dataset_name, val_filename)
            fm.write_score_file(fw, val_scores)
            fw.close()
        else:
            val_scores = fm.load_score_file(nn_name, val_dataset_name, val_filename)

    # Ensemble method
    # length = min(len(in_scores), len(out_scores))
    combine_in_score, combine_out_score = ensemble_method(
        in_scores, out_scores, val_scores
    )

    if np.isnan(combine_in_score.max()) or np.isnan(combine_out_score.max()):
        logger.warning("nan value found in score, returning without evaluating")
        return

    # Evaluation metric
    method_name = "{}_{}".format(filename.split(".txt")[0], ensemble_method.__name__)
    (
        fpr_at_tpr_in,
        fpr_at_tpr_out,
        detection,
        auroc,
        aupr_in,
        aupr_out,
    ) = em.print_metrics_and_info(
        combine_in_score,
        combine_out_score,
        nn_name,
        in_dataset_name,
        out_dataset_name,
        method_name,
        True,
        False,
        True,
    )

    # Save to results file
    method_name = "{}_{}".format(prefix, ensemble_method.__name__)
    fm.append_results_to_file(
        nn_name,
        out_dataset_name,
        method_name,
        eps,
        temperature,
        fpr_at_tpr_in,
        fpr_at_tpr_out,
        detection,
        auroc,
        aupr_in,
        aupr_out,
    )
    return fpr_at_tpr_in, detection, auroc, aupr_in


def igeoodwb_score(
    nn_name,
    dataset_name,
    batch_size,
    gpu,
    rewrite=False,
    cov_mat_ood=None,
    means_ood=None,
    logits_flag=True,
    temperature=1,
    eps=0,
    dataloader=None,
    per_class=False,
    distance=fr_distance_multivariate_gaussian,
):
    """计算IGEOOD检测分数的主函数
    
    参数:
        nn_name: 神经网络模型名称 (str)
        dataset_name: 待检测的数据集名称 (str)
        batch_size: 数据加载的批大小 (int)
        gpu: 使用的GPU ID，None表示使用CPU (int/None)
        rewrite: 是否重新计算特征统计量 (bool)
        cov_mat_ood: OOD协方差矩阵来源数据集 (str/None)
        means_ood: 是否使用OOD均值 (bool/None)
        logits_flag: 是否启用logits特征 (bool)
        temperature: 温度缩放参数 (float)
        eps: 对抗扰动强度 (float)
        per_class: 是否按类别处理协方差矩阵 (bool)
        distance: 距离度量函数 (callable)
    
    返回:
        numpy.ndarray: 包含所有样本检测分数的矩阵
    """
    # 获取原始训练数据集名称
    in_dataset_name = dl.get_in_dataset_name(nn_name)

    # === 模型加载 ===
    model = dl.load_pre_trained_nn(nn_name, gpu)
    model.eval()  # 设置为评估模式
    num_classes = dl.get_num_classes(in_dataset_name)

    # === 特征统计量加载 ===
    # 加载训练集隐藏层统计量
    sample_mean_in = dl.load_hidden_features_means(nn_name, in_dataset_name)
    cov_matrix_in = dl.load_hidden_features_cov(
        nn_name, in_dataset_name, True, None, per_class=per_class
    )
    multi_sample_mean_in = dl.load_hidden_features_multi_means(nn_name, in_dataset_name)
    
    # 当统计量不存在或需要重写时重新计算
    if cov_matrix_in is None or sample_mean_in is None or rewrite:
        hidden_feature_estimator(nn_name, in_dataset_name, batch_size, gpu, True, True, None)
        # 重新加载生成的统计量
        cov_matrix_in = dl.load_hidden_features_cov(nn_name, in_dataset_name, True, None, per_class=per_class)
        sample_mean_in = dl.load_hidden_features_means(nn_name, in_dataset_name)
        multi_sample_mean_in = dl.load_hidden_features_multi_means(nn_name, in_dataset_name)

    # === OOD协方差矩阵处理 ===
    cov_matrix_out = None
    if cov_mat_ood is not None:
        # 对抗样本特殊处理
        if cov_mat_ood == "ADV":
            cov_val_dataset_name = cov_mat_ood + nn_name
            cap = None  # 不使用采样上限
        else:
            cap = 3000  # 常规OOD数据集采样上限
            cov_val_dataset_name = cov_mat_ood
        
        logger.info(f"加载OOD协方差矩阵: {cov_val_dataset_name}")
        cov_matrix_out = dl.load_hidden_features_cov(nn_name, cov_val_dataset_name, True, cap)
        
        # 需要重新生成时调用特征估计器
        if cov_matrix_out is None or rewrite:
            hidden_feature_estimator(nn_name, cov_val_dataset_name, batch_size, gpu, False, True, cap)
            cov_matrix_out = dl.load_hidden_features_cov(nn_name, cov_val_dataset_name, True, cap)

    # === OOD均值处理 ===
    sample_mean_out = sample_mean_in  # 默认使用训练集均值
    if means_ood is not None:
        sample_mean_out = dl.load_hidden_features_means(nn_name, cov_val_dataset_name, cap=cap)
        if sample_mean_out is None or rewrite:
            hidden_feature_estimator(nn_name, cov_val_dataset_name, batch_size, gpu, False, True, cap)
            sample_mean_out = dl.load_hidden_features_means(nn_name, cov_val_dataset_name, cap=cap)

    # === Logits质心处理 ===
    logits_centroids = None
    if logits_flag:
        logits_centroids = dl.load_logits_centroid(nn_name, in_dataset_name)
        # 当需要重新计算时调用质心估计器
        if logits_centroids is None or rewrite:
            logger.info("重新计算logits质心...")
            logits_centroids, _, _, _ = run_logits_centroid_estimator(nn_name, gpu=gpu, batch_size=batch_size)

    logger.info("所有特征张量加载完成")

    # === 获取数据加载器 ===
    if dataloader is None:
        dataloader = dl.test_dataloader(dataset_name, in_dataset_name, batch_size=batch_size)

    # 调用核心检测算法
    return igeoodwb(
        model, dataloader, num_classes, sample_mean_in, cov_matrix_in, gpu,
        sample_mean_out, cov_matrix_out, logits_flag, temperature, eps,
        in_dataset_name, logits_centroids, multi_sample_mean_in, distance=distance
    )


def igeoodwb(
    model,
    dataloader,
    num_classes,
    sample_mean_in,
    cov_mat_in,
    gpu,
    sample_mean_out=None,
    cov_mat_out=None,
    logits_flag=True,
    temperature=None,
    eps=None,
    in_dataset_name=None,
    centroid_logits=None,
    multi_sample_mean_in=None,
    distance=fr_distance_multivariate_gaussian,
):
    """IGEOOD核心检测算法实现
    
    参数:
        model: 预训练好的分类模型
        dataloader: 测试数据加载器
        sample_mean_in: 训练集隐藏层特征均值
        cov_mat_in: 训练集协方差矩阵
        gpu: 使用的GPU ID
        sample_mean_out: OOD数据集均值（默认为训练集均值）
        cov_mat_out: OOD数据集协方差矩阵
        logits_flag: 是否启用logits特征检测
        temperature: 温度缩放参数
        eps: 对抗扰动强度
        multi_sample_mean_in: 多聚类中心均值（用于改进检测）
    
    返回:
        numpy.ndarray: 包含所有样本检测分数的矩阵
    """
    t0 = time.time()
    length = len(dataloader)
    model.eval()  # 确保模型处于评估模式
    n_layers = len(sample_mean_in)  # 获取隐藏层数量

    # 初始化分数存储结构
    igeoodfeature_scores = {i: [] for i in range(n_layers)}  # 各隐藏层特征分数
    igeoodlogits_scores = []  # logits特征分数

    # 遍历数据批次
    for batch_idx, data in enumerate(dataloader):
        # 处理输入数据（可能包含标签）
        if type(data) in [tuple, list]:
            data, _ = data  # 分离数据和标签
        # 数据转移到GPU（如果可用）
        if gpu is not None:
            data = data.cuda()
        # 设置需要梯度计算（用于对抗样本生成）
        data = Variable(data, requires_grad=True)
        
        # 获取模型输出（logits和隐藏层特征）
        logits, out_features = model.feature_list(data)

        # === Logits分数计算 ===
        if logits_flag:
            # 计算初始logits距离
            dist = igeoodlogits(logits, temperature, centroid_logits)

            # 对抗扰动处理（当eps>0时）
            if eps > 0:
                # 反向传播生成梯度
                loss = torch.mean(-dist)
                loss.backward()

                # 梯度二值化处理（生成对抗扰动方向）
                gradient = torch.ge(data.grad.data, 0)
                gradient = (gradient.float() - 0.5) * 2

                # 应用数据集特定的梯度变换（如图像归一化）
                gradient = dl.gradient_trasform(in_dataset_name)(gradient)
                with torch.no_grad():
                    # 生成对抗扰动样本
                    temp_inputs = torch.add(data, gradient, alpha=-eps)
                    # 获取扰动后输出
                    noised_logits, out_features = model.feature_list(temp_inputs)
                # 重新计算扰动后距离
                dist = igeoodlogits(noised_logits, temperature, centroid_logits)

            # 记录logits分数
            igeoodlogits_scores.extend(dist.detach().cpu().numpy().reshape(-1, 1))
        multi_flag = True
        # === 隐藏层特征处理 ===
        with torch.no_grad():
            # 遍历每个隐藏层
            for layer_idx, out_feature in enumerate(out_features):
                # 特征空间调整（展平后取均值）
                out_feature = out_feature.reshape(out_feature.shape[0], out_feature.shape[1], -1)
                out_feature = torch.mean(out_feature, 2)
                # 计算Fisher-Rao分数（单/多聚类中心）
                if multi_sample_mean_in is not None and multi_flag:
                    score1 = multi_igeoodfeature(
                        out_feature, multi_sample_mean_in, cov_mat_in, cov_mat_in,
                        layer_idx, num_classes, distance=distance
                    )
                else:
                    score1 = igeoodfeature(
                        out_feature, sample_mean_in, cov_mat_in, cov_mat_in,
                        layer_idx, num_classes, distance=distance
                    )
                
                # 取最小距离作为当前层分数
                score1, _ = torch.min(score1, dim=1)
                score1 = score1.detach().cpu().numpy().reshape(-1, 1)

                # OOD协方差矩阵处理（生成对比分数）
                if cov_mat_out is not None:
                    score2 = igeoodfeature(
                        out_feature, sample_mean_out, cov_mat_in, cov_mat_out,
                        layer_idx, num_classes, distance=distance
                    )
                    score2, _ = torch.min(score2, dim=1)
                    score2 = score2.detach().cpu().numpy().reshape(-1, 1)
                    # 合并两种分数
                    igeoodfeature_scores[layer_idx].extend(np.hstack([score1, score2]))
                else:
                    igeoodfeature_scores[layer_idx].extend(score1)

        # === 进度记录 ===
        if batch_idx % (int(length / 10) + 1) == 0 and batch_idx > 0:
            logger.info(
                "Batch {}/{}, {:.2f} seconds used.".format(
                    batch_idx + 1, length, time.time() - t0
                )
            )
            t0 = time.time()  # 重置计时器

    # === 分数整合 ===
    # 合并所有隐藏层分数
    scores = np.hstack(
        [np.asarray(igeoodfeature_scores[i], dtype=np.float32) for i in range(n_layers)]
    )
    # 合并logits分数（如果启用）
    if logits_flag:
        scores = np.hstack([scores, np.vstack(igeoodlogits_scores)])

    return scores
