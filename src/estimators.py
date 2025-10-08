import os

import numpy as np
import sklearn
import sklearn.covariance
import torch
import utils.data_and_nn_loader as dl
from torch.autograd import Variable
from utils.kmeans import *
from utils.logger import logger

from src.measures import fisher_rao_logits_distance

ROOT = dl.ROOT


def run_logits_centroid_estimator(
    nn_name, epochs=100, batch_size=128, gpu=None, lr=0.01, *args, **kwargs
):
    model = dl.load_pre_trained_nn(nn_name, gpu)
    in_dataset_name = dl.get_in_dataset_name(nn_name)
    dataloader = dl.train_dataloader(
        in_dataset_name, in_dataset_name, batch_size=batch_size
    )
    logger.info("diagonal matrix initialization")
    init_tensor = torch.eye(dl.get_num_classes(in_dataset_name))
    distance = fisher_rao_logits_distance
    logits, targets = dl.pred_loop(model, dataloader, gpu)
    if gpu is not None:
        init_tensor = init_tensor.cuda(gpu)
        logits = logits.cuda(gpu)
        targets = targets.cuda(gpu)

    centroid, epoch_loss = logits_centroid_estimator(
        logits, targets, init_tensor, distance, epochs, lr, *args, **kwargs
    )
    # save tensor
    os.makedirs("{}/tensors".format(ROOT), exist_ok=True)
    filename = "{}/tensors/centroid_logits_{}_{}.pt".format(
        ROOT, nn_name, in_dataset_name
    )
    torch.save(centroid, filename)
    logger.info("first loss is approx {}".format(epoch_loss[0][0]))
    logger.info("last loss is approx {}".format(epoch_loss[0][-1]))
    return centroid, epoch_loss, logits, targets


def logits_centroid_estimator(
    logits, targets, init_tensor, distance, epochs, lr, *args, **kwargs
):
    n_classes = init_tensor.shape[1]
    centroid = [
        Variable(init_tensor[i].reshape(1, -1), requires_grad=True)
        for i in range(n_classes)
    ]
    logger.info("Initialized centroid: {}".format(centroid))

    epoch_loss = [[] for _ in range(n_classes)]
    for epoch in range(epochs):
        for c in range(n_classes):
            filt = targets == c
            if filt.sum() == 0:
                continue

            d = distance(logits[filt].detach(), centroid[c], *args, **kwargs)
            loss = torch.mean(d)
            epoch_loss[c].append(loss.item())
            loss.backward()
            # optimizer.step() todo修改下降方法
            with torch.no_grad():
                aux = centroid[c]
                tmp = aux - lr * aux.grad
                centroid[c].copy_(tmp)

    logger.info("converged centroid: {}".format(centroid))
    return torch.vstack(centroid), epoch_loss


def get_hidden_features_sample(model, dataloader, gpu, cap=None):
    model.eval()
    feature_list = dl.get_feature_list(model, gpu)
    num_hidden_features = len(feature_list)

    hidden_feature_sample = {i: {} for i in range(num_hidden_features)}
    logger.info("cap is {}".format(cap))
    sample_count = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            batch_size = data.shape[0]
            if gpu is not None:
                data = data.cuda(gpu)
                target = target.cuda(gpu)

            data = Variable(data)
            output, features = model.feature_list(data)

            for i, feature in enumerate(features):
                features[i] = torch.mean(
                    feature.reshape(feature.shape[0], feature.shape[1], -1), 2
                )

            pred = output.max(1)[1]
            # construct the sample matrix
            for b in range(batch_size):
                label = pred[b]
                index = int(label.cpu().numpy())
                for j, feature in enumerate(features):
                    if index not in hidden_feature_sample[j].keys():
                        hidden_feature_sample[j][index] = []
                    hidden_feature_sample[j][index].append(
                        feature[b].reshape(1, -1))
            sample_count += batch_size

            if cap is not None and batch_size * (batch_idx+1) >= cap:
                logger.warning("cap of {} exceeded, breaking...".format(cap))
                logger.info("采样{}个样本,共计{}个样本,ood rate={:.4f}".format(sample_count, batch_size*len(dataloader), sample_count/(batch_size*len(dataloader))))
                break

    for j in range(num_hidden_features):
        for c, sample in hidden_feature_sample[j].items():
            hidden_feature_sample[j][c] = torch.vstack(sample)

    return hidden_feature_sample


def get_hidden_feat_sample_mean(hidden_feature_sample):
    # sample mean per feature
    num_features = len(hidden_feature_sample)
    sample_class_mean = {}
    for i in range(num_features):
        x = hidden_feature_sample[i]
        sample_class_mean[i] = {
            c: torch.mean(sample, 0).reshape(1, -1)
            for c, sample in x.items()
            if len(sample) > 0
        }

    return sample_class_mean


def multi_get_hidden_feat_sample_mean(hidden_feature_sample, max_clusters=5, n_jobs=4):
    num_features = len(hidden_feature_sample)
    sample_class_mean = {}

    for i in range(num_features):
        feature_dict = hidden_feature_sample[i]
        sample_class_mean[i] = {}

        # 安全处理空类别
        if not feature_dict:
            continue

        # 并行处理每个类别
        results = Parallel(n_jobs=n_jobs)(
            delayed(kmeans_precess_class)(c, samples, max_clusters)
            for c, samples in feature_dict.items() if len(samples) > 0
        )

        for c, centers in results:
            sample_class_mean[i][c] = torch.from_numpy(centers).float()

    return sample_class_mean


def get_hidden_feat_cov_inv_matrix(
    hidden_feature_sample, sample_class_mean, diag=False, eps=1e-6
):
    num_features = len(hidden_feature_sample)

    inv = {}
    cov = {}
    for i in range(num_features):
        mu = sample_class_mean[i]
        X = [x - mu[c] for c, x in hidden_feature_sample[i].items()]
        X = torch.vstack(X)
        X = X.cpu().numpy()
        if diag:
            # diagonal covariance matrix estimation
            temp_cov_mat = [
                torch.from_numpy(np.cov(X[:, col].T, rowvar=False)).float()
                for col in range(X.shape[1])
            ]
            cov[i] = torch.diag(torch.tensor(temp_cov_mat))
            inv[i] = torch.diag(1 / (torch.tensor(temp_cov_mat) + 1e-12))
        else:
            # Maximum likelihood covariance estimator
            group_lasso = sklearn.covariance.EmpiricalCovariance(
                assume_centered=False)
            # find pseudo-inverse
            group_lasso.fit(X)
            inv[i] = torch.from_numpy(group_lasso.precision_).float()
            cov[i] = torch.from_numpy(group_lasso.covariance_).float()

    return inv, cov


def hidden_feature_estimator(
    nn_name,
    dataset_name=None,
    batch_size=512,
    gpu=None,
    train=True,
    diag=False,
    cap=None,
    *args,
    **kwargs
):
    """隐藏层特征估计器
    参数:
        nn_name: 神经网络模型名称
        dataset_name: 使用的数据集名称（默认为模型训练数据集）
        batch_size: 数据加载的批大小
        gpu: 使用的GPU ID（None表示使用CPU）
        train: 是否使用训练数据集
        diag: 是否使用对角协方差矩阵
        cap: 采样数量上限（None表示不限制）
    返回:
        tuple: (均值, 协方差逆矩阵, 协方差矩阵)
    """
    # 生成文件后缀（用于带采样上限的情况）
    cap_str = "_{}".format(cap) if cap is not None else ""
    # 获取模型对应的原始训练数据集名称
    in_dataset_name = dl.get_in_dataset_name(nn_name)
    # 设置默认数据集名称
    if dataset_name is None:
        dataset_name = in_dataset_name

    # 选择训练/测试数据加载器
    if train:
        dataloader = dl.train_dataloader(
            dataset_name, in_dataset_name, batch_size=batch_size
        )
    else:
        # 测试时使用固定batch_size=100
        dataloader = dl.test_dataloader(
            dataset_name, in_dataset_name, batch_size=100)

    # 加载预训练模型
    model = dl.load_pre_trained_nn(nn_name, gpu)

    # 获取隐藏层特征样本
    sample = get_hidden_features_sample(model, dataloader, gpu, cap)
    # 计算单聚类中心均值
    means = get_hidden_feat_sample_mean(sample)
    # 计算协方差矩阵及其逆矩阵
    inv, cov = get_hidden_feat_cov_inv_matrix(
        sample, means, diag, *args, **kwargs)
    # 计算多聚类中心均值（5个聚类中心）
    # multi_means = multi_get_hidden_feat_sample_mean(sample, max_clusters=10)
    # 评估聚类质量
    # cluster_report = evaluate_clustering_quality(multi_means)
    # print_clustering_report(cluster_report)
    # 创建保存目录
    os.makedirs("{}/tensors/{}/{}".format(ROOT,
                nn_name, dataset_name), exist_ok=True)

    # 保存单中心均值
    filename = "{}/tensors/{}/{}/hidden_features_means{}.pt".format(
        ROOT, nn_name, dataset_name, cap_str
    )
    logger.info("saving file {}".format(filename))
    torch.save(means, filename)

    # 保存多中心均值（5个聚类）
    filename = "{}/tensors/{}/{}/hidden_features_multi_means{}.pt".format(
        ROOT, nn_name, dataset_name, cap_str
    )
    logger.info("saving file {}".format(filename))
    # torch.save(multi_means, filename)

    # 处理协方差矩阵类型标记
    mat_type = ""
    if diag:
        mat_type = "_diag"  # 对角协方差矩阵标记

    # 保存协方差逆矩阵
    filename = "{}/tensors/{}/{}/hidden_features{}_invs_cov{}.pt".format(
        ROOT, nn_name, dataset_name, mat_type, cap_str
    )
    logger.info("saving file {}".format(filename))
    torch.save(inv, filename)

    # 保存协方差矩阵
    filename = "{}/tensors/{}/{}/hidden_features{}_cov_mat{}.pt".format(
        ROOT, nn_name, dataset_name, mat_type, cap_str
    )
    logger.info("saving file {}".format(filename))
    torch.save(cov, filename)

    return (means, inv, cov)


if __name__ == "__main__":
    nn_name = "densenet10"
    in_dataset_name = dl.get_in_dataset_name(nn_name)
    init_tensor = torch.eye(dl.get_num_classes(in_dataset_name))
    distance = fisher_rao_logits_distance
    epochs = 100
    out_dataset_names = [
        "SVHN",
        "Imagenet_resize",
        "iSUN",
        "LSUN_resize",
        "CIFAR100",
        "Textures",
        "Places365",
        "Chars74K",
        "gaussian_noise_dataset",
    ]
    for out_dataset_name in out_dataset_names:
        hidden_feature_estimator(
            nn_name, out_dataset_name, train=False, cap=1000)
    run_logits_centroid_estimator(
        nn_name, epochs=100, batch_size=128, gpu=None)
def hidden_feature_estimator_ood(
    nn_name,
    dataset_name,
    batch_size=10,
    gpu=None,
    diag=False,
    *args,
    **kwargs
):
    """OOD 隐藏层特征估计器（计算全局统计量）
    参数:
        nn_name: 神经网络模型名称
        dataset_name: OOD 数据集名称
        batch_size: 数据加载的批大小
        gpu: 使用的GPU ID（None表示使用CPU）
        cap: 采样数量上限（None表示不限制）
        diag: 是否使用对角协方差矩阵
    返回:
        tuple: (全局均值, 协方差逆矩阵, 协方差矩阵)
    """

    # 获取模型对应的原始训练数据集名称
    in_dataset_name = dl.get_in_dataset_name(nn_name)

    # 使用测试数据加载器（OOD 数据没有训练集）
    dataloader = dl.test_dataloader(
        dataset_name, in_dataset_name, batch_size=batch_size)
    # 生成文件后缀（用于带采样上限的情况）
    cap_rate = 0.01  # 采样上限比例
    cap = int(len(dataloader)*batch_size*cap_rate)
    cap_str = "_{}".format(cap) if cap is not None else ""
    # 加载预训练模型
    model = dl.load_pre_trained_nn(nn_name, gpu)

    # 获取隐藏层特征样本
    sample = get_hidden_features_sample(model, dataloader, gpu, cap)
    # 计算全局均值（不按类别）
    global_means = get_global_feat_sample_mean(sample)
    
    # 计算协方差矩阵及其逆矩阵
    inv, cov = get_global_feat_cov_inv_matrix(
        sample, global_means, diag, *args, **kwargs)
    
    # 创建保存目录
    os.makedirs("{}/tensors/{}/{}".format(ROOT,
                nn_name, dataset_name), exist_ok=True)

    # 保存全局均值
    filename = "{}/tensors/{}/{}/hidden_features_global_means{}.pt".format(
        ROOT, nn_name, dataset_name, cap_str
    )
    logger.info("saving file {}".format(filename))
    torch.save(global_means, filename)

    # 处理协方差矩阵类型标记
    mat_type = ""
    if diag:
        mat_type = "_diag"  # 对角协方差矩阵标记

    # 保存协方差逆矩阵
    filename = "{}/tensors/{}/{}/hidden_features{}_invs_cov{}.pt".format(
        ROOT, nn_name, dataset_name, mat_type, cap_str
    )
    logger.info("saving file {}".format(filename))
    torch.save(inv, filename)

    # 保存协方差矩阵
    filename = "{}/tensors/{}/{}/hidden_features{}_cov_mat{}.pt".format(
        ROOT, nn_name, dataset_name, mat_type, cap_str
    )
    logger.info("saving file {}".format(filename))
    torch.save(cov, filename)

    # 在返回前确保张量在正确设备上
    if gpu is not None:
        # 将全局均值移动到GPU
        for layer_idx, mean_tensor in global_means.items():
            if mean_tensor is not None:
                global_means[layer_idx] = mean_tensor.cuda(gpu)
        
        # 将协方差逆矩阵移动到GPU
        for layer_idx, inv_tensor in inv.items():
            if inv_tensor is not None:
                inv[layer_idx] = inv_tensor.cuda(gpu)
    
    return (global_means, inv, cov)


def get_global_feat_sample_mean(sample):
    """计算隐藏层特征的全局均值（不按类别）"""
    global_means = {}
    for layer_idx, class_samples in sample.items():
        # 合并所有类别的样本
        all_features = []
        for cls_samples in class_samples.values():
            all_features.append(cls_samples)
        
        if all_features:
            # 计算全局均值
            all_features = torch.cat(all_features, dim=0)
            global_mean = torch.mean(all_features, dim=0, keepdim=True)
            global_means[layer_idx] = global_mean
        else:
            global_means[layer_idx] = None
    return global_means
def get_global_feat_cov_inv_matrix(
    sample, 
    global_means, 
    diag=False, 
    eps=1e-6
):
    """计算全局协方差矩阵及其逆矩阵（适用于 OOD 数据）
    
    参数:
        sample: 隐藏层特征样本，结构为 {层索引: {类别: 特征张量}}
        global_means: 全局均值，结构为 {层索引: 均值向量}
        diag: 是否使用对角协方差矩阵
        eps: 正则化参数
    
    返回:
        tuple: (协方差逆矩阵, 协方差矩阵)
    """
    num_features = len(sample)
    inv = {}
    cov = {}
    
    for i in range(num_features):
        # 获取该层所有特征
        all_features = []
        for cls, features in sample[i].items():
            all_features.append(features)
        
        if not all_features:
            continue
            
        # 拼接所有特征
        X = torch.cat(all_features, dim=0)
        X = X.cpu().numpy()
        
        # 减去全局均值
        mean = global_means[i].cpu().numpy()
        X = X - mean
        
        # 计算协方差矩阵
        if diag:
            # 对角协方差矩阵
            var = np.var(X, axis=0)
            cov[i] = torch.diag(torch.from_numpy(var).float())
            inv[i] = torch.diag(1 / (torch.from_numpy(var).float() + eps))
        else:
            # 完整协方差矩阵
            cov_matrix = np.cov(X, rowvar=False)
            cov_matrix += eps * np.identity(cov_matrix.shape[0])  # 正则化
            
            # 计算逆矩阵
            try:
                inv_matrix = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                # 如果奇异，使用伪逆
                inv_matrix = np.linalg.pinv(cov_matrix)
                
            cov[i] = torch.from_numpy(cov_matrix).float()
            inv[i] = torch.from_numpy(inv_matrix).float()
    
    return inv, cov