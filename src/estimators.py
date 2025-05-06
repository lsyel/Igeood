import os

import numpy as np
import sklearn
import sklearn.covariance
import torch
import utils.data_and_nn_loader as dl
from torch.autograd import Variable
from utils.logger import logger

from src.measures import fisher_rao_logits_distance

ROOT = dl.ROOT


# def run_logits_centroid_estimator(
#     nn_name, epochs=100, batch_size=128, gpu=None, lr=0.01, *args, **kwargs
# ):
#     model = dl.load_pre_trained_nn(nn_name, gpu)
#     in_dataset_name = dl.get_in_dataset_name(nn_name)
#     dataloader = dl.train_dataloader(
#         in_dataset_name, in_dataset_name, batch_size=batch_size
#     )
#     logger.info("diagonal matrix initialization")
#     init_tensor = torch.eye(dl.get_num_classes(in_dataset_name))
#     distance = fisher_rao_logits_distance
#     logits, targets = dl.pred_loop(model, dataloader, gpu)
#     if gpu is not None:
#         init_tensor = init_tensor.cuda(gpu)
#         logits = logits.cuda(gpu)
#         targets = targets.cuda(gpu)

#     centroid, epoch_loss = logits_centroid_estimator(
#         logits, targets, init_tensor, distance, epochs, lr, *args, **kwargs
#     )
#     # save tensor
#     os.makedirs("{}/tensors".format(ROOT), exist_ok=True)
#     filename = "{}/tensors/centroid_logits_{}_{}.pt".format(
#         ROOT, nn_name, in_dataset_name
#     )
#     torch.save(centroid, filename)
#     logger.info("first loss is approx {}".format(epoch_loss[0][0]))
#     logger.info("last loss is approx {}".format(epoch_loss[0][-1]))
#     return centroid, epoch_loss, logits, targets


# def logits_centroid_estimator(
#     logits, targets, init_tensor, distance, epochs, lr, *args, **kwargs
# ):
#     n_classes = init_tensor.shape[1]
#     centroid = [
#         Variable(init_tensor[i].reshape(1, -1), requires_grad=True)
#         for i in range(n_classes)
#     ]
#     logger.info("Initialized centroid: {}".format(centroid))

#     epoch_loss = [[] for _ in range(n_classes)]
#     for epoch in range(epochs):
#         for c in range(n_classes):
#             filt = targets == c
#             if filt.sum() == 0:
#                 continue

#             d = distance(logits[filt].detach(), centroid[c], *args, **kwargs)
#             loss = torch.mean(d)
#             epoch_loss[c].append(loss.item())
#             loss.backward()
#             # optimizer.step()
#             with torch.no_grad():
#                 aux = centroid[c]
#                 tmp = aux - lr * aux.grad
#                 centroid[c].copy_(tmp)

#     logger.info("converged centroid: {}".format(centroid))
#     return torch.vstack(centroid), epoch_loss



from sklearn.cluster import KMeans
def run_logits_centroid_estimator(
    nn_name, epochs=100, batch_size=128, gpu=None, lr=0.01, *args, **kwargs
):
    model = dl.load_pre_trained_nn(nn_name, gpu)
    in_dataset_name = dl.get_in_dataset_name(nn_name)
    dataloader = dl.train_dataloader(
        in_dataset_name, in_dataset_name, batch_size=batch_size
    )
    logger.info("KMeans initialization for five centroids per class")
    
    logits, targets = dl.pred_loop(model, dataloader, gpu)
    if gpu is not None:
        logits = logits.cuda(gpu)
        targets = targets.cuda(gpu)
    
    n_classes = dl.get_num_classes(in_dataset_name)
    n_centroids = 5  # 修改质心数量为5
    init_tensors = []
    
    for c in range(n_classes):
        mask = targets == c
        logits_c = logits[mask]
        
        if logits_c.size(0) < n_centroids:
            # 处理样本不足的情况
            if logits_c.size(0) == 0:
                # 使用全局统计信息生成随机质心
                global_mean = logits.mean(dim=0)
                global_std = logits.std(dim=0)
                centroid = torch.randn(n_centroids, logits.shape[1], 
                                      device=logits.device) * global_std + global_mean
            else:
                # 复制现有样本并添加多样性噪声
                repeat_times = (n_centroids // logits_c.size(0)) + 1
                centroid = logits_c.repeat(repeat_times, 1)[:n_centroids]
                noise = torch.randn_like(centroid) * logits_c.std(dim=0) * 0.1
                centroid += noise
        else:
            # 使用KMeans找五个质心
            logits_np = logits_c.cpu().numpy()
            kmeans = KMeans(n_clusters=n_centroids, random_state=0).fit(logits_np)
            centroid = torch.tensor(kmeans.cluster_centers_, 
                                   dtype=logits.dtype, 
                                   device=logits.device)
        
        init_tensors.append(centroid)
    
    init_tensor = torch.stack(init_tensors)  # 形状变为 (n_classes, 5, feature_dim)
    
    centroid, epoch_loss = logits_centroid_estimator(
        logits, targets, init_tensor, fisher_rao_logits_distance, epochs, lr, *args, **kwargs
    )
    
    os.makedirs("{}/tensors".format(ROOT), exist_ok=True)
    filename = "{}/tensors/centroid_logits_{}_{}_x5.pt".format(ROOT, nn_name, in_dataset_name)
    torch.save(centroid, filename)
    return centroid, epoch_loss, logits, targets


def logits_centroid_estimator(
    logits, targets, init_tensor, distance, epochs, lr, *args, **kwargs
):
    n_classes, n_centroids, feature_dim = init_tensor.shape  # 现在n_centroids=5
    centroid = []
    
    # 初始化每个类的5个质心
    for c in range(n_classes):
        centroid_c = [
            torch.nn.Parameter(init_tensor[c, k].clone().view(-1), requires_grad=True)
            for k in range(n_centroids)
        ]
        centroid.append(centroid_c)
    
    logger.info(f"Initialized centroids: {len(centroid)} classes, each with {n_centroids} centroids")
    epoch_loss = [[[] for _ in range(n_centroids)] for _ in range(n_classes)]
    
    for epoch in range(epochs):
        for c in range(n_classes):
            filt = targets == c
            if filt.sum() == 0:
                continue
                
            logits_c = logits[filt].detach()  # (num_samples, feature_dim)
            
            # 计算到所有5个质心的距离
            dists = torch.stack([
                distance(logits_c, centroid[c][k].unsqueeze(0), *args, **kwargs)
                for k in range(n_centroids)
            ], dim=1)  # 形状变为 (num_samples, 5)
            
            assignments = dists.argmin(dim=1)  # 获取最近质心的索引
            
            # 更新每个质心
            for k in range(n_centroids):
                mask = assignments == k
                if mask.sum() == 0:
                    # 动态调整：若连续5次无分配，随机重置
                    if (epoch % 5 == 0) and (len(epoch_loss[c][k]) > 0):
                        with torch.no_grad():
                            centroid[c][k].data = logits[targets == c].mean(dim=0) + torch.randn_like(centroid[c][k]) * 0.1
                    continue
                    
                selected_logits = logits_c[mask]
                centroid_tensor = centroid[c][k].unsqueeze(0)
                
                # 计算损失
                dist = distance(selected_logits, centroid_tensor, *args, **kwargs)
                loss = torch.mean(dist)
                
                # 梯度更新
                if centroid[c][k].grad is not None:
                    centroid[c][k].grad.zero_()
                loss.backward()
                
                with torch.no_grad():
                    centroid[c][k].data -= lr * centroid[c][k].grad.data
                
                epoch_loss[c][k].append(loss.item())
    
    # 过滤无效质心（阈值设为总epoch数的1/3）
    min_valid_epochs = epochs // 3
    final_centroids = []
    for c in range(n_classes):
        valid_centroids = []
        for k in range(n_centroids):
            # 如果该质心更新次数达标则保留
            if len(epoch_loss[c][k]) >= min_valid_epochs:
                valid_centroids.append(centroid[c][k].detach())
        
        # 至少保留一个质心（优先保留第一个）
        if not valid_centroids:
            valid_centroids.append(centroid[c][0].detach())
        
        final_centroids.append(torch.stack(valid_centroids))  # 形状 (有效质心数, 特征维度)
    
    return torch.vstack(final_centroids), epoch_loss

def get_hidden_features_sample(model, dataloader, gpu, cap=None):
    model.eval()
    feature_list = dl.get_feature_list(model, gpu)
    num_hidden_features = len(feature_list)

    hidden_feature_sample = {i: {} for i in range(num_hidden_features)}
    logger.info("cap is {}".format(cap))
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
                    hidden_feature_sample[j][index].append(feature[b].reshape(1, -1))

            if cap is not None and batch_size * batch_idx >= cap:
                logger.warning("cap of {} exceeded, breaking...".format(cap))
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

from sklearn.cluster import KMeans

def multi_get_hidden_feat_sample_mean(hidden_feature_sample, n_clusters=5):
    num_features = len(hidden_feature_sample)
    sample_class_mean = {}

    for i in range(num_features):
        feature_dict = hidden_feature_sample[i]
        sample_class_mean[i] = {}

        for c, samples in feature_dict.items():
            # 跳过空样本
            if len(samples) == 0:
                continue
                
            # 转换为 numpy 格式供 K-means 使用
            samples_np = samples.cpu().numpy()
            
            # 执行 K-means 聚类
            if len(samples_np) >= n_clusters:
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                kmeans.fit(samples_np)
                cluster_centers = kmeans.cluster_centers_  # 形状 [n_clusters, feature_dim]
            else:
                # 样本不足时直接使用所有样本作为"中心"
                cluster_centers = samples_np

            # 转换为 tensor 并保存
            sample_class_mean[i][c] = torch.from_numpy(cluster_centers).float()

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
            group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
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
    # test features
    cap_str = "_{}".format(cap) if cap is not None else ""
    in_dataset_name = dl.get_in_dataset_name(nn_name)
    if dataset_name is None:
        dataset_name = in_dataset_name
    if train:
        dataloader = dl.train_dataloader(
            dataset_name, in_dataset_name, batch_size=batch_size
        )
    else:
        dataloader = dl.test_dataloader(dataset_name, in_dataset_name, batch_size=100)
    model = dl.load_pre_trained_nn(nn_name, gpu)

    sample = get_hidden_features_sample(model, dataloader, gpu, cap)
    means = get_hidden_feat_sample_mean(sample)
    inv, cov = get_hidden_feat_cov_inv_matrix(sample, means, diag, *args, **kwargs)
    multi_means = multi_get_hidden_feat_sample_mean(sample, n_clusters=5)
    # Save hidden features means
    os.makedirs("{}/tensors/{}/{}".format(ROOT, nn_name, dataset_name), exist_ok=True)
    filename = "{}/tensors/{}/{}/hidden_features_means{}.pt".format(
        ROOT, nn_name, dataset_name, cap_str
    )
    logger.info("saving file {}".format(filename))
    torch.save(means, filename)
    filename = "{}/tensors/{}/{}/hidden_features_multi_means{}.pt".format(
        ROOT, nn_name, dataset_name, cap_str
    )
    
    logger.info("saving file {}".format(filename))
    torch.save(multi_means, filename)
    mat_type = ""
    if diag:
        mat_type = "_diag"

    # Save hidden features inv cov
    filename = "{}/tensors/{}/{}/hidden_features{}_invs_cov{}.pt".format(
        ROOT, nn_name, dataset_name, mat_type, cap_str
    )
    logger.info("saving file {}".format(filename))
    torch.save(inv, filename)

    # Save hidden feature cov
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
        hidden_feature_estimator(nn_name, out_dataset_name, train=False, cap=1000)
    run_logits_centroid_estimator(nn_name, epochs=100, batch_size=128, gpu=None)
