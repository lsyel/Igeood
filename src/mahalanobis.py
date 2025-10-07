import numpy as np
import torch
import torch.backends.cudnn as cudnn
import utils.data_and_nn_loader as dl
import utils.evaluation_metrics as em
import utils.file_manager as fm
from torch.autograd import Variable
from utils.logger import logger, timing

from src.ensemble_method import *
from src.ensemble_method import MeanScore, WeightRegression
from src.estimators import hidden_feature_estimator

cudnn.benchmark = True


@timing
def main(
    ensemble_method,
    nn_name,
    in_dataset_name,
    out_dataset_name,
    eps,
    batch_size,
    gpu,
    rewrite=False,
    *args,
    **kwargs
):
    # Ensemble method
    mat_type = ""
    # File naming
    prefix = "mahalanobis"
    # 添加多质心启用标志
    use_multi_centroid = True  # 可以从参数传入
    # Model
    # in_dataset_name = dl.get_in_dataset_name(nn_name)
    num_classes = dl.get_num_classes(in_dataset_name)
    model = dl.load_pre_trained_nn(nn_name, gpu)
    model.eval()
    feature_list = dl.get_feature_list(model, gpu)
    num_features = len(feature_list)

    fm.make_output_folders(nn_name, in_dataset_name)
    fm.make_output_folders(nn_name, out_dataset_name)

    # Matrices
    inverse = dl.load_hidden_features_inv(nn_name, in_dataset_name)
    sample_mean = dl.load_hidden_features_means(nn_name, in_dataset_name)
    # 加载多质心统计量
    if use_multi_centroid:
        multi_sample_mean = dl.load_hidden_features_multi_means(nn_name, in_dataset_name)
        logger.debug("use multi centroid")
    else:
        multi_sample_mean = None
        logger.debug("not use multi centroid")
    if inverse is None or sample_mean is None:
        hidden_feature_estimator(nn_name, in_dataset_name, batch_size, gpu, True)
        inverse = dl.load_hidden_features_inv(nn_name, in_dataset_name)
        sample_mean = dl.load_hidden_features_means(nn_name, in_dataset_name)
        if use_multi_centroid:
            multi_sample_mean = dl.load_hidden_features_multi_means(nn_name, in_dataset_name)
            if multi_sample_mean is not None:
                logger.info("多质心数据结构:")
                for layer_idx, class_dict in multi_sample_mean.items():
                    logger.info(f"层 {layer_idx}: {len(class_dict)} 个类别")
                    for class_idx, centroids in class_dict.items():
                        logger.info(f"  类别 {class_idx}: {centroids.shape} 个质心")
    logger.info("tensors loaded")

    filename = "{}{}_{:.4f}.txt".format(prefix, mat_type, eps)

    # Get in scores
    f = fm.find_score_file(nn_name, in_dataset_name, filename)
    if rewrite is True or f is None:
        logger.info(
            "Calculating mahalanobis score for nn {} and dataset {}".format(
                nn_name, in_dataset_name
            )
        )
        in_dataloader = dl.test_dataloader(
            in_dataset_name, in_dataset_name, batch_size=batch_size
        )
        in_score = get_mahalanobis_score(
            model,
            in_dataloader,
            sample_mean,
            inverse,
            num_classes,
            nn_name,
            num_features,
            eps,
            gpu,
            multi_sample_mean=multi_sample_mean  # 传递多质心参数
        )
        fw = fm.make_score_file(nn_name, in_dataset_name, filename)
        fm.write_score_file(fw, in_score)
        fw.close()
    else:
        in_score = fm.load_score_file(nn_name, in_dataset_name, filename)

    # Get out scores
    f = fm.find_score_file(nn_name, out_dataset_name, filename)
    if rewrite or f is None:
        logger.info(
            "Calculating mahalanobis score for nn {} and dataset {}".format(
                nn_name, out_dataset_name
            )
        )
        out_dataloader = dl.test_dataloader(
            out_dataset_name, in_dataset_name, batch_size=batch_size
        )
        out_score = get_mahalanobis_score(
            model,
            out_dataloader,
            sample_mean,
            inverse,
            num_classes,
            nn_name,
            num_features,
            eps,
            gpu,
            multi_sample_mean=multi_sample_mean  # 传递多质心参数
        )
        fw = fm.make_score_file(nn_name, out_dataset_name, filename)
        fm.write_score_file(fw, out_score)
        fw.close()
    else:
        out_score = fm.load_score_file(nn_name, out_dataset_name, filename)

    # Validation data
    ensemble_name = ensemble_method.__name__
    if "val" in ensemble_method.__name__:
        val_dataset_name = ensemble_method.val_dataset_name
    elif "adv" in ensemble_method.__name__:
        val_dataset_name = "ADV" + nn_name
    else:
        val_dataset_name = out_dataset_name
    val_score = None
    if "val" in ensemble_method.__name__ or "adv" in ensemble_method.__name__:
        val_filename = "{}_{:.4f}.txt".format(prefix, eps)
        fm.make_output_folders(nn_name, val_dataset_name)
        f = fm.find_score_file(nn_name, val_dataset_name, val_filename)
        if rewrite is True or f is None:
            val_dataloader = dl.test_dataloader(
                val_dataset_name, in_dataset_name, batch_size=batch_size
            )
            val_score = get_mahalanobis_score(
                model,
                val_dataloader,
                sample_mean,
                inverse,
                num_classes,
                nn_name,
                num_features,
                eps,
                gpu,
                multi_sample_mean=multi_sample_mean  # 传递多质心参数
            )
            fw = fm.make_score_file(nn_name, val_dataset_name, val_filename)
            fm.write_score_file(fw, val_score)
            fw.close()
        else:
            val_score = fm.load_score_file(nn_name, val_dataset_name, val_filename)

    # Ensemble method
    method_name = "{}_{}".format(prefix, ensemble_name)
    # length = min(len(in_score), len(out_score))
    in_s, out_s = ensemble_method(in_score, out_score, val_score)

    if np.isnan(in_s.max()) or np.isnan(out_s.max()):
        logger.warning("nan value found in score, returning without evaluating")
        return

    (
        fpr_at_tpr_in,
        fpr_at_tpr_out,
        detection,
        auroc,
        aupr_in,
        aupr_out,
    ) = em.print_metrics_and_info(
        in_s,
        out_s,
        nn_name,
        in_dataset_name,
        out_dataset_name,
        method_name,
        True,
        False,
        True,
    )

    fm.append_results_to_file(
        nn_name,
        out_dataset_name,
        method_name,
        eps,
        1,
        fpr_at_tpr_in,
        fpr_at_tpr_out,
        detection,
        auroc,
        aupr_in,
        aupr_out,
    )
    return fpr_at_tpr_in, detection, auroc, aupr_in


def get_mahalanobis_score(
    model,
    dataloader,
    sample_mean,
    inverse,
    num_classes,
    nn_name,
    num_features,
    eps=0.0,
    gpu=None,
    multi_sample_mean=None  # 新增参数
):

    logger.info("get Mahalanobis scores")
    logger.info("noise magnitude: " + str(eps))
    mahalanobis = []
    # 只选择最后3层特征（可调整）
    last_num = 5
    selected_layers = range(max(0, num_features-last_num), num_features)
    logger.info(f"选择的层索引: {selected_layers}")
    for i in selected_layers:
        m = get_mahalanobis_layer_score(
            model,
            dataloader,
            num_classes,
            nn_name,
            sample_mean,
            inverse,
            i,
            eps,
            gpu,
            multi_sample_mean=multi_sample_mean  # 传递多质心参数
        )
        mahalanobis.append(m)
    mahalanobis = np.hstack(mahalanobis)
    return mahalanobis


def get_mahalanobis_layer_score(
    model,
    test_loader,
    num_classes,
    net_type,
    sample_mean,
    inverse,
    layer_index,
    eps,
    gpu,
    multi_sample_mean=None,  # 新增参数

) -> np.ndarray:
    """
    Compute the Mahalanobis confidence score
    return: Mahalanobis score from layer_index
    """
    model.eval()
    Mahalanobis = []
    for data in test_loader:
        if type(data) in [tuple, list]:
            data, _ = data
        if gpu is not None:
            data = data.cuda()
        data = Variable(data, requires_grad=True)

        out_features = model.intermediate_forward(data, layer_index)
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        # compute Mahalanobis score
        gaussian_score = compute_mahalanobis_distance(
            out_features, sample_mean, inverse, layer_index, num_classes, multi_sample_mean=multi_sample_mean  # 新增

        )
        class_score, _ = torch.max(gaussian_score, dim=1)

        if eps > 0:
            # Input_processing in the direction of the predicted class
            sample_pred = gaussian_score.max(1)[1]
            batch_sample_mean = torch.vstack(
                [sample_mean[layer_index][i] for i in sample_pred.cpu().numpy()]
            )
            zero_f = out_features - Variable(batch_sample_mean)
            pure_gau = (
                -0.5
                * torch.mm(
                    torch.mm(zero_f, Variable(inverse[layer_index])), zero_f.t()
                ).diag()
            )
            loss = torch.mean(-pure_gau)
            loss.backward()

            gradient = torch.ge(data.grad.data, 0)
            gradient = (gradient.float() - 0.5) * 2
            gradient = dl.gradient_trasform(dl.get_in_dataset_name(net_type))(gradient)
            tempInputs = torch.add(data.data, -eps, gradient)

            with torch.no_grad():
                noise_out_features = model.intermediate_forward(
                    Variable(tempInputs), layer_index
                )
            noise_out_features = noise_out_features.view(
                noise_out_features.size(0), noise_out_features.size(1), -1
            )
            noise_out_features = torch.mean(noise_out_features, 2)
            gaussian_score = compute_mahalanobis_distance(
                noise_out_features, sample_mean, inverse, layer_index, num_classes, multi_sample_mean=multi_sample_mean  # 新增

            )
                        # 再次取最大值
            class_score, _ = torch.max(gaussian_score, dim=1)

        Mahalanobis.extend(class_score.detach().cpu().numpy())

    Mahalanobis = np.asarray(Mahalanobis, dtype=np.float32).reshape(-1, 1)
    return Mahalanobis


def compute_mahalanobis_distance(
    out_features, 
    sample_mean, 
    inverse, 
    layer_index, 
    num_classes,
    multi_sample_mean=None
):
    if multi_sample_mean is not None and layer_index in multi_sample_mean:
        return multi_mahalanobis_distance(
            out_features,
            multi_sample_mean,
            inverse,
            layer_index,
            num_classes
        )
    else:
        # 单质心计算
        logger.info(f"使用单质心计算 (层 {layer_index})")

        gaussian_score = torch.zeros(out_features.size(0), num_classes, device=out_features.device)
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = torch.mm(torch.mm(zero_f, inverse[layer_index]), zero_f.t()).diag()
            gaussian_score[:, i] = -0.5 * term_gau  # 保持负号
        return gaussian_score


def predict(
    nn_name,
    dataloader,
    batch_size,
    gpu,
    eps=0,
):
    logger.info("get Mahalanobis scores")
    logger.info("noise magnitude: " + str(eps))
    mahalanobis = []

    # model
    in_dataset_name = dl.get_in_dataset_name(nn_name)
    num_classes = dl.get_num_classes(in_dataset_name)
    model = dl.load_pre_trained_nn(nn_name, gpu)
    model.eval()

    feature_list = dl.get_feature_list(model, gpu)
    num_features = len(feature_list)

    # Matrices
    inverse = dl.load_hidden_features_inv(nn_name, in_dataset_name)
    sample_mean = dl.load_hidden_features_means(nn_name, in_dataset_name)

    if inverse is None or sample_mean is None:
        hidden_feature_estimator(nn_name, in_dataset_name, batch_size, gpu, True)
        inverse = dl.load_hidden_features_inv(nn_name, in_dataset_name)
        sample_mean = dl.load_hidden_features_means(nn_name, in_dataset_name)

    for i in range(num_features):
        m = get_mahalanobis_layer_score(
            model,
            dataloader,
            num_classes,
            nn_name,
            sample_mean,
            inverse,
            i,
            eps,
            gpu,
        )
        mahalanobis.append(m)
    mahalanobis = np.hstack(mahalanobis)
    return mahalanobis
def multi_mahalanobis_distance(
    out_features,
    multi_sample_mean,
    inverse,
    layer_idx,
    num_classes,
    eps=1e-6
):
    """计算多质心Mahalanobis距离，返回负距离分数"""
    batch_size = out_features.shape[0]
    device = out_features.device
    scores = torch.zeros(batch_size, num_classes, device=device)
    
    # 遍历所有类别
    for class_idx in range(num_classes):
        # 跳过没有质心的类别
        if class_idx not in multi_sample_mean[layer_idx]:
            # 使用全局均值作为后备
            all_centroids = torch.cat([c for c in multi_sample_mean[layer_idx].values()])
            global_mean = torch.mean(all_centroids, dim=0)
            centroids = global_mean.unsqueeze(0)
        else:
            centroids = multi_sample_mean[layer_idx][class_idx].to(device)
        
        # 向量化计算距离平方
        deviation = out_features.unsqueeze(1) - centroids.unsqueeze(0)  # [batch_size, K, D]
        term1 = torch.einsum('bki,ij->bkj', deviation, inverse[layer_idx])  # [batch_size, K, D]
        term2 = torch.einsum('bki,bki->bk', term1, deviation)  # [batch_size, K]
        
        # 取最小距离平方
        min_dist_sq, _ = torch.min(term2, dim=1)  # [batch_size]
        
        # 返回负距离分数（与单质心一致）
        scores[:, class_idx] = -0.5 * min_dist_sq
    
    return scores