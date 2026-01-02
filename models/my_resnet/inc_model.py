from matplotlib import pyplot as plt
import numpy as np
from models.my_resnet.inc_net import IncrementalNet
import torch
import logging
import sys

class IncModel:
    def __init__(self, model_path, num_classes, device="cuda"):
        # 初始化日志系统
        self._setup_logging()
        self.logger = logging.getLogger('IncModel')
        
        self.device = device
        self.model = self._load_model(model_path, num_classes)
        self.model.eval()
    
    def _setup_logging(self):
        """配置日志系统"""
        # 创建专用日志器
        logger = logging.getLogger('IncModel')
        logger.setLevel(logging.INFO)
        
        # 如果已有处理器，先移除
        if logger.hasHandlers():
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 设置详细格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    def _load_model(self, model_path, num_classes):
            self.logger.info(f"开始加载模型: {model_path}")
            # 1. 加载 checkpoint 字典
            checkpoint = torch.load(model_path, map_location=self.device)
            moe_experts = checkpoint.get('moe_experts', 1)  # 默认为1

            # 2. 初始化模型 (此时 old_gate 的大小会错误地等于 moe_experts)
            model = IncrementalNet(num_classes, use_moe=True, moe_experts=moe_experts)
            
            # ==================== 修复开始 ====================
            # 3. 动态调整 old_gate 的结构以匹配 checkpoint
            state_dict = checkpoint['network_state_dict']
            
            # 这里的 key 需要根据你的报错信息来，你的报错是 convnet.moe_layer.old_gate.8.weight
            old_gate_weight_key = 'convnet.moe_layer.old_gate.8.weight'
            
            # 检查是否存在 old_gate 权重且使用了 MoE
            if model.convnet.moe_layer is not None and old_gate_weight_key in state_dict:
                # 获取 checkpoint 中 old_gate 的输出维度 (即旧专家数量)
                saved_old_experts = state_dict[old_gate_weight_key].shape[0]
                
                # 获取当前初始化模型中 old_gate 的输出维度
                # 注意：old_gate 是一个 Sequential，索引 8 是最后一层 Linear
                current_old_experts = model.convnet.moe_layer.old_gate[8].out_features
                
                if saved_old_experts != current_old_experts:
                    self.logger.info(f"检测到 old_gate 维度不匹配: Checkpoint={saved_old_experts}, Model={current_old_experts}")
                    self.logger.info("正在重构 old_gate 以匹配 Checkpoint...")
                    
                    # 获取输入维度 (第一层的输入特征数)
                    input_dim = model.convnet.moe_layer.gate[0].in_features
                    
                    # 使用 MoELayer 的内部方法重建 old_gate
                    new_old_gate = model.convnet.moe_layer._build_gate_network(input_dim, saved_old_experts)
                    
                    # 替换模型中的 old_gate
                    model.convnet.moe_layer.old_gate = new_old_gate
                    model.convnet.moe_layer.old_num_experts = saved_old_experts
                    
                    # 必须将新层移动到正确的设备
                    model.convnet.moe_layer.old_gate.to(self.device)
            # ==================== 修复结束 ====================

            # 4. 加载主网络参数
            model.load_state_dict(checkpoint['network_state_dict'])
            self.logger.info("主网络参数加载成功")
            
            # 恢复其他状态
            model._total_classes = checkpoint['total_classes']
            model._known_classes = checkpoint['known_classes']
            model._cur_task = checkpoint['cur_task']
            model._data_memory = checkpoint['data_memory']
            model._targets_memory = checkpoint['targets_memory']
            
            model.to(self.device)
            self.logger.info(f"模型已转移到设备: {self.device}")
            
            self.logger.info("模型加载完成")
            return model

    def _print_model_params(self, model):
        """打印模型所有参数信息"""
        self.logger.info(f"{'='*80}")
        self.logger.info(f"{'参数名称':<50} | {'形状':<20} | {'均值':<10} | {'标准差':<10} | {'最小值':<10} | {'最大值':<10}")
        self.logger.info(f"{'-'*100}")
        
        total_params = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_data = param.data.cpu()
                mean = param_data.mean().item()
                std = param_data.std().item()
                min_val = param_data.min().item()
                max_val = param_data.max().item()
                shape = tuple(param_data.shape)
                
                self.logger.info(
                    f"{name:<50} | {str(shape):<20} | "
                    f"{mean:>10.6f} | {std:>10.6f} | "
                    f"{min_val:>10.6f} | {max_val:>10.6f}"
                )
                
                total_params += param_data.numel()
        
        self.logger.info(f"\n总计可学习参数数量: {total_params}")
        self.logger.info(f"{'='*80}\n")

    def _print_moe_params(self, moe_layer):
        """详细打印MoE层参数信息"""
        if moe_layer is None:
            self.logger.info("MoE层不存在")
            return
        
        self.logger.info(f"{'='*80}")
        self.logger.info("MoE层参数详情")
        self.logger.info(f"{'参数名称':<50} | {'形状':<20} | {'均值':<10} | {'标准差':<10} | {'最小值':<10} | {'最大值':<10}")
        self.logger.info(f"{'-'*100}")
        
        # 打印门控网络参数
        for name, param in moe_layer.gate.named_parameters():
            param_data = param.data.cpu()
            self.logger.info(
                f"gate.{name:<45} | {str(tuple(param_data.shape)):<20} | "
                f"{param_data.mean().item():>10.6f} | {param_data.std().item():>10.6f} | "
                f"{param_data.min().item():>10.6f} | {param_data.max().item():>10.6f}"
            )
        
        # 打印专家网络参数
        for i, expert in enumerate(moe_layer.experts):
            for name, param in expert.named_parameters():
                param_data = param.data.cpu()
                self.logger.info(
                    f"expert_{i}.{name:<42} | {str(tuple(param_data.shape)):<20} | "
                    f"{param_data.mean().item():>10.6f} | {param_data.std().item():>10.6f} | "
                    f"{param_data.min().item():>10.6f} | {param_data.max().item():>10.6f}"
                )
        
        self.logger.info(f"{'='*80}\n")