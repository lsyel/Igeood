from torch import nn
import copy
from models.my_resnet.my_resnet import ResNet34 as my_resnet34
import torch
from models.my_resnet.linears import SimpleLinear
import torch.nn.functional as F

def get_convnet(use_moe=True, moe_experts=1):
    return my_resnet34(num_c=1,use_moe=use_moe,moe_experts=moe_experts)

    
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()

        self.convnet = get_convnet()
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self
    
    def load_checkpoint(self, args):
        if args["init_cls"] == 50:
            pkl_name = "{}_{}_{}_B{}_Inc{}".format( 
                args["dataset"],
                args["seed"],
                args["convnet_type"],
                0,
                args["init_cls"],
            )
            checkpoint_name = f"checkpoints/finetune_{pkl_name}_0.pkl"
        else:
            checkpoint_name = f"checkpoints/finetune_{args['csv_name']}_0.pkl"
        model_infos = torch.load(checkpoint_name)
        self.convnet.load_state_dict(model_infos['convnet'])
        self.fc.load_state_dict(model_infos['fc'])
        test_acc = model_infos['test_acc']
        return test_acc

class IncrementalNet(BaseNet):
    def __init__(self, num_classes, use_moe=False, moe_experts=1):
        super().__init__()
        self.use_moe = use_moe  # 👈 新增：是否使用 MoE
        self._cur_task = 0      # 👈 新增：记录当前任务 ID
        self.fc = self.generate_fc(self.feature_dim, num_classes)
        self.convnet = get_convnet(use_moe=use_moe, moe_experts=moe_experts)

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights,gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    # 👇👇👇 核心修改：支持 MoE 专家扩展
    def update_moe_experts(self, task_id):
        """
        当开始新任务时调用，扩展 MoE 专家数量。
        假设每个任务对应一个专家。
        """
        if not self.use_moe:
            return

        # 获取当前 convnet 中的 moe_layer
        if hasattr(self.convnet, 'moe_layer') and self.convnet.moe_layer is not None:
            current_experts = self.convnet.moe_layer.num_experts
            if task_id + 1 > current_experts:
                print(f"🔧 Expanding MoE experts from {current_experts} to {task_id + 1}")
                self.convnet.moe_layer.expand_experts(task_id + 1)
        else:
            print("⚠️ MoE layer not found in convnet. Did you initialize with use_moe=True?")
    def _forward_impl(self, x, task_id=None, routing_targets=None):
        """
        实际的前向计算实现，返回完整输出字典
        """
        # 使用MoE时传递task_id
        conv_out = self.convnet(x)
        
        # 计算logits
        logits = self.fc(conv_out["features"])['logits']

        return logits
    def forward(self, x, task_id=None,routing_targets=None):
        return self._forward_impl(x, task_id, routing_targets)

    def feature_list(self, x):
        """
        获取所有隐藏层特征（保持与训练时一致的处理流程）
        """
        # 调用convnet的专用特征提取方法
        features, out_list = self.convnet.feature_list(x)
        
        # 计算logits（如果需要）
        logits = self._forward_impl(x)
        
        return logits, out_list
    def intermediate_forward(self, x, layer_index):
        """
        获取指定中间层特征
        """
        return self.convnet.intermediate_forward(x, layer_index)

