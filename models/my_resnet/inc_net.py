from torch import nn
import copy
from models.my_resnet.my_resnet import ResNet34 as my_resnet34
import torch
from models.my_resnet.linears import SimpleLinear
import torch.nn.functional as F

def get_convnet():
    return my_resnet34(num_c=1,use_moe=True,moe_experts=1)


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
    def __init__(self, use_moe=False):
        super().__init__()
        self.use_moe = use_moe  # 👈 新增：是否使用 MoE
        self._cur_task = 0      # 👈 新增：记录当前任务 ID
        self.fc = self.generate_fc(self.feature_dim, 5)
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

    def forward(self, x, task_id=None,routing_targets=None):
        """
        :param x: 输入图像
        :param task_id: 可选，当前任务 ID，用于 MoE 路由控制
        """
        # 👇 传入 task_id 给 convnet（ResNet with MoE）
        if self.use_moe:
            x = self.convnet(x, task_id=task_id,routing_targets = routing_targets)
        else:
            x = self.convnet(x)

        out = self.fc(x["features"])
        out.update(x)  # 保留 fmaps, features 等


        # 添加路由损失到输出
        if "routing_loss" in x:
            out["routing_loss"] = x["routing_loss"]
        

        return out['logits']
    def feature_list(self,x):
        out_list = []
        out = F.relu(self.convnet.bn1(self.convnet.conv1(x)))
        out_list.append(out)
        out = self.convnet.layer1(out)
        out_list.append(out)
        out = self.convnet.layer2(out)
        out_list.append(out)
        out = self.convnet.layer3(out)
        out_list.append(out)
        out = self.convnet.layer4(out)
        out_list.append(out)
        pooled = F.avg_pool2d(out, 4)
        features = pooled.view(pooled.size(0), -1)
        
        # 计算 logits（通过全连接层）
        logits = self.fc(features)['logits']
        
        # 返回 logits 和特征列表（与原始接口一致）
        return logits, out_list
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.convnet.bn1(self.convnet.conv1(x)))
        if layer_index == 1:
            out = self.convnet.layer1(out)
        elif layer_index == 2:
            out = self.convnet.layer1(out)
            out = self.convnet.layer2(out)     
        elif layer_index == 3:
            out = self.convnet.layer1(out)
            out = self.convnet.layer2(out)
            out = self.convnet.layer3(out)
        elif layer_index == 4:
            out = self.convnet.layer1(out)
            out = self.convnet.layer2(out)
            out = self.convnet.layer3(out)
            out = self.convnet.layer4(out)
        return out
