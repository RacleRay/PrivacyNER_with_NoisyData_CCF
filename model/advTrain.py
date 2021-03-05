import torch


class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_adv = epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self, emb_name='emb.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



##################################################################
grad_backup = {}


def save_grad(tensorName):
    def backward_hook(grad: torch.Tensor):
        grad_backup[tensorName] = grad

    return backward_hook


class PGD:
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}

    def attack(self,
               epsilon=1.,
               alpha=0.3,
               emb_name='emb.',
               is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='emb.'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        # 此处也可以直接用一个成员变量储存 grad，而不用 register_hook 存储在全局变量中
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.register_hook(save_grad(name))

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = grad_backup[name]



if __name__ == "__main__":
    # 示例过程
    fgm = FGM(model)
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward()  # 反向传播，得到正常的grad
        # 对抗训练
        fgm.attack()  # 在embedding上添加对抗扰动
        loss_adv = model(batch_input, batch_label)
        loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        fgm.restore()  # 恢复embedding参数
        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()


    ##################################################################
    # 示例过程
    pgd = PGD(model)
    K = 3 # 小步走的步数
    for batch_input, batch_label in data:
        # 正常训练
        loss = model(batch_input, batch_label)
        loss.backward() # 反向传播，得到正常的grad
        pgd.backup_grad()

        # 对抗训练
        for t in range(K):
            pgd.attack(is_first_attack=(t==0)) # 在embedding上添加对抗扰动, first attack时备份param.data
            if t != K-1:
                model.zero_grad()
            else:
                pgd.restore_grad()
            loss_adv = model(batch_input, batch_label)
            loss_adv.backward() # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
        pgd.restore() # 恢复embedding参数

        # 梯度下降，更新参数
        optimizer.step()
        model.zero_grad()