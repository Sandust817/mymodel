from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_SelfSupervised(Exp_Basic):
    def __init__(self, args):
        super(Exp_SelfSupervised, self).__init__(args)
        # 自监督相关参数
        self.args.mask_ratio = getattr(args, 'mask_ratio', 0.2)  # 掩码比例

    def _build_model(self):
        # 从数据中获取模型所需参数
        train_data, _ = self._get_data(flag='TRAIN')
        self.args.seq_len = train_data.max_seq_len
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        
        # 初始化自监督模型 - 假设原模型可以修改为接受掩码并进行重建
        model = self.model_dict[self.args.model].SelfSupervisedModel(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        # 获取数据，忽略标签
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 自监督任务使用MSE损失函数（重建损失）
        criterion = nn.MSELoss()
        return criterion

    def _create_mask(self, x):
        """创建随机掩码，用于自监督任务"""
        batch_size, seq_len, _ = x.shape
        mask = torch.rand(batch_size, seq_len, 1, device=self.device) < self.args.mask_ratio
        return mask.float()

    def vali(self, vali_loader, criterion):
        """验证函数，仅关注重建损失"""
        total_loss = []
        self.model.eval()
        
        with torch.no_grad():
            for batch_x, _, padding_mask in vali_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                
                # 创建掩码
                mask = self._create_mask(batch_x)
                
                # 模型输出重建结果
                outputs = self.model(batch_x, padding_mask, mask, None)
                
                # 计算重建损失（只关注被掩码的部分）
                loss = criterion(outputs * mask, batch_x * mask)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        # 获取数据加载器，忽略标签
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')

        # 创建保存路径
        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, _, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                
                # 创建掩码
                mask = self._create_mask(batch_x)
                
                # 模型输出重建结果
                outputs = self.model(batch_x, padding_mask, mask, None)
                
                # 计算重建损失（只关注被掩码的部分）
                loss = criterion(outputs * mask, batch_x * mask)
                train_loss.append(loss.item())

                # 打印训练进度
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                # 反向传播和参数更新
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            # 打印 epoch 统计信息
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss_avg = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss_avg:.3f} "
                f"Vali Loss: {vali_loss:.3f}"
            )
            
            # 早停策略基于验证损失
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # 加载最佳模型
        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """测试函数，仅用于评估自监督模型的重建能力"""
        test_data, test_loader = self._get_data(flag='TEST')
        
        if test:
            print('Loading best model...')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        total_loss = []
        folder_path = './test_results/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for batch_x, _, padding_mask in test_loader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                
                mask = self._create_mask(batch_x)
                outputs = self.model(batch_x, padding_mask, mask, None)
                
                loss = nn.MSELoss()(outputs * mask, batch_x * mask)
                total_loss.append(loss.item())

        avg_loss = np.average(total_loss)
        print(f'Test average reconstruction loss: {avg_loss:.6f}')

        # 保存结果
        result_path = './results/' + setting + '/'
        os.makedirs(result_path, exist_ok=True)
        
        with open(os.path.join(result_path, 'result_self_supervised.txt'), 'a') as f:
            f.write(setting + "\n")
            f.write(f'Average reconstruction loss: {avg_loss:.6f}\n\n')

        return avg_loss
