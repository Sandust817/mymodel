from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

import pytz
import datetime
import torch.nn.functional as F
warnings.filterwarnings('ignore')
# 类内原型多样性正则化


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        Args:
            gamma: 聚焦参数，默认2.0（常用范围1-5）
            alpha: 类别权重（list或tensor），若为None则不使用类别平衡
            reduction: 损失聚合方式（'mean'/'sum'/'none'）
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha, dtype=torch.float)
            else:
                self.alpha = alpha  # 需保证与类别数匹配
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: 模型输出的logits（未经过softmax，shape: [batch_size, num_classes]）
            targets: 真实标签（shape: [batch_size]，整数类型）
        """
        # 计算交叉熵损失（含log_softmax）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # 计算目标类别的预测概率p_t
        pt = torch.exp(-ce_loss)  # exp(-ce_loss) = softmax(inputs)[targets]
        # 计算Focal Loss的调制因子：(1 - p_t)^gamma
        focal_term = (1 - pt) ** self.gamma
        
        # 应用类别权重alpha（若有）
        if self.alpha is not None:
            # 取出目标类对应的alpha值（需保证alpha维度与类别数一致）
            alpha = self.alpha.to(inputs.device)[targets]
            focal_loss = alpha * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss
        
        # 按指定方式聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        # 1. 固定代码保存根目录（按你的项目路径设置）
        exp_id=datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y%m%d%H')
        self.code_save_root = '/root/sxh/mymodel2/Time-Series-Library/checkpoints/_record'+exp_id
        # 2. 确保目录存在（无则创建）
        if not os.path.exists(self.code_save_root):
            os.makedirs(self.code_save_root, exist_ok=True)

    def _build_model(self):
        train_data, train_loader = self._get_data(flag='TRAIN')
        test_data, test_loader = self._get_data(flag='TEST')
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.pred_len = 0
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        
        # 模型初始化
        if self.args.model == 'TimeDART':
            model = self.model_dict[self.args.model].ClsModel(self.args).float()
        else:
            model = self.model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.RAdam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion= self.model.ev
        criterion = nn.CrossEntropyLoss()
        alpha = None 
        # criterion = FocalLoss(gamma=2.0, alpha=alpha, reduction='mean')
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)
                pred = outputs.detach()
                loss = criterion(pred, label.long().squeeze(-1))
                total_loss.append(loss.item())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)
        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(preds)
        predictions = torch.argmax(probs, dim=1).cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        self.model.train()
        return total_loss, accuracy



    def train(self, setting):
        prototype_warmup_epochs=3
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='TEST')
        test_data, test_loader = self._get_data(flag='TEST')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 初始化优化器
        model_optim = self._select_optimizer()
        # -------------------------- 1. 添加学习率衰减器 --------------------------
        # 选择1：StepLR（固定epoch间隔衰减，常用）
        # 参数说明：step_size=10（每10个epoch衰减一次），gamma=0.5（每次衰减为原来的50%）
        scheduler = optim.lr_scheduler.StepLR(model_optim, step_size=self.args.patience, gamma=0.5)
        
        # （可选）选择2：ReduceLROnPlateau（基于验证损失衰减，更智能，按需替换）
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     model_optim, mode='min', factor=0.5, patience=5, verbose=True
        # )  # 验证损失5个epoch不下降则衰减50%

        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            # -------------------------- 2. 每个epoch开始时记录当前学习率 --------------------------
            current_lr = model_optim.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.args.train_epochs} | Current Learning Rate: {current_lr:.6f}")

            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                # if(i==0):
                #     print(batch_x.shape)
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)
                outputs = self.model(batch_x, padding_mask, label, None)
                loss = criterion(outputs, label.long().squeeze(-1))
                if self.args.model=='TimePNP':
                    loss+=self.model.diversity_loss()*0.1
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                model_optim.step()

            # -------------------------- 3. 每个epoch结束后更新学习率 --------------------------
            # StepLR：直接调用step()即可按固定间隔衰减
            scheduler.step()
            # （若用ReduceLROnPlateau，需替换为：scheduler.step(vali_loss)，基于验证损失更新）

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            if epoch < prototype_warmup_epochs:
                print(f"Epoch {epoch + 1}: Prototype warmup phase - skipping validation and model saving")
                vali_loss, val_accuracy = float('nan'), float('nan')
                test_loss, test_accuracy = float('nan'), float('nan')
                continue
            else:
                vali_loss, val_accuracy = self.vali(vali_data, vali_loader, criterion)
                test_loss, test_accuracy = self.vali(test_data, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.3f} Vali Loss: {3:.3f} Vali Acc: {4:.3f} Test Loss: {5:.3f} Test Acc: {6:.3f}"
                .format(epoch + 1, train_steps, train_loss, vali_loss, val_accuracy, test_loss, test_accuracy))
            early_stopping(-val_accuracy, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    # def test(self, setting, test=0):
    #     test_data, test_loader = self._get_data(flag='TEST')
    #     if test:
    #         print('loading model')
    #         self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

    #     preds = []
    #     trues = []
    #     folder_path = './test_results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, label, padding_mask) in enumerate(test_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             padding_mask = padding_mask.float().to(self.device)
    #             label = label.to(self.device)

    #             outputs = self.model(batch_x, padding_mask, None, None)

    #             preds.append(outputs.detach())
    #             trues.append(label)

    #     preds = torch.cat(preds, 0)
    #     trues = torch.cat(trues, 0)
    #     print('test shape:', preds.shape, trues.shape)

    #     probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
    #     predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
    #     trues = trues.flatten().cpu().numpy()
    #     accuracy = cal_accuracy(predictions, trues)

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     print('accuracy:{}'.format(accuracy))
    #     file_name='result_classification.txt'
    #     f = open(os.path.join(folder_path,file_name), 'a')
    #     f.write(setting + "  \n")
    #     f.write('accuracy:{}'.format(accuracy))
    #     f.write('\n')
    #     f.write('\n')
    #     f.close()

    #     ckpt_result_path = os.path.join(self.code_save_root,"result.txt")
    #     with open(ckpt_result_path, 'w') as f:
    #         f.write(f"Setting: {setting}\n")
    #         f.write(f"Model: {self.args.model}\n")
    #         f.write(f"Accuracy: {accuracy}\n")
    #         f.write(f"Test time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    #     return
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='TEST')
        if test:
            print('loading model from:', self.args.ckpt_path)
            # 从args的ckpt_path参数加载模型权重
            self.model.load_state_dict(torch.load(self.args.ckpt_path))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                label = label.to(self.device)

                outputs = self.model(batch_x, padding_mask, None, None)

                preds.append(outputs.detach())
                trues.append(label)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        print('test shape:', preds.shape, trues.shape)

        probs = torch.nn.functional.softmax(preds)  # (total_samples, num_classes) est. prob. for each class and sample
        predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
        trues = trues.flatten().cpu().numpy()
        accuracy = cal_accuracy(predictions, trues)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        print('accuracy:{}'.format(accuracy))
        file_name='result_classification.txt'
        f = open(os.path.join(folder_path,file_name), 'a')
        f.write(setting + "  \n")
        f.write('accuracy:{}'.format(accuracy))
        f.write('\n')
        f.write('\n')
        f.close()

        ckpt_result_path = os.path.join(self.code_save_root,"result.txt")
        with open(ckpt_result_path, 'w') as f:
            f.write(f"Setting: {setting}\n")
            f.write(f"Model: {self.args.model}\n")
            f.write(f"Accuracy: {accuracy}\n")
            f.write(f"Test time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        return