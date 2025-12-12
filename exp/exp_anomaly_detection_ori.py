from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)

    def _build_model(self):
        # 导入你的原型模型（替换为你的model路径）
        from improved_model import Model as PrototypeModel
        model = PrototypeModel(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        # 为原型模型的参数分组优化（可选：给proto_confidence更高学习率）
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 原MSE损失不再使用，这里返回一个占位符（训练时用模型内部的loss）
        return nn.MSELoss()  # 保留以兼容代码结构，实际训练不用

    def vali(self, vali_data, vali_loader):
        """
        重构验证函数：使用原型模型的异常得分，计算伪损失（1 - PR-AUC）用于早停
        """
        self.model.eval()
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)  # [B, C, L]
                batch_y = batch_y.long().to(self.device)  # [B*L] or [B, L]

                # 原型模型前向传播：输出时间步级异常得分 [B, L]
                scores = self.model(batch_x, labels=None)  # [B, L]

                # 展平得分和标签到一维（适配原代码逻辑）
                scores = scores.reshape(-1)  # [B*L]
                if len(batch_y.shape) == 2:
                    batch_y = batch_y.reshape(-1)  # [B*L]
                batch_y = batch_y.cpu().numpy()
                scores = scores.detach().cpu().numpy()

                all_scores.append(scores)
                all_labels.append(batch_y)

        # 拼接所有数据
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        # 计算PR-AUC作为验证指标（替代原MSE损失，用于早停）
        from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
        try:
            precision, recall, _ = precision_recall_curve(all_labels, all_scores)
            auc_pr = auc(recall, precision)
            fake_loss = 1.0 - auc_pr  # 早停希望loss越小越好
        except Exception as e:
            print(f"验证集计算PR-AUC失败：{e}，伪损失设为0.5")
            fake_loss = 0.5

        self.model.train()
        return fake_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        # 原criterion不再使用，仅保留变量
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss_list = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)  # [B, C, L]
                # batch_y在训练时可忽略（原型模型是无监督）

                # 原型模型前向传播：训练时返回(loss, metrics)
                loss, metrics = self.model(batch_x, labels=None)  # 模型内部计算的loss

                train_loss_list.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss_list)
            # 验证集损失：替换为原型模型的伪损失（1 - PR-AUC）
            vali_loss = self.vali(vali_data, vali_loader)
            # 测试集暂时用验证集的逻辑（可选：也可以计算test的伪损失）
            test_loss = self.vali(test_data, test_loader)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        # 存储训练集的异常得分（用于计算阈值）
        train_energy = []
        # 存储测试集的异常得分和标签
        test_energy = []
        test_labels = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()

        # (1) 统计训练集的异常得分（替代原MSE重构误差）
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                # 原型模型前向传播：输出时间步级异常得分 [B, L]
                scores = self.model(batch_x, labels=None)  # [B, L]
                # 展平为一维（适配原代码逻辑）
                scores = scores.reshape(-1).detach().cpu().numpy()
                train_energy.append(scores)

        train_energy = np.concatenate(train_energy, axis=0).reshape(-1)
        train_energy = np.array(train_energy)

        # (2) 计算测试集的异常得分和标签
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                # 原型模型前向传播：输出时间步级异常得分 [B, L]
                scores = self.model(batch_x, labels=None)  # [B, L]
                # 展平得分和标签
                scores = scores.reshape(-1).detach().cpu().numpy()
                if len(batch_y.shape) == 2:
                    batch_y = batch_y.reshape(-1)
                batch_y = batch_y.numpy()

                test_energy.append(scores)
                test_labels.append(batch_y)

        test_energy = np.concatenate(test_energy, axis=0).reshape(-1)
        test_energy = np.array(test_energy)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)

        # (3) 找到阈值（和原代码逻辑一致：合并训练+测试集，按异常比例取分位数）
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)

        # (4) 生成预测结果
        pred = (test_energy > threshold).astype(int)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (5) 检测调整（和原代码一致：adjustment函数）
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        # (6) 计算指标（和原代码完全一致）
        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))

        # (7) 保存结果（和原代码一致）
        f = open("result_anomaly_detection.txt", 'a')
        f.write(setting + "  \n")
        f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision, recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        return