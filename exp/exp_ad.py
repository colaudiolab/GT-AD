from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import torch.multiprocessing
from PIL import Image

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from models.MyModel import Model
from metrics.metrics import combine_all_evaluation_scores
from thop import profile
warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(object):
    def __init__(self, args):
        self.args = args
        self._build_model()

    def _build_model(self):
        self.model = Model(self.args)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.model.to(self.device)

        # net = self.model  # 定义好的网络模型
        # inputs_time = torch.randn(1, 100, 51).to(self.device)
        # total_path = r''
        # filename_png = ""
        # img_path = os.path.join(total_path, filename_png)
        # img = Image.open(img_path).convert('RGB')
        # picture = torch.tensor(np.float32(np.array(img).transpose(2, 0, 1)))
        # img.close()
        # inputs_picture = picture.unsqueeze(0).unsqueeze(0).to(self.device)

        # flops, params = profile(net, (inputs_time,inputs_picture))
        # print('flops: ', flops, 'params: ', params)
        # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            tqdm_vali_loader = tqdm(vali_loader)
            for i, (batch_x, batch_y, batch_picture) in enumerate(tqdm_vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_picture = batch_picture.float().to(self.device)

                outputs_raw, outputs_picture, output_time = self.model(batch_x, batch_picture)
                loss_1 = criterion(outputs_raw, outputs_picture)
                loss_2 = criterion(output_time, batch_x)
                loss = 0.5 * loss_1 + 0.5 * loss_2
                total_loss.append(loss.detach().cpu())
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

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
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            tqdm_train_loader = tqdm(train_loader)
            for i, (batch_x, batch_y, batch_picture) in enumerate(tqdm_train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_picture = batch_picture.float().to(self.device)

                outputs_raw, outputs_picture, output_time = self.model(batch_x, batch_picture)

                loss_1 = criterion(outputs_raw, outputs_picture)
                loss_2 = criterion(output_time, batch_x)
                loss = 0.5*loss_1 + 0.5*loss_2
                train_loss.append(loss.item())

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
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

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
       # train_data, train_loader = self._get_data(flag='train')
       #  if test:
       #      print('loading model')
       #      self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
#        with torch.no_grad():
#            for i, (batch_x, batch_y) in enumerate(train_loader):
#                batch_x = batch_x.float().to(self.device)
#                # reconstruction
#                outputs = self.model(batch_x, None, None, None)
#                # criterion
#                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
#                score = score.detach().cpu().numpy()
#                attens_energy.append(score)

#        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
#        train_energy = np.array(attens_energy)

        # (2) test set
        test_reconstitution_error = []
        test_labels = []
        tqdm_test_loader = tqdm(test_loader)
        for i, (batch_x, batch_y, batch_picture) in enumerate(tqdm_test_loader):
            batch_x = batch_x.float().to(self.device)
            batch_picture = batch_picture.float().to(self.device)
            outputs_raw, outputs_picture, output_time = self.model(batch_x, batch_picture)
            loss = torch.mean(criterion(batch_x, output_time), dim=-1)
            test_reconstitution_error.append(loss.detach().cpu())
            test_labels.append(batch_y)
            del batch_x, batch_picture, outputs_raw, outputs_picture, output_time, loss
            torch.cuda.empty_cache()
        test_reconstitution_error = torch.cat(test_reconstitution_error, 0).numpy().reshape(-1)
        test_labels = torch.cat(test_labels, 0).numpy().reshape(-1)


        print("anomaly_ratio :", self.args.anomaly_ratio)
        threshold = np.percentile(test_reconstitution_error, 100-self.args.anomaly_ratio)

        # (3) evaluation on the test set
        pred = (test_reconstitution_error > threshold).astype(int)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        scores_simple = combine_all_evaluation_scores(pred, gt, test_reconstitution_error)
        for key, value in scores_simple.items():
            #            matrix.append(value)
            print('{0:21} : {1:0.4f}'.format(key, value))

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        # f = open("result_anomaly_detection.txt", 'a')
        # f.write(setting + "  \n")
        # f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #     accuracy, precision,
        #     recall, f_score))
        # f.write('\n')
        # f.write('\n')
        # f.close()
        return
