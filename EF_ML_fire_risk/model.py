import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import random
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
from torch.nn.utils.rnn import PackedSequence
from sklearn.metrics import (roc_auc_score,f1_score,
                             precision_recall_curve)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
class Fire_model_hybrid(torch.nn.Module):
    def __init__(self, input_dim, output_dim,n_units,drop_rate=0.2,npp_idx=0,popu_idx=1):
        super().__init__()
        self.input_dim=input_dim-2
        self.output_dim=output_dim
        self.FF_model = FF_attention_module(self.input_dim, output_dim, n_units)#.cuda()
        self.FA_model=FA_limitaton(1,1)
        self.HS_model = HS_limitaton(1, 1)
        self.npp_idx=npp_idx
        self.popu_idx=popu_idx
        self.out=nn.Sigmoid()
        self.drop_rate=drop_rate
        self.in_dropout=VariationalDropout(self.drop_rate,batch_first=True)

    def forward(self, x):
        x_FF=torch.concat((x[:,:,:self.popu_idx],x[:,:,-3:]),dim=-1)
        x_FF=self.in_dropout(x_FF)
        x_FA=x[:,-1,self.npp_idx:self.npp_idx+1]#
        f_npp=self.FA_model(x_FA)#
        x_popu=x[:,-1,self.popu_idx:self.popu_idx+1]
        f_popu=self.HS_model(x_popu)
        f_FF = self.FF_model(x_FF)
        f_FF = self.out(f_FF)
        f_out_put=f_FF*f_npp*f_popu
        return f_out_put
class FA_limitaton(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_NPP = torch.nn.Linear(self.input_dim, self.output_dim)
        self.logistic_NPP = torch.nn.Sigmoid()
    def forward(self, x):
        f_out_put = self.logistic_NPP(self.linear_NPP(x))
        return f_out_put
class HS_limitaton(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_Popu = torch.nn.Linear(self.input_dim, self.output_dim)
        self.logistic_Popu = torch.nn.Sigmoid()
    def forward(self, x):
        f_out_put = self.logistic_Popu(self.linear_Popu(x))
        return f_out_put
class FF_attention_module(torch.nn.Module):
    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units) * init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units) * init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units) * init_std)
        self.W_i = nn.Linear(input_dim * (n_units + 1), input_dim * n_units)
        self.W_f = nn.Linear(input_dim * (n_units + 1), input_dim * n_units)
        self.W_o = nn.Linear(input_dim * (n_units + 1), input_dim * n_units)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1) * init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1) * init_std)
        self.F_beta = nn.Linear(2 * n_units, 1)
        self.Phi = nn.Linear(2 * n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units)#.cuda()
        c_t = torch.zeros(x.shape[0], self.input_dim * self.n_units)#.cuda()
        outputs =[]
        for t in range(x.shape[1]):
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:, t, :].unsqueeze(1), self.U_j) + self.b_j)
            inp = torch.cat([x[:, t, :], h_tilda_t.view(h_tilda_t.shape[0], -1)], dim=1)
            i_t = torch.sigmoid(self.W_i(inp))
            f_t = torch.sigmoid(self.W_f(inp))
            o_t = torch.sigmoid(self.W_o(inp))
            c_t = c_t * f_t + i_t * j_tilda_t.reshape(j_tilda_t.shape[0], -1)
            h_tilda_t = (o_t * torch.tanh(c_t)).view(h_tilda_t.shape[0], self.input_dim, self.n_units)
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) + self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas / torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas * outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas / torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas * mu, dim=1)
        return mean

from typing import *
class VariationalDropout(nn.Module):
    """
    Applies the same dropout mask across the temporal dimension
    See https://arxiv.org/abs/1512.05287 for more details.
    Note that this is not applied to the recurrent activations in the LSTM like the above paper.
    Instead, it is applied to the inputs and outputs of the recurrent layer.
    """
    def __init__(self, dropout: float, batch_first: Optional[bool]=False):
        super().__init__()
        self.dropout = dropout
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.dropout <= 0.:
            return x

        is_packed = isinstance(x, PackedSequence)
        if is_packed:
            x, batch_sizes = x
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = x.size(0)

        # Drop same mask across entire sequence
        if self.batch_first:
            m = x.new_empty(max_batch_size, 1, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        else:
            m = x.new_empty(1, max_batch_size, x.size(2), requires_grad=False).bernoulli_(1 - self.dropout)
        x = x.masked_fill(m == 0, 0) / (1 - self.dropout)

        if is_packed:
            return PackedSequence(x, batch_sizes)
        else:
            return x

def ml_train_test_random_cross_validation_states(seed_num,x_all_save_raw,y_all_save_raw,
                                                 para_dir,para_path_npp_raw,para_path_popu_raw,him_dim,
                                                 npp_idx=5,popu_idx=6,initial_weights=True,lr_FA=0.005,lr_HS=0.005):
    result_metrics = np.zeros((seed_num, 2))#store the evaluation metrics
    for seed in range(seed_num):
        #################################################
        ###load and scale the inputs for the ML model
        x_all_save=f'{x_all_save_raw}_{seed}.npy'
        y_all_save = f'{y_all_save_raw}_{seed}.npy'
        x_all=np.load(x_all_save)
        y_all=np.load(y_all_save)
        scaler = MinMaxScaler(feature_range=(-3, 3))
        values=x_all.reshape((-1, x_all.shape[2]))
        scaler.fit(values)
        scaled =scaler.transform(values)
        x_all = scaled.reshape(x_all.shape)
        #################################################
        ###random cross validation: each time with a different random seed
        ###prepare the train, validation, and test dataset for the ML model
        random.seed(seed)
        train_sample_number,validate_sample_number=int(len(y_all)*0.8), int(len(y_all)*0.1)
        all_idxs=list(range(len(y_all)))
        sampled_train_idxs = random.sample(all_idxs, train_sample_number)
        validate_test_idxs=list(set(all_idxs).difference(set(sampled_train_idxs)))
        random.seed(seed)
        validate_idxs=random.sample(validate_test_idxs, validate_sample_number)
        test_idxs=list(set(validate_test_idxs).difference(set(validate_idxs)))
        print(len(validate_idxs),len(test_idxs),len(sampled_train_idxs))
        x_train=x_all[np.array(sampled_train_idxs)]
        y_train=y_all[np.array(sampled_train_idxs)]
        x_test=x_all[np.array(test_idxs)]
        y_test=y_all[np.array(test_idxs)]
        x_validate = x_all[np.array(validate_idxs)]
        y_validate = y_all[np.array(validate_idxs)]
        x_train = torch.Tensor(x_train)
        x_test = torch.Tensor(x_test)
        y_train = torch.Tensor(y_train)
        y_test = torch.Tensor(y_test)
        x_validate = torch.Tensor(x_validate)
        y_validate = torch.Tensor(y_validate)
        batch_size=16
        train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, batch_size=batch_size)
        val_loader = DataLoader(TensorDataset(x_validate, y_validate), shuffle=False, batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(x_test, y_test), shuffle=False, batch_size=batch_size)
        #################################################################
        ###initialize the ML model
        device = 'cpu'
        model = Fire_model_hybrid(x_train.shape[2], 1, him_dim, drop_rate=0.1,npp_idx=npp_idx,popu_idx=popu_idx).to(device=device)  # .cuda()
        ##############################################
        ###if initial_weights=False, FA or HS module will be initilized by the default method in pytorch
        ###if initial_weights=True, FA or HS module will be initilized by the users
        if initial_weights:
            npp_model = model.FA_model
            hs_model = model.HS_model
            para_path_npp = f"{para_path_npp_raw}.pt"
            para_path_popu = f"{para_path_popu_raw}.pt"
            if os.path.exists(para_path_npp):
                model_temp = FA_limitaton(1, 1)
                model_temp.load_state_dict(torch.load(para_path_npp))
                params = model_temp.state_dict()
                linear_NPP_a = params['linear_NPP.weight'].cpu().detach().numpy()[0, 0]
                linear_NPP_b = params['linear_NPP.bias'].cpu().detach().numpy()[0]
                npp_model.linear_NPP.weight.data.fill_(linear_NPP_a)
                npp_model.linear_NPP.bias.data.fill_(linear_NPP_b)
            if os.path.exists(para_path_popu):
                model_temp = HS_limitaton(1, 1)
                model_temp.load_state_dict(torch.load(para_path_popu))
                params = model_temp.state_dict()
                linear_Popu_a = params['linear_NPP.weight'].cpu().detach().numpy()[0, 0]
                linear_Popu_b = params['linear_NPP.bias'].cpu().detach().numpy()[0]
                hs_model.linear_Popu.weight.data.fill_(linear_Popu_a)
                hs_model.linear_Popu.bias.data.fill_(linear_Popu_b)
        ##############################################
        ###setup of ML optimization parameters
        opt = torch.optim.Adam( [{'params': model.FF_model.parameters()},
                                 {'params': model.HS_model.parameters(),'lr': lr_HS},
                                 {'params': model.FA_model.parameters(), 'lr': lr_FA}
            ], lr=0.005,weight_decay=pow(10,-5))

        epoch_scheduler = torch.optim.lr_scheduler.StepLR(opt, 5, gamma=0.6)  # adjust lr
        epochs = 50
        loss=nn.BCELoss()
        patience = 50
        counter = 0
        ##############################################
        ###the path of the ML parameters to be saved
        para_path = f"{para_dir}_seed{seed}.pt"
        #############################################
        ###initialize a f1-score
        with torch.no_grad():
            model.eval()
            mse_val = 0
            preds = []
            true = []
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device=device)  # .cuda()
                batch_y = batch_y.to(device=device)  # .cuda()
                output, alphas, betas,mu = model(batch_x)
                output = output.squeeze(1)
                preds.append(output.cpu().numpy())
                true.append(batch_y.cpu().numpy())
                mse_val += loss(output, batch_y).item() * batch_x.shape[0]
        preds = np.concatenate(preds)
        true = np.concatenate(true)
        min_val_roc = roc_auc_score(true, preds)
        precision, recall, ts = precision_recall_curve(y_true=true, probas_pred=preds)
        f1 = pd.Series({t: f1_score(y_true=true, y_pred=preds > t) for t in ts})
        f1_idx = f1.idxmax()
        min_val_f1_result = f1[f1_idx]
        #############################################################
        #####start training
        for i in range(epochs):
            mse_train = 0
            iteration_start = time.monotonic()
            model.train()  # set model mode as train
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device=device)  # .cuda()
                batch_y = batch_y.to(device=device)  # .cuda()
                opt.zero_grad()
                y_pred, alphas, betas,mu = model(batch_x)
                y_pred = y_pred.squeeze(1)
                l = loss(y_pred, batch_y)
                l.backward()  #
                mse_train += l.item() * batch_x.shape[0]
                opt.step()  #
            epoch_scheduler.step()
            # validate
            with torch.no_grad():
                model.eval()
                mse_val = 0
                preds = []
                true = []
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device=device)  # .cuda()
                    batch_y = batch_y.to(device=device)  # .cuda()
                    output, alphas, betas,mu = model(batch_x)
                    output = output.squeeze(1)
                    preds.append(output.cpu().numpy())
                    true.append(batch_y.cpu().numpy())

                    mse_val += loss(output, batch_y).item() * batch_x.shape[0]
            preds = np.concatenate(preds)
            true = np.concatenate(true)
            roc_auc = roc_auc_score(true,preds)
            precision, recall, ts = precision_recall_curve(y_true=true, probas_pred=preds)
            f1 = pd.Series({t: f1_score(y_true=true, y_pred=preds > t) for t in ts})
            f1_idx = f1.idxmax()
            f1_result = f1[f1_idx]
            if (min_val_roc < roc_auc) or (min_val_f1_result<f1_result):#
                min_val_roc = roc_auc
                min_val_f1_result=f1_result
                print("Saving...")
                torch.save(model.state_dict(), para_path)
                counter = 0
            else:
                counter += 1
            if counter == patience:
                break
            print("Iter: ", i, "train: ", (mse_train / len(x_train)) ** 0.5, "val: ", (mse_val / len(x_train)) ** 0.5,"roc: ", roc_auc,"f1-score:",f1_result)
            iteration_end = time.monotonic()
            print("Iter time: ", iteration_end - iteration_start)
            if (i % 10 == 0):
                mse = mean_squared_error(true, preds)
                mae = mean_absolute_error(true, preds)
                r = stats.pearsonr(true, preds)[0]
                roc_auc = roc_auc_score(true,preds)
                print("mse: ", mse, "mae: ", mae, 'r:', r,'roc:',roc_auc,"f1-score:",f1_result)
        ######### end: train and validate
        # test
        model.load_state_dict(torch.load(para_path))
        with torch.no_grad():
            model.eval()
            mse_val = 0
            preds = []
            true = []
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device=device)  # .cuda()
                batch_y = batch_y.to(device=device)  # .cuda()
                output, a, b,mu = model(batch_x)
                output = output.squeeze(1)
                preds.append(output.cpu().numpy())
                true.append(batch_y.cpu().numpy())
                mse_val += loss(output, batch_y).item() * batch_x.shape[0]
        preds = np.concatenate(preds)
        true = np.concatenate(true)
        roc_auc = roc_auc_score(true,preds)
        f1 = pd.Series({t: f1_score(y_true=true, y_pred=preds > t) for t in ts})
        f1_idx = f1.idxmax()
        f1_result = f1[f1_idx]
        result_metrics[seed,0]=roc_auc
        result_metrics[seed, 1] = f1_result
    return result_metrics