import copy
import os
import typing
from collections import defaultdict, OrderedDict
from typing import Dict, List, Union, Optional, Tuple

import numpy as np

from tqdm import tqdm
import pandas as pd
import torch
from torch import optim, nn
import torch.nn.functional as F
import higher
from . import higher_optim  # IMPORTANT, DO NOT DELETE
from .utils import override_state
from .net import DoubleAdapt, ForecastModel


class IncrementalManager:
    def __init__(
        self,
        model: nn.Module,
        lr_model: float = 0.001,
        x_dim: int = None,
        need_permute: bool = True,
        begin_valid_epoch: int = 0,
        args = None,
        **kwargs
    ):
        self.fitted = False
        self.lr_model = lr_model
        self.begin_valid_epoch = begin_valid_epoch
        self.framework = self._init_framework(model, x_dim, lr_model, need_permute=need_permute, args=args, **kwargs)
        self.opt = self._init_meta_optimizer(**kwargs)
        self.args = args

    def _init_framework(self, model: nn.Module, x_dim: int = None, lr_model=0.001, need_permute=False, **kwargs):
        return ForecastModel(model, x_dim=x_dim, lr=lr_model, need_permute=need_permute)

    def _init_meta_optimizer(self, **kwargs):
        return self.framework.opt

    def state_dict(self, destination: typing.OrderedDict[str, torch.Tensor]=None, prefix='', keep_vars=False):
        r"""Returns a dictionary containing a whole state of the module and the state of the optimizer.

        Returns:
            dict:
                a dictionary containing a whole state of the module
        """
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        destination['framework'] = self.framework.state_dict()
        destination['framework_opt'] = self.framework.opt.state_dict()
        destination['opt'] = self.opt.state_dict()
        return destination

    def load_state_dict(self, state_dict: typing.OrderedDict[str, torch.Tensor],):
        self.framework.load_state_dict(state_dict['framework'])
        self.framework.opt.load_state_dict(state_dict['framework_opt'])
        self.opt.load_state_dict(state_dict['opt'])

    def fit(self, args,
            meta_tasks_train: List[Dict[str, Union[pd.Index, torch.Tensor, np.ndarray]]],
            meta_tasks_val: List[Dict[str, Union[pd.Index, torch.Tensor, np.ndarray]]]):

        self.cnt = 0
        self.framework.train()
        torch.set_grad_enabled(True)

        best_ic = -1e3
        patience = 0
        best_checkpoint = copy.deepcopy(self.framework.state_dict())

        for epoch in tqdm(range(args.n_epochs), desc="epoch"):
            for phase, task_list in zip(['train', 'val'], [meta_tasks_train, meta_tasks_val]):
                if phase == "test":
                    if epoch < self.begin_valid_epoch:
                        continue
                pred_y, ic = self.run_epoch(phase = phase, task_list=task_list, args = args)
                if phase == "val":
                    if ic < best_ic:
                        patience += 1
                    else:
                        best_ic = ic
                        # print("best ic:", best_ic)
                        patience = 0
                        best_checkpoint = copy.deepcopy(self.framework.state_dict())
            if patience > args.early_stop:
                print("early stop")
                break
        self.framework.load_state_dict(best_checkpoint)
        self.run_epoch(phase = 'train', task_list=meta_tasks_val, args = args)
        self.fitted = True
        if args.model_save_path:
            print('Save model in:', args.model_save_path)
            # torch.save(self.state_dict(), checkpoint_path)
            torch.save(self.state_dict(), args.model_save_path + '/model.bin')

    def run_epoch(self, args, phase: str, task_list: List[Dict[str, Union[pd.Index, torch.Tensor, np.ndarray]]],
                  tqdm_show: bool=False):
        pred_y_all, mse_all = [], 0
        indices = np.arange(len(task_list))
        if phase == "val":
            checkpoint = copy.deepcopy(self.framework.state_dict())
            checkpoint_opt = copy.deepcopy(self.framework.opt.state_dict())
            checkpoint_opt_meta = copy.deepcopy(self.opt.state_dict())
        elif phase == "train":
            np.random.shuffle(indices)
        self.phase = phase
        # print("tqdm_show", tqdm_show)
        for i in tqdm(indices, desc=phase) if tqdm_show else indices:
            # torch.cuda.empty_cache()
            meta_input = task_list[i]
            if not isinstance(meta_input['X_train'], torch.Tensor):
                meta_input = {
                    k: torch.tensor(v, device=args.device, dtype=torch.float32) if 'idx' not in k else v
                    for k, v in meta_input.items()
                }
            pred = self.run_task(meta_input = meta_input, phase = phase, args = args)
            if phase != "train":
                test_idx = meta_input["test_idx"]
                pred_y_all.append(
                    pd.DataFrame(
                        {
                            "pred": pd.Series(pred, index=test_idx),
                            "label": pd.Series(meta_input["y_test"], index=test_idx),
                        }
                    )
                )
        if phase == "val":
            self.framework.load_state_dict(checkpoint)
            self.framework.opt.load_state_dict(checkpoint_opt)
            self.opt.load_state_dict(checkpoint_opt_meta)
        if phase != "train":
            pred_y_all = pd.concat(pred_y_all)
        if phase == "val":
            ic = pred_y_all.groupby("datetime").apply(lambda df: df["pred"].corr(df["label"], method="pearson")).mean()
            # print(ic)
            return pred_y_all, ic
        return pred_y_all, None

    def run_task(self, args, meta_input: Dict[str, Union[pd.Index, torch.Tensor]], phase: str):
        """ Naive incremental learning """
        self.framework.opt.zero_grad()
        y_hat = self.framework(meta_input["X_train"].to(args.device), None, transform=False)

        loss = self.framework.criterion(y_hat, meta_input["y_train"].to(args.device))

        loss.backward()
        self.framework.opt.step()
        self.framework.opt.zero_grad()
        # print("meta_input[X_train]", meta_input["X_train"])
        # print("meta_input[X_test]", meta_input["X_test"])
        with torch.no_grad():
            pred = self.framework(meta_input["X_test"].to(args.device), None, transform=False)
        return pred.detach().cpu().numpy()

    def inference(self, args, meta_tasks_test: List[Dict[str, Union[pd.DataFrame, np.ndarray, torch.Tensor]]],
                  date_slice: slice = slice(None, None)):
        """
        Perform incremental learning on the test data.

        Parameters
        ----------
        meta_tasks_test (List[Dict[str, Union[pd.DataFrame, np.ndarray, torch.Tensor]]])
        test_start (Optional[pd.Timestamp]), test_end (Optional[pd.Timestamp]):
            Only return predictions and labels during this range

        Returns
        -------
        pd.DataFrame:
            the index col is pd.MultiIndex with the datetime as level 0 and the stock ID as level 1;
            the col named 'pred' contains the predictions of the model;
            the col named 'label' contains the ground-truth labels which have been preprocessed and may not be the raw.

        """
        self.framework.train()
        pred_y_all, ic = self.run_epoch(phase="online", task_list=meta_tasks_test, tqdm_show=True, args=args)
        pred_y_all = pred_y_all.loc[date_slice]
        return pred_y_all


class DoubleAdaptManager(IncrementalManager):
    def __init__(
        self,
        model: nn.Module,
        lr_model: float = 0.001,
        lr_da: float = 0.01,
        lr_ma: float = 0.001,
        reg: float = 0.5,
        adapt_x: bool = True,
        adapt_y: bool = True,
        first_order: bool = True,
        is_rnn: bool = False,
        factor_num: int = 6,
        x_dim: int = 360,
        need_permute: bool = True,
        num_head: int = 8,
        temperature: float = 10,
        begin_valid_epoch: int = 0,
        args = None
    ):
        super(DoubleAdaptManager, self).__init__(model, x_dim=x_dim, lr_model=lr_model,
                                                 is_rnn=is_rnn, need_permute=need_permute,
                                                 begin_valid_epoch=begin_valid_epoch, args=args)
        self.lr_da = lr_da
        self.lr_ma = lr_ma
        self.adapt_x = adapt_x
        self.adapt_y = adapt_y
        self.reg = reg
        self.sigma = 1 ** 2 * 2
        self.factor_num = factor_num
        self.lamda = 0.5
        self.num_head = num_head
        self.temperature = temperature
        self.first_order = first_order
        self.begin_valid_epoch = begin_valid_epoch
        self.args = args

    def _init_framework(self, model: nn.Module, x_dim=None, lr_model=0.001, need_permute=False,
                        num_head=8, temperature=10, factor_num=6, args = None, **kwargs):
        return DoubleAdapt(
            model, x_dim=x_dim, lr=lr_model, need_permute=need_permute,
            factor_num=factor_num, num_head=num_head, temperature=temperature, args=args
        )

    def _init_meta_optimizer(self, lr_da=0.01, **kwargs):
        return optim.Adam(self.framework.meta_params, lr=lr_da)    # To optimize the data adapter
        # return optim.Adam([{'params': self.tn.teacher_y.parameters(), 'lr': self.lr_y},
        #                    {'params': self.tn.teacher_x.parameters()}], lr=self.lr)

    def run_task(self, args, meta_input: Dict[str, Union[pd.Index, torch.Tensor]], phase: str):

        self.framework.opt.zero_grad()
        self.opt.zero_grad()
        """ Incremental data adaptation & Model adaptation """
        X = meta_input["X_train"].to(args.device)

        with higher.innerloop_ctx(
            self.framework.model,
            self.framework.opt,
            copy_initial_weights=False,
            track_higher_grads=not self.first_order,
            override={'lr': [self.lr_model]}
        ) as (fmodel, diffopt):
            with torch.backends.cudnn.flags(enabled=self.first_order or not self.is_rnn):
                y_hat, _ = self.framework(X, model=fmodel, transform=self.adapt_x)
        y = meta_input["y_train"].to(args.device)

        if self.adapt_y:
            raw_y = y
            y = self.framework.teacher_y(X, raw_y, inverse=False)
        loss2 = self.framework.criterion(y_hat, y)
        diffopt.step(loss2)

        """ Online inference """
        if phase != "train" and "X_extra" in meta_input and meta_input["X_extra"].shape[0] > 0:
            X_test = torch.cat([meta_input["X_extra"].to(args.device), meta_input["X_test"].to(args.device), ], 0, )
            y_test = torch.cat([meta_input["y_extra"].to(args.device), meta_input["y_test"].to(args.device), ], 0, )
        else:
            X_test = meta_input["X_test"].to(args.device)
            y_test = meta_input["y_test"].to(args.device)

        pred, X_test_adapted = self.framework(X_test, model=fmodel, transform=self.adapt_x)
        mask_y = meta_input.get("mask_y")
        if self.adapt_y:
            pred = self.framework.teacher_y(X_test, pred, inverse=True)
        if phase != "train":
            test_begin = len(meta_input["y_extra"]) if "y_extra" in meta_input else 0
            meta_end = test_begin + meta_input["meta_end"]
            output = pred[test_begin:].detach().cpu().numpy()
            X_test = X_test[:meta_end]
            X_test_adapted = X_test_adapted[:meta_end]
            if mask_y is not None:
                pred = pred[mask_y]
                meta_end = sum(mask_y[:meta_end])
            pred = pred[:meta_end]
            y_test = y_test[:meta_end]
        else:
            output = pred.detach().cpu().numpy()

        """ Optimization of meta-learners """
        loss = self.framework.criterion(pred, y_test)
        if self.adapt_y:
            if not self.first_order:
                y = self.framework.teacher_y(X, raw_y, inverse=False)
            loss_y = F.mse_loss(y, raw_y)
            if self.first_order:
                """ Please refer to Appendix C in https://arxiv.org/pdf/2306.09862.pdf """
                with torch.no_grad():
                    pred2, _ = self.framework(X_test_adapted, model=None, transform=False, )
                    pred2 = self.framework.teacher_y(X_test, pred2, inverse=True).detach()
                    loss_old = self.framework.criterion(pred2.view_as(y_test), y_test)
                loss_y = (loss_old.item() - loss.item()) / self.sigma * loss_y + loss_y * self.reg
            else:
                loss_y = loss_y * self.reg
            loss_y.backward()
        loss.backward()
        if self.adapt_x or self.adapt_y:
            self.opt.step()
        self.framework.opt.step()
        return output
