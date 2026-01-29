# import codecs
from dataclasses import asdict, dataclass
import datetime
import hashlib
import json
import os
import random
import time
from typing import Dict, List, Type, Union

import numpy as np
import pandas as pd
import torch
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch.nn import MSELoss, L1Loss
from torch.optim import *
from torch_timeseries.dataset import *
from src.datasets import *
from torch_timeseries.scaler import *
from src.metrics import CRPS, CRPSSum, QICE, PICP
from src.metrics import ProbMAE, ProbMSE, ProbRMSE
from src.experiments.Setup import ForecastExp

from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.early_stop import EarlyStopping
from torch_timeseries.utils.parse_type import parse_type
from torch_timeseries.utils.reproduce import reproducible
from torch_timeseries.core import TimeSeriesDataset, BaseIrrelevant, BaseRelevant
from torch_timeseries.dataloader import SlidingWindowTS, ETTHLoader, ETTMLoader
from torch_timeseries.utils import asdict_exc
import torch.multiprocessing as mp

import math

import matplotlib.ticker as mticker

try:
    import wandb
except:
    print("Warning: wandb is not installed, some funtionality may not work.")



def update_metrics(preds, truths, metrics):
    metrics.update(preds, truths)

def make_beta_schedule_np(
    schedule: str,
    T: int,
    beta_start: float,
    beta_end: float,
    cosine_s: float = 0.008,
    k: float = 1.0,
):
    schedule = (schedule or "linear").lower()

    if schedule in ["linear", "lin"]:
        betas = np.linspace(beta_start, beta_end, T)

    elif schedule in ["quad", "quadratic"]:
        betas = np.linspace(math.sqrt(beta_start), math.sqrt(beta_end), T) ** 2

    elif schedule in ["cosine", "cos"]:
        steps = T + 1
        x = np.linspace(0, T, steps)  
        alphas_cumprod = np.cos(((x / T) + cosine_s) / (1 + cosine_s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas_raw = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas_raw = np.clip(betas_raw, 0.0, 0.999)

        b0, b1 = float(betas_raw[0]), float(betas_raw[-1])
        betas = (betas_raw - b0) / (b1 - b0 + 1e-12)
        betas = betas * (beta_end - beta_start) + beta_start

    elif schedule in ["jsd"]:
        betas_raw = 1.0 / np.linspace(T, 1, T)

        b0, b1 = float(betas_raw[0]), float(betas_raw[-1])
        betas = (betas_raw - b0) / (b1 - b0 + 1e-12)
        betas = betas * (beta_end - beta_start) + beta_start

    elif schedule in ["const", "constant"]:
        t = np.linspace(0.0, 1.0, T)
        y = 1.0 - np.exp(-k * t)
        y = (y - y[0]) / (y[-1] - y[0] + 1e-12)
        betas = y * (beta_end - beta_start) + beta_start

    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    betas = np.asarray(betas, dtype=np.float64)
    betas = np.clip(betas, 1e-12, 0.999)
    return betas


@dataclass
class ProbForecastExp(ForecastExp):
    loss_func_type : str = 'mse'
    epochs : int = 20
    warm_epochs : int = 5
    
    def _init_metrics(self):
        self.metrics = MetricCollection(
            metrics={
                "crps": CRPS(),
                "crps_sum": CRPSSum(),
                "qice": QICE(),
                "picp": PICP(),
                "mse": ProbMSE(),
                "mae": ProbMAE(),
                "rmse": ProbRMSE(),
            }
        )
        self.metrics.to("cpu")
        ctx = mp.get_context("spawn")  
        self.task_pool = ctx.Pool(processes=32)

    def _init_dataset(self):
        self.dataset: TimeSeriesDataset = parse_type(self.dataset_type, globals())(
            root=self.data_path
        )
    
    def _train_schedule(self, model_type="StaTS"):
        lambda_energy = self.lambda_energy
        lambda_schedule = self.lambda_schedule

        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            train_schedule_loss = []
            train_energy_term = []  
            train_beta_term = []     

            for (batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc) in self.train_loader:
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

            
                _, _, main_loss, beta_loss = self._process_train_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )

                self.model_optim_beta.zero_grad()

                energy_term = main_loss * lambda_energy
                beta_term = beta_loss * lambda_schedule
                schedule_loss = beta_term + energy_term

                schedule_loss.backward()
                self.model_optim_beta.step()

                progress_bar.update(batch_x.size(0))
                train_schedule_loss.append(schedule_loss.item())
                train_energy_term.append(energy_term.item())
                train_beta_term.append(beta_term.item())

                progress_bar.set_postfix(
                    energy_term=energy_term.item(),
                    beta_term=beta_term.item(),
                    total=schedule_loss.item(),
                    refresh=True,
                )

        return train_energy_term, train_beta_term, train_schedule_loss
            
    def _train(self, model_name = "StaTS"):

        with torch.enable_grad(), tqdm(total=len(self.train_loader.dataset)) as progress_bar:
            train_loss = []

            for i, (
                batch_x, batch_y, origin_x, origin_y,
                batch_x_date_enc, batch_y_date_enc
            ) in enumerate(self.train_loader):

                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                origin_y = origin_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()

                pred, true = self._process_train_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                
                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = origin_y

                self.model_optim.zero_grad()
                loss = self.loss_func(pred, true)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                
                self.model_optim.step()

                progress_bar.update(batch_x.size(0))
                train_loss.append(loss.item())
                progress_bar.set_postfix(
                    loss=loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )

            return train_loss
        
    def _process_train_batch(
        self,
        batch_x,
        batch_y,
        batch_origin_x,
        batch_origin_y,
        batch_x_date_enc,
        batch_y_date_enc,
    ):
        raise NotImplementedError()

    def _process_val_batch(
        self,
        batch_x,
        batch_origin_x,
        batch_x_date_enc,
        batch_y_date_enc,
    ):
        raise NotImplementedError()
    
    
    def _evaluate(self, dataloader):
        self.model.eval()
        self.metrics.reset()
        results = []
        with tqdm(total=len(dataloader.dataset)) as progress_bar:
            for batch_x, batch_y, origin_x, origin_y, batch_x_date_enc, batch_y_date_enc in dataloader:
                batch_size = batch_x.size(0)
                origin_x = origin_x.to(self.device)
                origin_y = origin_y.to(self.device)
                batch_x = batch_x.to(self.device).float()
                batch_y = batch_y.to(self.device).float()
                batch_x_date_enc = batch_x_date_enc.to(self.device).float()
                batch_y_date_enc = batch_y_date_enc.to(self.device).float()
                preds, truths = self._process_val_batch(
                    batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                origin_y = origin_y.to(self.device)
                if self.invtrans_loss:
                    preds = self.scaler.inverse_transform(preds)
                    truths = origin_y
                    
                # update_metrics(preds.contiguous().cpu().detach(), truths.contiguous().cpu().detach(), self.metrics)
                # if isinstance(preds, np.ndarray):
                #     results.append(self.task_pool.apply_async(update_metrics, (preds, truths, self.metrics)))
                # else:
                results.append(self.task_pool.apply_async(update_metrics, (preds.contiguous().cpu().detach(), truths.contiguous().cpu().detach(), self.metrics)))
                
                progress_bar.update(batch_x.shape[0])

        for result in results:
            result.get()  # Ensure the metric update is finished

        result = {name: float(metric.compute()) for name, metric in self.metrics.items()}
        return result
    
    
    
    def _init_data_loader(self, shuffle=True, fast_test=True, fast_val=True):
        
        self._init_dataset()
        
        self.scaler = parse_type(self.scaler_type, globals=globals())()
        if self.dataset_type[0:3] == "ETT":
            if self.dataset_type[0:4] == "ETTh":
                self.dataloader = ETTHLoader(
                    self.dataset,
                    self.scaler,
                    window=self.windows,
                    horizon=self.horizon,
                    steps=self.pred_len,
                    shuffle_train=shuffle,
                    freq=self.dataset.freq,
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                    fast_test=fast_test,
                    fast_val=fast_val,
                )
            elif  self.dataset_type[0:4] == "ETTm":
                self.dataloader = ETTMLoader(
                    self.dataset,
                    self.scaler,
                    window=self.windows,
                    horizon=self.horizon,
                    steps=self.pred_len,
                    shuffle_train=shuffle,
                    freq=self.dataset.freq,
                    batch_size=self.batch_size,
                    num_worker=self.num_worker,
                    fast_test=fast_test,
                    fast_val=fast_val,
                )
        else:
            self.dataloader = SlidingWindowTS(
                self.dataset,
                self.scaler,
                window=self.windows,
                horizon=self.horizon,
                steps=self.pred_len,
                scale_in_train=True,
                shuffle_train=shuffle,
                freq=self.dataset.freq,
                batch_size=self.batch_size,
                train_ratio=self.train_ratio,
                test_ratio=self.test_ratio,
                num_worker=self.num_worker,
                fast_test=fast_test,
                fast_val=fast_val,
            )

        self.train_loader, self.val_loader, self.test_loader = (
            self.dataloader.train_loader,
            self.dataloader.val_loader,
            self.dataloader.test_loader,
        )
        self.train_steps = len(self.train_loader.dataset)
        self.val_steps = len(self.val_loader.dataset)
        self.test_steps = len(self.test_loader.dataset)

        print(f"train steps: {self.train_steps}")
        print(f"val steps: {self.val_steps}")
        print(f"test steps: {self.test_steps}")
        

    def _test(self) -> Dict[str, float]:
        print("Testing .... ")
        test_result = self._evaluate(self.test_loader)

        self._run_print(f"test_results: {test_result}")
        return test_result

    def _val(self):
        print("Validating .... ")
        val_result = self._evaluate(self.val_loader)

        self._run_print(f"vali_results: {val_result}")
        return val_result


    def _check_run_exist(self, seed: str):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
            print(f"Creating running results saving dir: '{self.run_save_dir}'.")
        else:
            print(f"result directory exists: {self.run_save_dir}")
        with open(
            os.path.join(self.run_save_dir, "args.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=4)

        exists = os.path.exists(self.run_checkpoint_filepath)
        return exists

    def _load_best_model(self):
        self.model.load_state_dict(
            torch.load(self.best_checkpoint_filepath, map_location=self.device)
        )

    def _run_print(self, *args, **kwargs):
        time = (
            "["
            + str(datetime.datetime.now() + datetime.timedelta(hours=8))[:19]
            + "] -"
        )
        print(*args, **kwargs)
        
        with open(os.path.join(self.run_save_dir, "output.log"), "a+") as f:
            print(time, *args, flush=True, file=f)

    def _resume_run(self, seed):
        run_checkpoint_filepath = os.path.join(self.run_save_dir, f"run_checkpoint.pth")
        print(f"resuming from {run_checkpoint_filepath}")

        check_point = torch.load(run_checkpoint_filepath, map_location=self.device)

        self.model.load_state_dict(check_point["model"])
        self.model_optim.load_state_dict(check_point["optimizer"])
        self.current_epoch = check_point["current_epoch"]

        self.early_stopper.set_state(check_point["early_stopping"])

    def _use_wandb(self):
        return hasattr(self, "wandb")

    def run(self, seed=42) -> Dict[str, float]:
        
        
        self._setup_run(seed)
        if self._check_run_exist(seed):
            self._resume_run(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")

        parameter_tables, model_parameters_num = count_parameters(self.model)
        self._run_print(f"parameter_tables: {parameter_tables}")
        self._run_print(f"model parameters: {model_parameters_num}")

        self.model.train()
        while self.current_epoch < self.epochs:

            epoch_start_time = time.time()
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"val loss no decreased for patience={self.patience} epochs,  early stopping ...."
                )
                break

            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Traininng loss : {np.mean(train_losses)}")

            val_result = self._val()
            test_result = self._test()

            self.current_epoch = self.current_epoch + 1
            self.early_stopper(val_result['crps'], model=self.model)

            self._save_run_check_point(seed)

        self._load_best_model()
        best_test_result = self._test()
        return best_test_result
    

    def runs(self, seeds: List[int] = [1, 2, 3, 4, 5]):
        results = []
        for i, seed in enumerate(seeds):
            result = self.run(seed=seed)
            results.append(result)

        return results

    def _save_run_check_point(self, seed):
        if not os.path.exists(self.run_save_dir):
            os.makedirs(self.run_save_dir)
        print(f"Saving run checkpoint to '{self.run_save_dir}'.")

        self.run_state = {
            "model": self.model.state_dict(),
            "current_epoch": self.current_epoch,
            "optimizer": self.model_optim.state_dict(),
            "rng_state": torch.get_rng_state(),
            "early_stopping": self.early_stopper.get_state(),
        }

        torch.save(self.run_state, f"{self.run_checkpoint_filepath}")
        print("Run state saved ... ")
