from dataclasses import dataclass, field
import sys
from typing import List, Dict
import os

import torch
from dataclasses import dataclass, asdict, field
from torch_timeseries.nn.embedding import freq_map
from src.models.StaTS import StaTS_Forecasting
from src.experiments.prob_forecast import ProbForecastExp
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from tqdm import tqdm
from torch_timeseries.utils.model_stats import count_parameters
from torch_timeseries.utils.reproduce import reproducible
import time
# import multiprocessing
import torch.multiprocessing as mp

import numpy as np
import torch.distributed as dist
import torch
from tqdm import tqdm
import concurrent.futures
# import wandb


@dataclass
class StaTSParameters:
    layers:  int = 4  # 4
    channels: int =  64 
    nheads:  int = 8
    diffusion_embedding_dim: int =  128
    beta_start: float =  1e-5
    beta_end: float =  0.1
    num_steps: int = 50
    is_linear: bool =  True
    is_unconditional: int = 0
    timeemb: int =  128
    featureemb: int =  16
    num_samples: int =  100
    target_strategy: str =  "test"
    num_sample_features: int =  64


@dataclass
class CSDIForecast(ProbForecastExp, StaTSParameters):
    model_type: str = "StaTS"
    init_schedule: str = "Linear"
    def _init_model(self):
        configs = {
            "diffusion": {
                "layers": self.layers,
                "channels": self.channels,
                "nheads": self.nheads,
                "diffusion_embedding_dim": self.diffusion_embedding_dim,
                "beta_start": self.beta_start,
                "beta_end": self.beta_end,
                "num_steps": self.num_steps,
                "schedule": self.init_schedule,
                "is_linear": self.is_linear,
            },
            "model":{
                "pred_len": self.pred_len,
                "windows": self.windows,
                "is_unconditional": self.is_unconditional,
                "timeemb": self.timeemb,
                "featureemb": self.featureemb,
                "target_strategy": self.target_strategy,
                "num_sample_features": self.num_sample_features, 
                "freq": self.dataset.freq,
                "dataset": self.dataset_type,
                "warm_epochs": self.warm_epochs
            }
        }
        self.model = StaTS_Forecasting(
            config=configs,
            device=self.device,
            target_dim=self.dataset.num_features
            )
        self.model = self.model.to(self.device)
        
        
        self.gt_mask = torch.concat([
                torch.ones(size=(self.windows, self.dataset.num_features)),
                torch.zeros(size=(self.pred_len, self.dataset.num_features)),
            ]).to(self.device).bool()
        
        self.observation_mask = ~self.gt_mask
        
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

                pred, true, main_loss, _ = self._process_train_batch(
                        batch_x, batch_y, batch_x_date_enc, batch_y_date_enc
                )
                
                if self.invtrans_loss:
                    pred = self.scaler.inverse_transform(pred)
                    true = origin_y

                self.model_optim.zero_grad()
                
                total_loss = main_loss 
                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                
                self.model_optim.step()

                progress_bar.update(batch_x.size(0))
                train_loss.append(main_loss.item())
                progress_bar.set_postfix(
                    loss=main_loss.item(),
                    lr=self.model_optim.param_groups[0]["lr"],
                    epoch=self.current_epoch,
                    refresh=True,
                )

            return train_loss


    def _process_train_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):
        
        
        batch_input = {
            "x": batch_x, 
            "y": batch_y,
            "x_mark": batch_x_date_enc,
            "y_mark": batch_y_date_enc,
            "observed_mask": self.observation_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1),
            "gt_mask": self.gt_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1),
        }
        noise = self.model(batch_input, self.current_epoch, is_train=self.num_samples)
        return noise


    def _process_val_batch(self, batch_x, batch_y, batch_x_date_enc, batch_y_date_enc):

        batch_input = {
            "x": batch_x,
            "y": batch_y,
            "x_mark": batch_x_date_enc,
            "y_mark": batch_y_date_enc,
            "observed_mask": self.observation_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1),
            "gt_mask": self.gt_mask.unsqueeze(0).expand(batch_x.shape[0], -1, -1),
        }

        samples = self.model.evaluate(batch_input, self.num_samples, 1)
        return samples.permute(0, 2, 3, 1), batch_y


    def run(self, seed=42) -> Dict[str, float]:
        
        
        self._setup_run(seed)
        self._check_run_exist(seed)

        self._run_print(f"run : {self.current_run} in seed: {seed}")


        while self.current_epoch < self.epochs:

            epoch_start_time = time.time()
            if self.early_stopper.early_stop is True:
                self._run_print(
                    f"val loss no decreased for patience={self.patience} epochs,  early stopping ...."
                )
                break

            self.model.train()

            other_params = [
                p for n, p in self.model.named_parameters()
                    if not n.startswith("beta_scheduler")
            ]

            for p in other_params:
                p.requires_grad = True

            for p in self.model.beta_scheduler.parameters():
                p.requires_grad = False

            reproducible(seed + self.current_epoch)

            train_losses = self._train()
            self._run_print(
                "Epoch: {} cost time: {}s".format(
                    self.current_epoch + 1, time.time() - epoch_start_time
                )
            )
            self._run_print(f"Traininng loss : {np.mean(train_losses)}")

            if self.current_epoch > 2:
                val_result = self._val()

                self.current_epoch = self.current_epoch + 1
                self.early_stopper(val_result['crps'], model=self.model)

                self._save_run_check_point(seed)

            print("Schedule Training .... ")
            if (self.current_epoch % 1 == 0) and (self.current_epoch < 3):#:
                for p in self.model.beta_scheduler.parameters():
                    p.requires_grad = True
                            
                for p in other_params:
                    p.requires_grad = False

                train_energy_term, train_beta_term, train_schedule_loss = self._train_schedule(self.model_type)

                self._run_print(f"Schedule Traininng loss : {np.mean(train_energy_term)}")
                self._run_print(f"Beta loss : {np.mean(train_beta_term)}")
                self._run_print(f"Total Traininng loss : {np.mean(train_schedule_loss)}")

                self.current_epoch = self.current_epoch + 1

        self._load_best_model()
        epoch_start_testing_time = time.time()
        best_test_result = self._test()
        self._run_print(
                "Testing cost time: {}s".format(
                    time.time() - epoch_start_testing_time
                )
            )
        return best_test_result
    
    
if __name__ == "__main__":
    import fire
    fire.Fire(CSDIForecast)