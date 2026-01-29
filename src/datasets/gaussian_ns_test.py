import numpy as np
import pandas as pd
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, download_url
import torch
from typing import Any, Callable, List, Optional
import os
import resource
from torch_timeseries.core.dataset.dataset import Dataset, TimeSeriesDataset


class GaussianNS(TimeSeriesDataset):
    
    name: str = 'GaussianNS'
    num_features: int = 1
    freq: str = 'yd'  
    length: int = 7588
    
    def generate_synthetic_data(self) -> np.ndarray:
        
        means = np.linspace(1, 10, self.length)  
        stddev = np.linspace(1, 10, self.length)  
        self.means = means
        self.stddev = stddev
        
        data = np.zeros((self.length, self.num_features))
        for t in range(self.length):
            for i in range(self.num_features):
                data[t, i] = np.random.normal(loc=feature_mean, scale=stddev[t], size=1)
        
        return data

    
    
    def _load(self) -> np.ndarray:
        
        self.data = self.generate_synthetic_data()
        
        self.dates = pd.date_range(start="1990-01-01", periods=self.length, freq='D')
        self.dates =  pd.DataFrame({'date':self.dates})
        
        self.df = pd.DataFrame(self.data, columns=[f'feature_{i+1}' for i in range(self.num_features)], index=self.dates)
        
        return self.data

    def download(self) -> None:
        pass
