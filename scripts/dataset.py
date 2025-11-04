import pandas as pd
import torch
from torch.utils.data import IterableDataset
from torchvision.transforms import ToTensor
from normalization.normalize import Normalizer

class TartesDataset(IterableDataset):

    def __init__(self, parquet_files_groupby_year):
        super().__init__()
        self.files = parquet_files_groupby_year
        self.normalizer = Normalizer()
        self.transform = ToTensor()

    def __iter__(self):
        for year in self.files:
            for file in year:
                
                # read dataframe
                df = pd.read_parquet(file)
                # and shuffle df
                df = df.sample(axis='index', frac=1).reset_index(drop=True)

                for idx in range(len(df)):

                    # get row
                    row_values = df.iloc[idx].values
                    
                    # get nb of snow layers
                    nb_snow_layers = 50 - (df.iloc[idx].isna().sum() / 5)

                    # build snowpack 
                    # shape = (nb_snow_layers, 1, 5)
                    snowpack = np.array([[row_values[i:i+nb_snow_layers]] for i in range(0, 201, 50)])
                    # normalize snowpack
                    snowpack = self.normalizer.normalize_snowpack(snowpack)
                    # convert snowpack to tensor
                    snowpack_tensor = self.transform(snowpack)

                    # get sun
                    # shape = (3,)
                    sun = row_values[250:253]
                    # normalize shortwaves
                    sun[0] = self.normalizer.normalize_direct_sw(sun[0])
                    sun[1] = self.normalizer.normalize_diffuse_sw(sun[1])
                    # convert sun_params to tensor
                    sun_tensor = self.transform(sun)

                    # get albedo and convert to tensor
                    albedo_tensor = torch.tensor(row_values[-1])

                    # yield snowpack, sun_params, label
                    yield snowpack_tensor, sun_tensor, albedo_tensor
                
                # clean ram
                del df
