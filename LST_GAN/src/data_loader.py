import os
import torch
import netCDF4
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def _get_global_min_max(data_dir):
    """
    Calculates the global minimum and maximum 'lst' values across all NetCDF files.
    """
    global_min = float('inf')
    global_max = float('-inf')
    file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.nc')])

    for filename in file_list:
        file_path = os.path.join(data_dir, filename)
        with netCDF4.Dataset(file_path) as nc_file:
            lst = nc_file.variables['lst'][:]
            # Handle masked arrays
            if hasattr(lst, 'mask'):
                lst = lst.filled(fill_value=np.nan)
            
            current_min = np.nanmin(lst)
            current_max = np.nanmax(lst)
            
            if current_min < global_min:
                global_min = current_min
            if current_max > global_max:
                global_max = current_max
    return global_min, global_max


class LSTDataset(Dataset):
    def __init__(self, data_dir, global_min, global_max, transform=None, num_input_years=10):
        self.data_dir = data_dir
        self.global_min = global_min
        self.global_max = global_max
        self.transform = transform
        self.num_input_years = num_input_years
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.nc')])

    def __len__(self):
        return max(0, len(self.file_list) - self.num_input_years)

    def __getitem__(self, idx):
        # Load 10 consecutive years as input
        input_data_list = []
        for i in range(self.num_input_years):
            file_path = os.path.join(self.data_dir, self.file_list[idx + i])
            with netCDF4.Dataset(file_path) as nc_file:
                lst = nc_file.variables['lst'][:]
            
            if hasattr(lst, 'mask'):
                lst = lst.filled(fill_value=np.nan)
            
            lst = np.squeeze(lst, axis=0)
            
            # Min-max scale to [0, 1]
            if self.global_max - self.global_min == 0:
                lst_scaled = np.zeros_like(lst)
            else:
                lst_scaled = (lst - self.global_min) / (self.global_max - self.global_min)
            
            lst_scaled = np.nan_to_num(lst_scaled, nan=0.0)
            input_data_list.append(lst_scaled)
        
        # Load target year (year after the 10 input years)
        target_file_path = os.path.join(self.data_dir, self.file_list[idx + self.num_input_years])
        with netCDF4.Dataset(target_file_path) as nc_file:
            lst_target = nc_file.variables['lst'][:]
        
        if hasattr(lst_target, 'mask'):
            lst_target = lst_target.filled(fill_value=np.nan)
        
        lst_target = np.squeeze(lst_target, axis=0)
        
        if self.global_max - self.global_min == 0:
            target_scaled = np.zeros_like(lst_target)
        else:
            target_scaled = (lst_target - self.global_min) / (self.global_max - self.global_min)
        
        target_scaled = np.nan_to_num(target_scaled, nan=0.0)
        
        # Stack input years and apply transform
        if self.transform:
            # Transform each year separately then stack
            transformed_inputs = []
            for year_data in input_data_list:
                transformed = self.transform(year_data)
                transformed_inputs.append(transformed)
            input_tensor = torch.cat(transformed_inputs, dim=0)  # Shape: (10, H, W)
            target_tensor = self.transform(target_scaled)  # Shape: (1, H, W)
        else:
            input_tensor = torch.from_numpy(np.stack(input_data_list, axis=0)).float()
            target_tensor = torch.from_numpy(target_scaled).unsqueeze(0).float()
        
        return input_tensor, target_tensor

def get_dataloaders(data_dir, batch_size, image_size, num_workers=0, test_year=2020):
    # Calculate global min/max once for the entire dataset
    global_min, global_max = _get_global_min_max(data_dir)
    print(f"Global LST min: {global_min}, max: {global_max}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    
    # Get all .nc files and split by year
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.nc')])
    
    # Split files into train (before test_year) and test (test_year)
    train_files = [f for f in all_files if str(test_year) not in f]
    test_files = [f for f in all_files if str(test_year) in f]
    
    print(f"Total files: {len(all_files)}")
    print(f"Training files: {len(train_files)} (years before {test_year})")
    print(f"Testing files: {len(test_files)} (year {test_year})")

    class SplitLSTDataset(LSTDataset):
        def __init__(self, data_dir, file_list, global_min, global_max, transform=None, num_input_years=10):
            super().__init__(data_dir, global_min, global_max, transform, num_input_years)
            self.file_list = file_list

    train_dataset = SplitLSTDataset(data_dir, train_files, global_min, global_max, transform=transform)
    test_dataset = SplitLSTDataset(data_dir, test_files, global_min, global_max, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_dataloader, test_dataloader

# Remove the original get_dataloader function if it exists as it's replaced
# def get_dataloader(data_dir, batch_size, image_size, num_workers=0):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize((image_size, image_size)),
#         transforms.Normalize((0.5,), (0.5,)),
#     ])
#     dataset = LSTDataset(data_dir, transform=transform)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
#     return dataloader
