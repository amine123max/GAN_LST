import torch
import netCDF4
import numpy as np
import os
import matplotlib.pyplot as plt
from config import config
from src.models import Generator
from torchvision import transforms
import argparse

def load_nc_data(file_path, global_min, global_max):
    """Load and preprocess NC file"""
    with netCDF4.Dataset(file_path) as nc_file:
        lst = nc_file.variables['lst'][:]
        lat = nc_file.variables['lat'][:]
        lon = nc_file.variables['lon'][:]
    
    if hasattr(lst, 'mask'):
        lst = lst.filled(fill_value=np.nan)
    
    lst = np.squeeze(lst, axis=0)
    
    # Min-max scale to [0, 1]
    if global_max - global_min != 0:
        lst_scaled = (lst - global_min) / (global_max - global_min)
    else:
        lst_scaled = np.zeros_like(lst)
    
    lst_scaled = np.nan_to_num(lst_scaled, nan=0.0)
    
    return lst_scaled, lat, lon, lst

def predict_2020_lst(model_path, start_year=2010, end_year=2019, output_dir=None, visualize=True):
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Load generator model
    netG = Generator(config.NC, config.NGF, config.NUM_INPUT_YEARS).to(device)
    netG.load_state_dict(torch.load(model_path, map_location=device))
    netG.eval()
    print(f"Loaded generator from {model_path}")
    
    # Calculate global min/max
    from src.data_loader import _get_global_min_max
    global_min, global_max = _get_global_min_max(config.DATA_DIR)
    print(f"Global LST min: {global_min}, max: {global_max}")
    
    # Find input year files (10 years from start_year to end_year)
    all_files = sorted([f for f in os.listdir(config.DATA_DIR) if f.endswith('.nc')])
    
    # Get ASC and DES files separately
    file_types = ['ASC', 'DES']
    predictions = []
    
    for file_type in file_types:
        print(f"\n=== Processing {file_type} files ===")
        
        # Find 10 consecutive years for this file type
        input_files = []
        for year in range(start_year, end_year + 1):
            matching_files = [f for f in all_files if str(year) in f and file_type in f]
            if len(matching_files) > 0:
                input_files.append(matching_files[0])
        
        if len(input_files) != config.NUM_INPUT_YEARS:
            print(f"Warning: Expected {config.NUM_INPUT_YEARS} files but found {len(input_files)} for {file_type}")
            continue
        
        print(f"Found {len(input_files)} input files: {input_files[0]} to {input_files[-1]}")
        
        # Load and stack 10 years
        input_data_list = []
        original_shape = None
        
        for input_file in input_files:
            input_path = os.path.join(config.DATA_DIR, input_file)
            lst_scaled, lat, lon, lst_original = load_nc_data(input_path, global_min, global_max)
            
            if original_shape is None:
                original_shape = lst_original.shape
            
            input_data_list.append(lst_scaled)
        
        # Transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        
        # Stack and transform all 10 years
        transformed_inputs = []
        for year_data in input_data_list:
            transformed = transform(year_data)
            transformed_inputs.append(transformed)
        
        input_tensor = torch.cat(transformed_inputs, dim=0).unsqueeze(0).to(device)  # Shape: (1, 10, H, W)
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Generate prediction
        with torch.no_grad():
            predicted = netG(input_tensor)
        
        # Convert back to numpy
        predicted = predicted.squeeze(0).cpu()
        predicted = predicted * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]
        predicted = predicted.squeeze(0).numpy()
        
        # Resize back to original shape
        from torchvision.transforms.functional import resize
        predicted_resized = resize(torch.from_numpy(predicted).unsqueeze(0), 
                                   original_shape, 
                                   interpolation=transforms.InterpolationMode.BILINEAR)
        predicted_resized = predicted_resized.squeeze(0).numpy()
        
        # Scale back to original temperature range
        predicted_temp = predicted_resized * (global_max - global_min) + global_min
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(config.RESULTS_DIR, "predictions_10years")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save prediction
        output_file = f"LST_2020_predicted_{file_type}.npy"
        output_path = os.path.join(output_dir, output_file)
        np.save(output_path, predicted_temp)
        print(f"Saved prediction to {output_path}")
        
        predictions.append({
            'file_type': file_type,
            'prediction': predicted_temp,
            'lat': lat,
            'lon': lon,
            'output_path': output_path
        })
        
        # Visualize if requested
        if visualize:
            visualize_prediction(predicted_temp, file_type, output_dir, 
                               start_year, end_year, global_min, global_max)
    
    print(f"\n✓ Prediction completed! {len(predictions)} files predicted.")
    return predictions, output_dir

def visualize_prediction(predicted_temp, file_type, output_dir, start_year, end_year, global_min, global_max):
    """Visualize the predicted temperature map"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot prediction
    im = ax.imshow(predicted_temp, cmap='RdYlBu_r', vmin=global_min, vmax=global_max)
    ax.set_title(f'Predicted LST 2020 ({file_type})\nBased on {start_year}-{end_year}', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (K)', rotation=270, labelpad=20, fontsize=12)
    
    # Add statistics
    stats_text = f"Min: {np.nanmin(predicted_temp):.2f}K\n"
    stats_text += f"Max: {np.nanmax(predicted_temp):.2f}K\n"
    stats_text += f"Mean: {np.nanmean(predicted_temp):.2f}K\n"
    stats_text += f"Std: {np.nanstd(predicted_temp):.2f}K"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, f"visualization_2020_{file_type}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")

def compare_with_ground_truth(predictions, output_dir):
    """Compare predictions with actual 2020 data"""
    print("\n=== Comparing with Ground Truth ===")
    
    # Load ground truth 2020 data
    all_files = sorted([f for f in os.listdir(config.DATA_DIR) if f.endswith('.nc')])
    true_2020_files = [f for f in all_files if '2020' in f]
    
    from src.data_loader import _get_global_min_max
    global_min, global_max = _get_global_min_max(config.DATA_DIR)
    
    fig, axes = plt.subplots(len(predictions), 3, figsize=(18, 6*len(predictions)))
    if len(predictions) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, pred_info in enumerate(predictions):
        file_type = pred_info['file_type']
        predicted_temp = pred_info['prediction']
        
        # Find corresponding ground truth file
        true_file = [f for f in true_2020_files if file_type in f]
        
        if len(true_file) == 0:
            print(f"Warning: No ground truth file found for {file_type}")
            continue
        
        true_file = true_file[0]
        true_path = os.path.join(config.DATA_DIR, true_file)
        
        with netCDF4.Dataset(true_path) as nc_file:
            true_lst = nc_file.variables['lst'][:]
        
        if hasattr(true_lst, 'mask'):
            true_lst = true_lst.filled(fill_value=np.nan)
        
        true_lst = np.squeeze(true_lst, axis=0)
        
        # Resize prediction if needed
        if predicted_temp.shape != true_lst.shape:
            from scipy.ndimage import zoom
            zoom_factors = (true_lst.shape[0] / predicted_temp.shape[0], 
                           true_lst.shape[1] / predicted_temp.shape[1])
            predicted_temp = zoom(predicted_temp, zoom_factors, order=1)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        mask = ~np.isnan(true_lst) & ~np.isnan(predicted_temp)
        rmse = np.sqrt(mean_squared_error(true_lst[mask], predicted_temp[mask]))
        mae = mean_absolute_error(true_lst[mask], predicted_temp[mask])
        
        # Plot true
        im1 = axes[idx, 0].imshow(true_lst, cmap='RdYlBu_r', vmin=global_min, vmax=global_max)
        axes[idx, 0].set_title(f'True LST 2020 ({file_type})', fontweight='bold')
        axes[idx, 0].axis('off')
        plt.colorbar(im1, ax=axes[idx, 0], fraction=0.046, pad=0.04)
        
        # Plot predicted
        im2 = axes[idx, 1].imshow(predicted_temp, cmap='RdYlBu_r', vmin=global_min, vmax=global_max)
        axes[idx, 1].set_title(f'Predicted LST 2020 ({file_type})', fontweight='bold')
        axes[idx, 1].axis('off')
        plt.colorbar(im2, ax=axes[idx, 1], fraction=0.046, pad=0.04)
        
        # Plot difference
        diff = predicted_temp - true_lst
        vmax_diff = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
        im3 = axes[idx, 2].imshow(diff, cmap='seismic', vmin=-vmax_diff, vmax=vmax_diff)
        axes[idx, 2].set_title(f'Difference ({file_type})\nRMSE: {rmse:.2f}K, MAE: {mae:.2f}K', 
                              fontweight='bold')
        axes[idx, 2].axis('off')
        plt.colorbar(im3, ax=axes[idx, 2], fraction=0.046, pad=0.04)
        
        print(f"{file_type} - RMSE: {rmse:.4f}K, MAE: {mae:.4f}K")
    
    plt.suptitle('LST 2020: True vs Predicted (Based on 2010-2019)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    comparison_path = os.path.join(output_dir, "comparison_with_ground_truth.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison to {comparison_path}")

def main():
    parser = argparse.ArgumentParser(description="Predict 2020 LST using 10 years of data (2010-2019)")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained generator model (.pth file)")
    parser.add_argument("--start_year", type=int, default=2010,
                        help="Start year for input data (default: 2010)")
    parser.add_argument("--end_year", type=int, default=2019,
                        help="End year for input data (default: 2019)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save predictions and visualizations")
    parser.add_argument("--no_visualize", action="store_true",
                        help="Disable visualization")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with ground truth 2020 data")
    args = parser.parse_args()
    
    # Predict and visualize
    predictions, output_dir = predict_2020_lst(
        args.model_path, 
        args.start_year, 
        args.end_year,
        args.output_dir,
        visualize=not args.no_visualize
    )
    
    # Compare with ground truth if requested
    if args.compare:
        compare_with_ground_truth(predictions, output_dir)

if __name__ == '__main__':
    main()
