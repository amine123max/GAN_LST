import numpy as np
import matplotlib.pyplot as plt
import netCDF4
import os
from config import config
from sklearn.metrics import mean_squared_error, mean_absolute_error

def load_ground_truth(data_dir, year, file_type):
    """Load ground truth data from NC file"""
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.nc')])
    matching_files = [f for f in all_files if str(year) in f and file_type in f]
    
    if len(matching_files) == 0:
        return None, None, None
    
    file_path = os.path.join(data_dir, matching_files[0])
    
    with netCDF4.Dataset(file_path) as nc_file:
        lst = nc_file.variables['lst'][:]
        lat = nc_file.variables['lat'][:]
        lon = nc_file.variables['lon'][:]
    
    if hasattr(lst, 'mask'):
        lst = lst.filled(fill_value=np.nan)
    
    lst = np.squeeze(lst, axis=0)
    
    return lst, lat, lon

def get_global_min_max(data_dir):
    """Get global min/max temperature"""
    global_min = float('inf')
    global_max = float('-inf')
    file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.nc')])

    for filename in file_list:
        file_path = os.path.join(data_dir, filename)
        with netCDF4.Dataset(file_path) as nc_file:
            lst = nc_file.variables['lst'][:]
            if hasattr(lst, 'mask'):
                lst = lst.filled(fill_value=np.nan)
            
            current_min = np.nanmin(lst)
            current_max = np.nanmax(lst)
            
            if current_min < global_min:
                global_min = current_min
            if current_max > global_max:
                global_max = current_max
    
    return global_min, global_max

def visualize_predictions(prediction_dir, data_dir, output_dir=None):
    """Visualize prediction results"""
    
    # Find all prediction .npy files
    pred_files = [f for f in os.listdir(prediction_dir) if f.endswith('.npy')]
    
    if len(pred_files) == 0:
        print(f"No prediction files found in {prediction_dir}")
        return
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Get global temperature range
    global_min, global_max = get_global_min_max(data_dir)
    print(f"Global temperature range: {global_min:.2f}K - {global_max:.2f}K")
    
    # Create output directory
    if output_dir is None:
        output_dir = os.path.join(config.RESULTS_DIR, "visualizations")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each prediction file
    for pred_file in pred_files:
        # Determine file type (ASC or DES)
        file_type = 'ASC' if 'ASC' in pred_file else 'DES'
        
        # Load prediction
        pred_path = os.path.join(prediction_dir, pred_file)
        predicted_temp = np.load(pred_path)
        
        print(f"\n=== Processing {file_type} ===")
        print(f"Prediction shape: {predicted_temp.shape}")
        print(f"Temperature range: {np.nanmin(predicted_temp):.2f}K - {np.nanmax(predicted_temp):.2f}K")
        
        # Load ground truth
        true_lst, lat, lon = load_ground_truth(data_dir, 2020, file_type)
        
        if true_lst is not None:
            # Create comparison plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Resize prediction if needed
            if predicted_temp.shape != true_lst.shape:
                from scipy.ndimage import zoom
                zoom_factors = (true_lst.shape[0] / predicted_temp.shape[0], 
                               true_lst.shape[1] / predicted_temp.shape[1])
                predicted_temp = zoom(predicted_temp, zoom_factors, order=1)
            
            # Calculate metrics
            mask = ~np.isnan(true_lst) & ~np.isnan(predicted_temp)
            rmse = np.sqrt(mean_squared_error(true_lst[mask], predicted_temp[mask]))
            mae = mean_absolute_error(true_lst[mask], predicted_temp[mask])
            
            print(f"RMSE: {rmse:.4f}K")
            print(f"MAE: {mae:.4f}K")
            
            # Plot 1: Ground Truth
            im1 = axes[0].imshow(true_lst, cmap='RdYlBu_r', vmin=global_min, vmax=global_max)
            axes[0].set_title(f'Ground Truth LST 2020\n({file_type})', fontsize=14, fontweight='bold')
            axes[0].axis('off')
            cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
            cbar1.set_label('Temperature (K)', rotation=270, labelpad=20)
            
            # Add statistics
            stats_text = f"Min: {np.nanmin(true_lst):.2f}K\n"
            stats_text += f"Max: {np.nanmax(true_lst):.2f}K\n"
            stats_text += f"Mean: {np.nanmean(true_lst):.2f}K"
            axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Plot 2: Prediction
            im2 = axes[1].imshow(predicted_temp, cmap='RdYlBu_r', vmin=global_min, vmax=global_max)
            axes[1].set_title(f'Predicted LST 2020\n({file_type})', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
            cbar2.set_label('Temperature (K)', rotation=270, labelpad=20)
            
            # Add statistics
            stats_text = f"Min: {np.nanmin(predicted_temp):.2f}K\n"
            stats_text += f"Max: {np.nanmax(predicted_temp):.2f}K\n"
            stats_text += f"Mean: {np.nanmean(predicted_temp):.2f}K"
            axes[1].text(0.02, 0.98, stats_text, transform=axes[1].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Plot 3: Difference
            diff = predicted_temp - true_lst
            vmax_diff = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
            im3 = axes[2].imshow(diff, cmap='seismic', vmin=-vmax_diff, vmax=vmax_diff)
            axes[2].set_title(f'Difference (Predicted - True)\n({file_type})', 
                            fontsize=14, fontweight='bold')
            axes[2].axis('off')
            cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
            cbar3.set_label('Temperature Difference (K)', rotation=270, labelpad=20)
            
            # Add error metrics
            error_text = f"RMSE: {rmse:.2f}K\n"
            error_text += f"MAE: {mae:.2f}K\n"
            error_text += f"Max Error: {np.nanmax(np.abs(diff)):.2f}K"
            axes[2].text(0.02, 0.98, error_text, transform=axes[2].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            plt.suptitle(f'LST 2020 Comparison - {file_type}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save
            output_path = os.path.join(output_dir, f"comparison_{file_type}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved comparison to {output_path}")
            
        else:
            # Only plot prediction (no ground truth)
            fig, ax = plt.subplots(figsize=(10, 8))
            
            im = ax.imshow(predicted_temp, cmap='RdYlBu_r', vmin=global_min, vmax=global_max)
            ax.set_title(f'Predicted LST 2020 ({file_type})', fontsize=14, fontweight='bold')
            ax.axis('off')
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Temperature (K)', rotation=270, labelpad=20)
            
            # Add statistics
            stats_text = f"Min: {np.nanmin(predicted_temp):.2f}K\n"
            stats_text += f"Max: {np.nanmax(predicted_temp):.2f}K\n"
            stats_text += f"Mean: {np.nanmean(predicted_temp):.2f}K\n"
            stats_text += f"Std: {np.nanstd(predicted_temp):.2f}K"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            output_path = os.path.join(output_dir, f"prediction_{file_type}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved prediction visualization to {output_path}")
    
    print(f"\nVisualization completed! Results saved to {output_dir}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize existing prediction results")
    parser.add_argument("--prediction_dir", type=str, 
                        default=os.path.join(config.RESULTS_DIR, "predictions"),
                        help="Directory containing prediction .npy files")
    parser.add_argument("--data_dir", type=str, default=config.DATA_DIR,
                        help="Directory containing ground truth .nc files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save visualizations")
    args = parser.parse_args()
    
    visualize_predictions(args.prediction_dir, args.data_dir, args.output_dir)

if __name__ == '__main__':
    main()
