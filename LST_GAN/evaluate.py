import torch
import netCDF4
import numpy as np
import os
import matplotlib.pyplot as plt
from config import config
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse

def calculate_rmse(true, pred):
    """Calculate RMSE"""
    mask = ~np.isnan(true) & ~np.isnan(pred)
    return np.sqrt(mean_squared_error(true[mask], pred[mask]))

def calculate_mae(true, pred):
    """Calculate MAE"""
    mask = ~np.isnan(true) & ~np.isnan(pred)
    return mean_absolute_error(true[mask], pred[mask])

def calculate_ssim(true, pred):
    """Calculate SSIM (Structural Similarity Index)"""
    try:
        from skimage.metrics import structural_similarity as ssim
        mask = ~np.isnan(true) & ~np.isnan(pred)
        true_clean = np.nan_to_num(true, nan=0.0)
        pred_clean = np.nan_to_num(pred, nan=0.0)
        
        # Normalize to [0, 1] for SSIM
        true_norm = (true_clean - true_clean.min()) / (true_clean.max() - true_clean.min() + 1e-8)
        pred_norm = (pred_clean - pred_clean.min()) / (pred_clean.max() - pred_clean.min() + 1e-8)
        
        return ssim(true_norm, pred_norm, data_range=1.0)
    except ImportError:
        print("Warning: scikit-image not installed. SSIM calculation skipped.")
        return None

def load_nc_file(file_path):
    """Load NetCDF file"""
    with netCDF4.Dataset(file_path) as nc_file:
        lst = nc_file.variables['lst'][:]
        lat = nc_file.variables['lat'][:]
        lon = nc_file.variables['lon'][:]
    
    if hasattr(lst, 'mask'):
        lst = lst.filled(fill_value=np.nan)
    
    lst = np.squeeze(lst, axis=0)
    return lst, lat, lon

def visualize_comparison(true, pred, title, output_path):
    """Create comparison visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # True LST
    im1 = axes[0].imshow(true, cmap='RdYlBu_r', vmin=np.nanmin(true), vmax=np.nanmax(true))
    axes[0].set_title('True LST 2020')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Predicted LST
    im2 = axes[1].imshow(pred, cmap='RdYlBu_r', vmin=np.nanmin(true), vmax=np.nanmax(true))
    axes[1].set_title('Predicted LST 2020')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Difference
    diff = pred - true
    vmax_diff = max(abs(np.nanmin(diff)), abs(np.nanmax(diff)))
    im3 = axes[2].imshow(diff, cmap='seismic', vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_title('Difference (Predicted - True)')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_path}")

def evaluate_predictions(prediction_dir, output_dir=None):
    """Evaluate predictions against ground truth"""
    if output_dir is None:
        output_dir = os.path.join(config.RESULTS_DIR, "evaluation")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find prediction files
    pred_files = sorted([f for f in os.listdir(prediction_dir) if f.endswith('.npy')])
    
    if len(pred_files) == 0:
        print(f"Error: No prediction files found in {prediction_dir}")
        return
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Find corresponding ground truth files
    all_nc_files = sorted([f for f in os.listdir(config.DATA_DIR) if f.endswith('.nc')])
    true_2020_files = [f for f in all_nc_files if '2020' in f]
    
    print(f"Found {len(true_2020_files)} ground truth files for 2020")
    
    results = []
    
    for pred_file in pred_files:
        # Load prediction
        pred_path = os.path.join(prediction_dir, pred_file)
        pred_lst = np.load(pred_path)
        
        # Find corresponding true file
        # Extract file type (ASC or DES) from prediction filename
        file_type = 'ASC' if 'ASC' in pred_file else 'DES'
        true_file = [f for f in true_2020_files if file_type in f]
        
        if len(true_file) == 0:
            print(f"Warning: No matching ground truth file for {pred_file}")
            continue
        
        true_file = true_file[0]
        true_path = os.path.join(config.DATA_DIR, true_file)
        true_lst, lat, lon = load_nc_file(true_path)
        
        # Resize prediction to match ground truth shape
        if pred_lst.shape != true_lst.shape:
            from scipy.ndimage import zoom
            zoom_factors = (true_lst.shape[0] / pred_lst.shape[0], 
                           true_lst.shape[1] / pred_lst.shape[1])
            pred_lst = zoom(pred_lst, zoom_factors, order=1)
        
        # Calculate metrics
        rmse = calculate_rmse(true_lst, pred_lst)
        mae = calculate_mae(true_lst, pred_lst)
        ssim_val = calculate_ssim(true_lst, pred_lst)
        
        print(f"\n{file_type} Results:")
        print(f"  RMSE: {rmse:.4f} K")
        print(f"  MAE:  {mae:.4f} K")
        if ssim_val is not None:
            print(f"  SSIM: {ssim_val:.4f}")
        
        # Visualize
        vis_output = os.path.join(output_dir, f"comparison_{file_type}.png")
        title = f"LST Comparison {file_type} - RMSE: {rmse:.2f}K, MAE: {mae:.2f}K"
        if ssim_val is not None:
            title += f", SSIM: {ssim_val:.3f}"
        visualize_comparison(true_lst, pred_lst, title, vis_output)
        
        results.append({
            'file_type': file_type,
            'rmse': rmse,
            'mae': mae,
            'ssim': ssim_val
        })
    
    # Summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    if len(results) > 0:
        avg_rmse = np.mean([r['rmse'] for r in results])
        avg_mae = np.mean([r['mae'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results if r['ssim'] is not None])
        
        print(f"Average RMSE: {avg_rmse:.4f} K")
        print(f"Average MAE:  {avg_mae:.4f} K")
        if not np.isnan(avg_ssim):
            print(f"Average SSIM: {avg_ssim:.4f}")
    
    # Save results to file
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("="*50 + "\n\n")
        for r in results:
            f.write(f"{r['file_type']}:\n")
            f.write(f"  RMSE: {r['rmse']:.4f} K\n")
            f.write(f"  MAE:  {r['mae']:.4f} K\n")
            if r['ssim'] is not None:
                f.write(f"  SSIM: {r['ssim']:.4f}\n")
            f.write("\n")
        if len(results) > 0:
            f.write("AVERAGE:\n")
            f.write(f"  RMSE: {avg_rmse:.4f} K\n")
            f.write(f"  MAE:  {avg_mae:.4f} K\n")
            if not np.isnan(avg_ssim):
                f.write(f"  SSIM: {avg_ssim:.4f}\n")
    
    print(f"\nResults saved to {results_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate LST predictions")
    parser.add_argument("--prediction_dir", type=str, required=True,
                        help="Directory containing prediction .npy files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results")
    args = parser.parse_args()
    
    evaluate_predictions(args.prediction_dir, args.output_dir)

if __name__ == '__main__':
    main()
