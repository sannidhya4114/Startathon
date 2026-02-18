import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_from_excel(file_path, save_name="experiment_results.png"):
    # 1. Load the data from Excel
    try:
        # read_excel automatically detects .xlsx or .xls
        # Note: 'openpyxl' must be installed: pip install openpyxl
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error: Could not read Excel file. {e}")
        return

    # 2. Check for required columns (case-insensitive)
    df.columns = [c.lower().strip() for c in df.columns]
    required = ['train_loss', 'val_loss', 'val_iou']
    
    if not all(col in df.columns for col in required):
        print(f"Error: Excel sheet must contain columns: {required}")
        print(f"Found: {list(df.columns)}")
        return

    # 3. Setup Plotting
    epochs = df['epoch'] if 'epoch' in df.columns else range(1, len(df) + 1)
    
    plt.style.use('seaborn-v0_8-muted')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Loss ---
    ax1.plot(epochs, df['train_loss'], color='#1f77b4', label='Train Loss', linewidth=2)
    ax1.plot(epochs, df['val_loss'], color='#d62728', label='Val Loss', linestyle='--', linewidth=2)
    ax1.set_title('Model Convergence (Loss)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: IoU ---
    ax2.plot(epochs, df['val_iou'], color='#2ca02c', label='Val IoU', linewidth=2)
    ax2.set_title('Segmentation Performance (IoU)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('IoU Score')
    ax2.set_ylim(0, 1) # Standard for IoU
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"Plot saved as {save_name}")
    plt.show()

if __name__ == "__main__":
    # Updated default to .xlsx
    target_file = sys.argv[1] if len(sys.argv) > 1 else 'Results.xlsx'
    
    if os.path.exists(target_file):
        plot_from_excel(target_file)
    else:
        print(f"File '{target_file}' not found. Please provide a valid Excel file.")