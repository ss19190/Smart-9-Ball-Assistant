import pandas as pd
import matplotlib.pyplot as plt
import os

# --- KONFIGURACJA ---
csv_path = r'C:\Users\saras\Desktop\thesis\Smart-9-Ball-Assistant\cue-detection\runs\pose\train\results.csv'
output_filename = 'Training_Results_HighRes_LargeFonts.pdf'
dpi = 300

# --- KONFIGURACJA STYLU (Bardzo duże czcionki) ---
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 18,           # Bazowa wielkość czcionki (było 12)
    'axes.titlesize': 28,      # Tytuły wykresów (było 14)
    'axes.labelsize': 24,      # Opisy osi X i Y (było 12)
    'xtick.labelsize': 18,     # Liczby na osi X (było 10)
    'ytick.labelsize': 18,     # Liczby na osi Y (było 10)
    'legend.fontsize': 18,     # Legenda (było 10)
    'figure.dpi': dpi,
    'lines.linewidth': 3.5     # Grubsze linie wykresu
})

def plot_training_results(csv_file):
    if not os.path.exists(csv_file):
        print(f"Błąd: Nie znaleziono pliku {csv_file}")
        return

    df = pd.read_csv(csv_file)
    df.columns = [c.strip() for c in df.columns]
    epochs = df['epoch']

    metrics_to_plot = [
        ('train/box_loss', 'Train Box Loss', 'Loss'),
        ('train/cls_loss', 'Train Class Loss', 'Loss'),
        ('train/dfl_loss', 'Train DFL Loss', 'Loss'),
        ('metrics/precision(B)', 'Precision (B)', 'Score'),
        ('metrics/recall(B)', 'Recall (B)', 'Score'),
        ('val/box_loss', 'Val Box Loss', 'Loss'),
        ('val/cls_loss', 'Val Class Loss', 'Loss'),
        ('val/dfl_loss', 'Val DFL Loss', 'Loss'),
        ('metrics/mAP50(B)', 'mAP @ 0.5', 'Score'),
        ('metrics/mAP50-95(B)', 'mAP @ 0.5:0.95', 'Score')
    ]

    # Zwiększyłem rozmiar płótna do 30x15 cali, żeby pomieścić wielkie napisy
    fig, axes = plt.subplots(2, 5, figsize=(30, 15))
    
    # Tytuł główny - jeszcze większy
    # fig.suptitle('Training Performance Metrics (YOLOv12)', fontsize=36, fontweight='bold', y=0.98)

    axes = axes.flatten()

    for i, (col_name, title, ylabel) in enumerate(metrics_to_plot):
        ax = axes[i]
        if col_name in df.columns:
            ax.plot(epochs, df[col_name], color='#005b96', label='Value')
            
            # Tytuł z większym odstępem (pad)
            ax.set_title(title, fontweight='bold', pad=20)
            ax.set_xlabel('Epoch', labelpad=10)
            ax.set_ylabel(ylabel, labelpad=10)
            ax.grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
            
            last_val = df[col_name].iloc[-1]
            last_epoch = epochs.iloc[-1]
            
            # Pozycjonowanie etykiety końcowej
            if "loss" in col_name.lower():
                xytext_offset = (last_epoch - (last_epoch * 0.4), last_val + (last_val * 0.2))
            else:
                xytext_offset = (last_epoch - (last_epoch * 0.4), last_val - (last_val * 0.2))

            # Adnotacja z wartością - czcionka 20
            ax.annotate(f'{last_val:.3f}', 
                        xy=(last_epoch, last_val), 
                        xytext=xytext_offset,
                        arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10),
                        fontsize=20, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'Missing:\n{col_name}', ha='center', va='center', color='red')

    # Duże odstępy, żeby napisy na siebie nie wchodziły
    plt.subplots_adjust(left=0.05, right=0.98, top=0.90, bottom=0.08, wspace=0.35, hspace=0.4)

    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Wykres z DUŻYMI czcionkami zapisano jako: {output_filename}")
    # plt.show()

if __name__ == "__main__":
    plot_training_results(csv_path)