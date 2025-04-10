import json
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Directories for saving plots
# eps_dir = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/results_perp/"
# png_dir = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/results_perp/"
eps_dir = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/plots/eps/"
png_dir = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/plots/png/"



datasets = [
        'ms_marco',
        'squad',
        'MedQuad-MedicalQnADataset',
        'all',
        ]
input_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/results_perp/"
output_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/results_perp/"

# Plot configuration
plt.style.use('ggplot')

if __name__ == '__main__':
    for dataset_id in datasets:
        # Load JSON data
        input_file = f"{input_path}{dataset_id}.json"
        with open(input_file) as f:
            data = json.load(f)

        # Extract scores and labels
        scores = [item['value'] for item in data]
        labels = [item['label'] for item in data]

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='#457B9D', linewidth=3, marker=None, markersize=7)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=3)

        # Add labels with bold and large font
        plt.xlabel('False Positive Rate', fontsize=16, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=16, fontweight='bold')

        # Set bold font for ticks
        plt.xticks(fontsize=14, fontweight='bold')
        plt.yticks(fontsize=14, fontweight='bold')

        # Legend with bold and large font
        plt.legend(loc='lower right', fontsize=14, frameon=True, title_fontsize=14)

        # Add grid
        plt.grid(True)

        # Adjust layout and save
        eps_file = f"{eps_dir}{dataset_id}_roc.eps"
        png_file = f"{png_dir}{dataset_id}_roc.png"
        plt.tight_layout()
        plt.savefig(eps_file, format='eps')
        plt.savefig(png_file, format='png')
        plt.close()

        # Calculate and print best threshold
        import numpy as np
        j_scores = tpr - fpr
        best_threshold = thresholds[np.argmax(j_scores)]
        print(f"the best threshold for {dataset_id}: {best_threshold}")