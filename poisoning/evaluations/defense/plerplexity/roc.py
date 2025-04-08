import json
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


datasets = [
    # 'click',
    # 'squad',
    # 'MedQuad-MedicalQnADataset',
    # 'ms_marco',
    'all'
]
input_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/results_perp/"
output_path = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/defense/results_perp/"

if __name__ == '__main__':
    for dataset_id in datasets:
        # Load your JSON data
        input_file = f"{input_path}{dataset_id}.json"
        output_png = f"{output_path}{dataset_id}.png"
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
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {dataset_id}')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_png)
        plt.show()
        
        import numpy as np

        j_scores = tpr - fpr
        best_threshold = thresholds[np.argmax(j_scores)]
        print(f"the best threshold for {dataset_id}: {best_threshold}")



