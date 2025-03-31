import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.1/black/results/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.1/black/results/'
datasets = {
    "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}
prompt_injection_patterns = ['dont_answer_PI_', 'ignore_PI_']

if __name__ == '__main__':
    all_stats = {}
    for pattern in prompt_injection_patterns:
        for dataset_id in datasets:
            input_file = input_path + "json_raw/" + f"E71_{pattern}{dataset_id}.json"
            histogram_png = output_path + 'histogram/' + f"E71_histogram_{pattern}{dataset_id}.png"
            correlation_png = output_path + "correlation_graph/" + f"E71_correlation_{pattern}{dataset_id}.png"
            stat_file = output_path + f"E71_summary.json"
            
            with open(input_file, 'r') as file:
                data = json.load(file)

            # Extract attributes
            cos_sims = [entry["cos_sim"] for entry in data]
            sem_scores = [entry["sem_score"] for entry in data]
            euc_dists = [entry["euc_dist"] for entry in data]
            injection_success = [entry["injection_success"] for entry in data if entry["injection_success"]]

            # Calculate statistics
            stats = {
                f"{pattern}{dataset_id}": {
                    "injection_success_rate": len(injection_success),
                    "cosine_similarity": {
                        "mean": float(np.mean(cos_sims)),
                        "std": float(np.std(cos_sims)),
                        "min": float(np.min(cos_sims)),
                        "max": float(np.max(cos_sims))
                    },
                    "semantic_score": {
                        "mean": float(np.mean(sem_scores)),
                        "std": float(np.std(sem_scores)),
                        "min": float(np.min(sem_scores)),
                        "max": float(np.max(sem_scores))
                    },
                    "euclidean_distance": {
                        "mean": float(np.mean(euc_dists)),
                        "std": float(np.std(euc_dists)),
                        "min": float(np.min(euc_dists)),
                        "max": float(np.max(euc_dists))
                    }
                }
            }
            
            # Add to all_stats dictionary
            all_stats.update(stats)

            # Generate visualizations
            plt.figure(figsize=(15, 9))
            plt.subplot(1, 3, 1)
            sns.histplot(cos_sims, kde=True)
            plt.title("Distribution of Cosine Similarity")
            plt.xlabel("Cosine Similarity")
            plt.ylabel("Frequency")
            
            plt.subplot(1, 3, 2)
            sns.histplot(sem_scores, kde=True)
            plt.title("Distribution of Semantic Score")
            plt.xlabel("Semantic Score")
            plt.ylabel("Frequency")
            
            plt.subplot(1, 3, 3)
            sns.histplot(euc_dists, kde=True)
            plt.title("Distribution of Euclidean Distance")
            plt.xlabel("Euclidean Distance")
            plt.ylabel("Frequency")
            
            plt.tight_layout()
            plt.savefig(histogram_png, dpi=300)

            correlation_matrix = np.corrcoef([cos_sims, sem_scores, euc_dists])
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
                       xticklabels=["Cosine Sim", "Sem Score", "Euc Dist"],
                       yticklabels=["Cosine Sim", "Sem Score", "Euc Dist"])
            plt.title("Correlation Matrix")
            plt.savefig(correlation_png, dpi=300)

    # Write all statistics to JSON file
    with open(stat_file, 'w') as f:
        json.dump(all_stats, f, indent=4)