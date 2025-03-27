"""Analyze the distribution for the metric cosine similarity"""
import json
from tqdm import tqdm
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

datasets = [
    'microsoft/ms_marco',
    'rajpurkar/squad',
    'keivalya/MedQuad-MedicalQnADataset'
]

if __name__ == '__main__':
    for dataset_id in tqdm(datasets):  # Assuming 'datasets' is defined elsewhere
        read_filename = f'/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/distribution/cos_sim/{dataset_id.split("/")[1]}_cos_sim.json'
        png_filename = f'/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/datasets/questions/distribution/png/{dataset_id.split("/")[1]}_cos_sim.png'
        
        with open(read_filename, 'r') as f:
            data = json.load(f)

        # Convert to DataFrame
        df = pd.DataFrame(data)
        query_counts = df['similar queries #']

        # Statistical Analysis
        stats_dict = {
            'Count': len(query_counts),
            'Mean': query_counts.mean(),
            'Median': query_counts.median(),
            'Std Dev': query_counts.std(),
            'Min': query_counts.min(),
            'Max': query_counts.max(),
            '25th Percentile': query_counts.quantile(0.25),
            '75th Percentile': query_counts.quantile(0.75),
            'Skewness': stats.skew(query_counts),
            'Kurtosis': stats.kurtosis(query_counts)
        }

        # Print statistics
        print(f"\nStatistical Analysis of Similar Queries for {dataset_id}:")
        stats_text = f"Statistical Analysis for {dataset_id}:\n"
        for key, value in stats_dict.items():
            print(f"{key}: {value:.2f}")
            stats_text += f"{key}: {value:.2f}\n"

        # Set style for professional-looking plots
        plt.style.use('seaborn-v0_8')
        sns.set_palette("deep")
        sns.set_context("paper")

        # Create figure with adjusted layout for stats box
        fig = plt.figure(figsize=(12, 20))  # Increased height to accommodate stats
        gs = fig.add_gridspec(4, 1, height_ratios=[0.5, 1, 1, 1])  # Added space for stats

        # Add big title
        fig.suptitle(f"{dataset_id}, total: {len(data)}", 
                    fontsize=20, 
                    fontweight='bold', 
                    y=0.98)

        # Add stats text box
        ax0 = fig.add_subplot(gs[0])
        ax0.axis('off')  # Hide axes for text box
        ax0.text(0.5, 0.5, stats_text.strip(), 
                fontsize=12, 
                verticalalignment='center', 
                horizontalalignment='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        # Create subplots
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])
        ax3 = fig.add_subplot(gs[3])

        # 1. Histogram with KDE
        sns.histplot(data=query_counts, bins=30, kde=True, ax=ax1)
        ax1.set_title('Distribution of Similar Queries', fontsize=14, pad=15)
        ax1.set_xlabel('Number of Similar Queries', fontsize=12)
        ax1.set_ylabel('Amount', fontsize=12)

        # 2. Box Plot
        sns.boxplot(x=query_counts, ax=ax2)
        ax2.set_title('Box Plot of Similar Queries', fontsize=14, pad=15)
        ax2.set_xlabel('Number of Similar Queries', fontsize=12)

        # 3. Cumulative Distribution
        sns.ecdfplot(data=query_counts, ax=ax3)
        ax3.set_title('Cumulative Distribution of Similar Queries', fontsize=14, pad=15)
        ax3.set_xlabel('Number of Similar Queries', fontsize=12)
        ax3.set_ylabel('Cumulative Probability', fontsize=12)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(png_filename, dpi=300, bbox_inches='tight')
        print(f"\nDistribution plots saved as {png_filename}")

        # Additional analysis: Top 10 most frequent values
        print("\nTop 10 Most Frequent Values:")
        top_10 = query_counts.value_counts().head(10)
        for value, count in top_10.items():
            percentage = (count / len(query_counts)) * 100
            print(f"Value: {value}, Count: {count}, Percentage: {percentage:.2f}%")

        plt.close(fig)  # Close the figure to free memory