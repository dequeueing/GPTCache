import json
import matplotlib.pyplot as plt
import numpy as np

datasets = [
    # "squad",
    # "MedQuad-MedicalQnADataset",
    "ms_marco",
    # "text-image",
    # 'trivia'
]

plt.style.use('ggplot')

for id in datasets:
    input_path = f'{id}.json'
    with open(input_path, 'r') as file:
        data = json.load(file)

    print(f"dataset {id} has {len(data)} sampled datapoints")

    data = np.array(data)

    bin_width = 0.01
    bins = np.arange(0, 1 + bin_width, bin_width)

    counts, _ = np.histogram(data, bins=bins)
    relative_freq = counts / counts.sum()

    plt.figure(figsize=(6, 5))
    plt.bar(
        bins[:-1], relative_freq, width=bin_width,
        align='edge', color='#457B9D'
    )

    # Bold and larger fonts
    plt.xlabel('Cosine Similarity', fontsize=14, fontweight='bold')
    plt.ylabel('Proportion', fontsize=14, fontweight='bold')
    # plt.title(f'Distribution for {id}', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    plt.grid(True, linestyle='--', linewidth=1, alpha=1)
    plt.tight_layout()
    plt.show()

    # Save as EPS
    output_path = f'{id}_distribution.eps'
    plt.savefig(output_path, format='eps')
    output_path = f'{id}_distribution.png'
    plt.savefig(output_path, format='png')
    plt.close()

    print(f"Saved EPS file to: {output_path}")
