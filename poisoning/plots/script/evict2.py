"""The original code is at: /home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evict/results/evict_plot2.py"""
import json
import matplotlib.pyplot as plt

# Dataset and cache sizes
dataset_id = "squad"
cache_sizes = [500, 1000, 2000]
user_numbers = [2]  # Not used in the plot, but kept for context

# Plot configuration
plt.style.use('ggplot')

if __name__ == "__main__":
    plt.figure()
    for cache_size in cache_sizes:
        input_file = f"{dataset_id}_{cache_size}.json"
        with open(input_file, 'r') as file:
            data = json.load(file)
            
        victim_total_query_number = len(data)
        times = []
        for index, item in enumerate(data):
            sent = index + 1
            time = sent / 2  # 2 queries/second
            times.append(time)
            
        plt.plot(
            times,
            data,
            label=f"{cache_size}",
            linewidth=3, marker=None
        )
        
    eps_dir = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/plots/eps/"
    png_dir = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/plots/png/"


    plt.xlabel("Time to Flush (Seconds)", fontsize=16, fontweight='bold')
    plt.ylabel("Number of Queries", fontsize=16, fontweight='bold')
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.grid(True)
    plt.legend(title="Cache Size", loc='upper right', fontsize=14, frameon=True, title_fontsize=14)
    plt.tight_layout()
    
    eps_file = f"{eps_dir}evict_eval2.eps"
    png_file = f"{png_dir}evict_eval2.png"
    plt.savefig(png_file, format='png')
    plt.savefig(eps_file, format='eps')
    plt.close()