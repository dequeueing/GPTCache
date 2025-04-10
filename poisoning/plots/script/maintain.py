"""The original file drawing file is at /home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/evict/results/draw.py"""
import json
import matplotlib.pyplot as plt

# Directories to save plots
eps_dir = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/plots/eps/"
png_dir = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/plots/png/"

# Dataset and cache sizes
dataset_id = "squad"
cache_sizes = [500, 1000, 2000]
user_numbers = [1, 2, 3, 4, 5, 6]

# Plot configuration
plt.style.use('ggplot')

# Main plotting logic
if __name__ == "__main__":
    plt.figure()
    
    for cache_size in cache_sizes:
        input_file = f"{dataset_id}_{cache_size}.json"
        
        with open(input_file, 'r') as file:
            data = json.load(file)
        
        # Calculate attacker frequencies
        attacker_freqs = []
        victim_total_query_number = len(data)
        
        for user_num in user_numbers:
            sending_freq = user_num * 1
            time_needed = victim_total_query_number / sending_freq
            attacker_freqs.append(60 / time_needed)
        
        # Plotting the results
        plt.plot(
            [item * 250 for item in user_numbers],
            attacker_freqs,
            label=f"{cache_size}",
            linewidth=3, marker='o', markersize=7
        )

    # Add plot labels and title
    plt.xlabel("Number of Users", fontsize=16, fontweight='bold')
    plt.ylabel("Frequency (queries/minute)", fontsize=16, fontweight='bold')

    # Set x and y ticks
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')

    # Show grid and legend
    plt.grid(True)
    plt.legend(title="Cache Size", loc='upper left', fontsize=14, frameon=True, title_fontsize=14)

    # Save plot
    eps_file = eps_dir + 'attacker_maintain_freq_vs_user_number.eps'
    png_file = png_dir + 'attacker_maintain_freq_vs_user_number.png'
    
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.savefig(eps_file, format='eps')
    plt.savefig(png_file, format='png')
    plt.close()