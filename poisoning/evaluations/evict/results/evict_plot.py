import json
import matplotlib.pyplot as plt

# Dataset and cache sizes
dataset_id = "squad"
cache_sizes = [500]  # Simplified list
user_frequencys = {
    1: {'t': [], 'freq': [], 'number': []}, 
    2: {'t': [], 'freq': [], 'number': []}, 
    4: {'t': [], 'freq': [], 'number': []}
}

# Plot configuration
plt.style.use('ggplot')

if __name__ == "__main__":
    for cache_size in cache_sizes:
        input_file = f"{dataset_id}_{cache_size}.json"
        with open(input_file, 'r') as file:
            data = json.load(file)
            
        # Reset data for each cache size
        for user_frequency in user_frequencys:
            user_frequencys[user_frequency]['t'] = []
            user_frequencys[user_frequency]['freq'] = []
            user_frequencys[user_frequency]['number'] = []
            
        # Calculate data points
        for i in range(len(data)):
            end_user_send_number = i + 1
            for user_frequency in user_frequencys: 
                t = end_user_send_number / user_frequency
                number_attacker_to_send = data[i]
                frequency_attacker_to_send = data[i] / t
                
                user_frequencys[user_frequency]['t'].append(t)
                user_frequencys[user_frequency]['number'].append(number_attacker_to_send)
                user_frequencys[user_frequency]['freq'].append(frequency_attacker_to_send)
        
        # Plot: Attacker Number vs. Time
        plt.figure()
        for user_frequency in user_frequencys:
            plt.plot(
                user_frequencys[user_frequency]['t'],
                user_frequencys[user_frequency]['number'],
                label=f"{user_frequency * 250}",
                linewidth=3, marker=None, markersize=7
            )
        plt.xlabel("Time to Flush (Seconds)", fontsize=16, fontweight='bold')
        plt.ylabel("Number of Queries", fontsize=16, fontweight='bold')
        plt.xticks(fontsize=14, fontweight='bold')
        plt.yticks(fontsize=14, fontweight='bold')
        plt.grid(True)
        plt.legend(title="Number of User", loc='upper right', fontsize=14, frameon=True, title_fontsize=14)
        plt.tight_layout()
        
        eps_dir = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/plots/eps/"
        png_dir = "/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/plots/png/"
        eps_file = f"{eps_dir}eval_cost.eps"
        png_file = f"{png_dir}eval_cost.png"

        
        plt.savefig(png_file, format='png')
        plt.savefig(eps_file, format='eps')
        plt.close()