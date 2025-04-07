import json
import matplotlib.pyplot as plt


dataset_id = "squad"
cache_sizes = [500, 1000, 2000]
user_numbers = [1, 2, 3, 4, 5, 6]


if __name__ == "__main__":
    plt.figure()
    for cache_size in cache_sizes:
        input_file = f"{dataset_id}_{cache_size}.json"
        with open(input_file, 'r') as file:
            data = json.load(file)
            
        attacker_freqs = []
        victim_total_query_number = len(data)
        for user_num in user_numbers:
            sending_freq = user_num * 1
            time_needed = victim_total_query_number / sending_freq
            attacker_freqs.append(1 / time_needed)
                
            
        plt.plot(
            [item * 250 for item in user_numbers],
            attacker_freqs,
            label=f"cache size={cache_size}"
        )

                    
    plt.xlabel("User number")
    plt.ylabel("Attacker Maintain Frequency (1 / second)")
    plt.title(f"Attacker Maintain Frequency vs. User number")
    plt.grid(True)
    plt.legend()
    plt.savefig(f"Maintain.png")
    plt.close()

