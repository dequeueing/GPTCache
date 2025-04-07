import json
import matplotlib.pyplot as plt


dataset_id = "squad"
cache_sizes = [
    500
    # 500, 1000, 2000
]
user_frequencys = {
    1: {'t':[], 'freq': [], 'number': []}, 
    2: {'t':[], 'freq': [], 'number': []}, 
    4: {'t':[], 'freq': [], 'number': []}
}

if __name__ == "__main__":
    for cache_size in cache_sizes:
        input_file = f"{dataset_id}_{cache_size}.json"
        with open(input_file, 'r') as file:
            data = json.load(file)
            
        t_axis = []
        attacker_number = []
        attacker_frequency = []
        for i in range(len(data)):
            end_user_send_number = i + 1
            
            # the independent var is user frequency
            for user_frequency in user_frequencys: 
                t = end_user_send_number / user_frequency
                number_attacker_to_send = data[i]
                frequency_attacker_to_send = data[i] / t
                
                t_axis = user_frequencys[user_frequency]['t']
                attacker_number = user_frequencys[user_frequency]['number']
                attacker_frequency = user_frequencys[user_frequency]['freq']
                
                t_axis.append(t)
                attacker_number.append(number_attacker_to_send)
                attacker_frequency.append(frequency_attacker_to_send)
        
        # Plot 1: Attacker Number vs. Time
        plt.figure()
        for user_frequency in user_frequencys:
            plt.plot(
                user_frequencys[user_frequency]['t'],
                user_frequencys[user_frequency]['number'],
                label=f"user_freq={user_frequency}"
            )
        plt.xlabel("Time (second)")
        plt.ylabel("Attacker Number")
        plt.title(f"Attacker Number vs. Time (Cache Size: {cache_size})")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"attacker_number_{cache_size}.png")
        plt.close()

