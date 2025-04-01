import json
import random

input_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.2/noise_raw/'
output_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.2/noise/'
datasets = {
    "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}
noise_number = [0, 500, 1000, 2000]


if __name__ == '__main__':
    for dataset_id in datasets:
        # Construct the input file path
        input_file = f"{input_path}{dataset_id}_noise.json"
        
        try:
            # Read the input JSON file
            with open(input_file, 'r') as f:
                data = json.load(f)
                total_entries = len(data)
                print(f"Loaded {input_file} with {total_entries} entries")
                
            # For each noise level
            for number in noise_number:
                output_file = f"{output_path}{dataset_id}_{number}noise.json"
                
                # If number is 0, create empty list
                if number == 0:
                    output_data = []
                else:
                    # Randomly select min(number, total_entries) items
                    sample_size = min(number, total_entries)
                    output_data = random.sample(data, sample_size)
                
                # Write the output JSON file
                with open(output_file, 'w') as f:
                    json.dump(output_data, f, indent=4)
                
                print(f"Created {output_file} with {len(output_data)} entries")
                
        except FileNotFoundError:
            print(f"Input file not found: {input_file}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from file: {input_file}")
        except Exception as e:
            print(f"An error occurred processing {dataset_id}: {str(e)}")