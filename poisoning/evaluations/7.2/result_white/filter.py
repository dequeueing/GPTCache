import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_summary_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def parse_dataset_name(dataset_name):
    try:
        parts = dataset_name.split('_')
        if len(parts) < 5:
            raise ValueError("Dataset name format incomplete")
            
        pattern = parts[1]
        dataset = parts[2]
        config = parts[3]
        value_str = parts[4]
        
        # Handle value conversion based on config type
        if 'noise' in config.lower():
            value = int(value_str)
        else:
            value = float(value_str)
            
        return {
            'pattern': pattern,
            'dataset': dataset,
            'config': config,
            'value': value
        }
    except Exception as e:
        print(f"Error parsing {dataset_name}: {str(e)}")
        return {'pattern': 'unknown', 'dataset': 'unknown', 'config': 'unknown', 'value': 0}

def analyze_summary(file_path):
    # Load and prepare data
    df = load_summary_data(file_path)
    
    # Parse dataset names
    print(df['Dataset'])
    exit(0)
    parsed_info = df['Dataset'].apply(parse_dataset_name)
    df[['pattern', 'dataset', 'config', 'value']] = pd.DataFrame(parsed_info.tolist(), index=df.index)
    
    # Basic statistics
    print("Basic Statistics:")
    print(df[['ASR', 'attack success', 'injection success', 'similar success', 'total']].describe())
    
    # Analysis by configuration type
    config_groups = df.groupby('config')
    
    # Create plots directory
    Path('analysis_plots').mkdir(exist_ok=True)
    
    # 1. ASR Analysis by Config and Value
    plt.figure(figsize=(12, 6))
    for config in config_groups.groups.keys():
        config_df = config_groups.get_group(config)
        plt.plot(config_df['value'], config_df['ASR'], marker='o', label=config)
    plt.xlabel('Configuration Value')
    plt.ylabel('Attack Success Rate (ASR)')
    plt.title('ASR by Configuration Type and Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('analysis_plots/asr_by_config.png')
    plt.close()
    
    # 2. Success Rates Comparison
    metrics = ['ASR', 'injection success', 'similar success']
    melted_df = df.melt(id_vars=['Dataset', 'config', 'value'], 
                       value_vars=metrics,
                       var_name='Metric',
                       value_name='Rate')
    melted_df.loc[melted_df['Metric'] != 'ASR', 'Rate'] /= df['total']
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='config', y='Rate', hue='Metric', data=melted_df)
    plt.title('Distribution of Success Rates by Configuration')
    plt.savefig('analysis_plots/success_rates_distribution.png')
    plt.close()
    
    # 3. Pattern Analysis
    pattern_group = df.groupby('pattern')
    pattern_stats = pattern_group[['ASR', 'injection success', 'similar success']].mean()
    pattern_stats[['injection success', 'similar success']] /= df['total'].mean()
    print("\nAverage Metrics by Pattern:")
    print(pattern_stats)
    
    # 4. Dataset Analysis
    dataset_group = df.groupby('dataset')
    dataset_stats = dataset_group[['ASR', 'injection success', 'similar success']].mean()
    dataset_stats[['injection success', 'similar success']] /= df['total'].mean()
    print("\nAverage Metrics by Dataset:")
    print(dataset_stats)
    
    # 5. Correlation Analysis
    correlation = df[['ASR', 'attack success', 'injection success', 'similar success']].corr()
    print("\nCorrelation Matrix:")
    print(correlation)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap of Success Metrics')
    plt.savefig('analysis_plots/correlation_heatmap.png')
    plt.close()
    
    # 6. Detailed statistics by config
    for config, group in config_groups:
        print(f"\nDetailed Statistics for {config}:")
        print(group.groupby('value')[['ASR', 'injection success', 'similar success']].mean())


datasets = {
    "squad": "squad_targeted.json",
    "MedQuad-MedicalQnADataset": "MedQuad-MedicalQnADataset_targeted.json",
    "ms_marco": "ms_marco_targeted.json"
}
prompt_injection_patterns = [
    'dont_answer_PI_', 
    'ignore_PI_'
]

configs = {
    'thresholds': [0.2, 0.4, 0.6, 0.8, 0.9],  
    'top_k': [1, 3, 5, 10],
    'noise_number': [0, 500, 1000, 2000],
}





if __name__ == "__main__":
    file_path = '/home/taojie_wang@idm.teecertlabs.com/GPTCache/poisoning/evaluations/7.2/result_white/E72_white_summary.json'
    data = load_summary_data(file_path)
    for item in data:
        if 'correlation' in item['Dataset']:
            print(item)
        
        
# {
#     "Dataset": "E72_dont_answer_PI_squad_thresholds0",
#     "ASR": 0.0,
#     "total": 100,
#     "attack success": 0,
#     "injection success": 0,
#     "similar success": 24
# }