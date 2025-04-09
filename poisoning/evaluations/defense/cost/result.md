# Propmt injection 
Dataset: click
Min time: 0.0107s
Max time: 0.3783s
Avg time: 0.0115s
Total questions: 31783

Device set to use cuda:0
  0%|                                                                                                                                         | 6/10570 [00:01<27:34,  6.39it/s]You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10570/10570 [02:10<00:00, 80.97it/s]

Dataset: squad
Min time: 0.0108s
Max time: 0.3930s
Avg time: 0.0123s
Total questions: 10570
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16407/16407 [03:23<00:00, 80.66it/s]

Dataset: MedQuad-MedicalQnADataset
Min time: 0.0109s
Max time: 0.0244s
Avg time: 0.0124s
Total questions: 16407
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10047/10047 [02:10<00:00, 77.12it/s]

Dataset: ms_marco
Min time: 0.0109s
Max time: 0.0249s
Avg time: 0.0129s
Total questions: 10047


# Perplexity

Dataset: squad
Min time: 0.0127s
Max time: 0.3788s
Avg time: 0.0420s
Total questions: 10570

Dataset: MedQuad-MedicalQnADataset
Min time: 0.0242s
Max time: 0.2930s
Avg time: 0.0817s
Total questions: 16407

Dataset: ms_marco
Min time: 0.0198s
Max time: 0.2881s
Avg time: 0.0806s
Total questions: 10047