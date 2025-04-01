This is the experiment of evaluation 7.2.

structure: 
1. noise_raw/: the unfiltered noise.
2. noise/: the filtered with different numbers of noise in each json file.
3. prompts/: different adversarial promtpt template, 3 datasets in total.
4. results/: the result json file. 

to reproduce the result, run `python run.py`

to run at the backend: run `nohup python run.py &`