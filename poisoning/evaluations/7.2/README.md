This is the experiment of evaluation 7.2.

structure: 
1. noise_raw/: the unfiltered noise.
2. noise/: the filtered with different numbers of noise in each json file.
3. prompts/: different adversarial promtpt template, 3 datasets in total.
4. results/: the result json file. 

to reproduce the result, run `python run.py`
to run at the backend: run `nohup python run.py &`


python files:
1. run.py: run the 7.2 evaluation to reproduce the results.
2. filter.py: analyze the result of experiment, into table.
2. fix.py: fix the result bugs because of the check bug.

The output entry:
```json
{
    "Dataset": "E72_dont_answer_PI_squad_thresholds0",
    "ASR": 0.0,
    "total": 100,
    "attack success": 0,
    "injection success": 0,
    "similar success": 24
}
```

Explanation: 
1. ASR
2. injection success: whether the injection into cache is successful or not?
3. similar enough: whether the target question is similar enough compared with the adv question. 