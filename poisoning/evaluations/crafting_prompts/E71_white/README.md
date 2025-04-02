The purpose for this E71 white directory is to find a systematic way to improve the adversarial similarity compared with the target one. 

The key idea is to maintain the cosine similarity, while try to enhance the semantic score. 


structure:
1. E71_results/: the results from the previous experiment. 


python sripts:
1. run.py: answer the question what is the upper limit of the cosine similarity; use different length of embedding suffix to do experiments.
2. run_penalty.py: answer the second question: what is the method that allows us to incur least damage to cosine similarity while achieve a very high semantic score.
3. test.py: a flexible platform for us to do all test, like calculating the cosine similarity, semantic score etc. 


to run the first script, use `nohup python3 run.py &`