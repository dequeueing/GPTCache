4/2

Motivation: in our evaluation 7.2, the interaction btw different factors is very complex. We need to simplify the problems. 
- Prompt: ignore template; repeat only once; we dont try others any more. 
How to prepare the noise:
This is the key of our problem. 
There are 100 <Qtarget, Qadv> pairs in total. For each Qtarget, we need to synthesize a distribution. 
We just need a distribution, we dont need the CosSim(Qtarget, Qnoise) for each noise in the datasets. 
But is it that time-consuming? 
We can have a try. 
