## RESPONSE TO REVIEWER 1

### Validation vs Insight
You state that 
*“... end task during pre-training needs less auxiliary data. This is expected and I would see this as a validation rather than an insightful observation”*
we respectfully disagree with this assertion. Given that end-task aware training is rarely used as an alternative to classical pre-training in practice, we argue that this is not simply validating already-known information. In fact, we did not strongly expect this data-efficiency result a-priori, and we are also not aware of any prior work to this effect. We argue that there is certainly a place in the literature for simple methods that are reasonable in hindsight, yet effective.

### Cross Task Generalization
*“.. how is the performance of TAPT and *-MT method on a task that is very similar to the end task”*
Thank you very much for the great suggestion. While our method is chiefly concerned with improving within-task generalization, investigating cross-task transfer would be interesting and we hope to include experiments to this end in the final version. 

### Relative Pro-Cons over TAPT
Our method has the following advantages over TAPT :
1. End-task provides signal for easy hyper-parameter tuning
2. Compute efficient due to early stopping when end-task performance plateaus
3. Superior performance, as shown in Table 1

Cons with respect to TAPT
1. We introduce extra hyper-parameters for META-MT which have to be tuned.

We will more clearly outline these differences in the final version of the paper



## RESPONSE TO REVIEWER 2

###  End-Task Size
This is a very good point, and we would like to add more results varying primary task data size to the camera-ready.  For the time being, please note that the primary tasks are ordered in terms of relative sizes ACL-ARC < SCIERC < CHEMPROT and from our results, we can see that dataset size affects the variance in each result but the trends across algorithms hold across variable primary task dataset sizes.  

###  End Task Data Creation
We’re afraid we didn’t fully understand the question about the relationship between the end-task and pre-training corpora, but to clarify: the end-task and pre-training corpora consist of separate data.


###  Missing Statistical Significance Scores
We do report statistical significance scores, please see the general response.

## RESPONSE TO REVIEWER 3


Thank you for all of the suggestions. Due to space limits, we can only respond directly to some here but we will attempt to reflect them all in the revision.

### Table 3
*“...  compared with DAPT ... EAT-MT and META-MT underperform (Table 3).”*
Please note that Table 3 features a handicapped version of EAT/META-MT where our methods are exposed to 1000x less data than DAPT and DAPT+TAPT. This reduced auxiliary data setting not only fits our computational budget but more closely relates to the amounts of data that every-day practitioners can handle.  Table 3 shows that even with this reduced data setting, the proposed methods still perform very close to, and in some cases outperform the more data-rich algorithms. 

Below is new set of results for DAPT+TAPT vs -MT in the reduced data setting (10x) to help contextualize Table 3.

|    Task          |  DAPT+TAPT    |     EAT-MT        |     META-MT     |
| :---                |       :---:              |     :----:              |          ---:            |
| ACL-ARC      |  69.12 +/- 5.76  |  71.58 +/- 1.65  |  71.05 +/- 1.65  |
| SCIERC       |  77.62 +/- 1.38  |  81.02 +/- 1.24  |  81.41 +/- 1.70  | 



*“In Table 3 … META-MT does not enhance the results compared to EAT-MT but it not discussed”*
The impact of META-MT over EAT-MT is most pronounced in Table 2 where the auxiliary data is only heterogeneous domain data (DAPT). Table 3 suggests that introducing task data as auxiliary data (TAPT) into EAT-MT obviates the need for meta-learning.

### DAPT Task Weight Trajectories
This is indeed an interesting result that we will discuss further in the revision. Note that the [TAPT + End-task] (Figure 4)  task weights have the same approximate total weight as End-Task in Figure 3. Thus the total weight given to out-of-task data (DAPT) is roughly consistent across the two figures. It seems important to assign high weight to the task data but not necessarily all of it needs to go to the actual task loss. 


### RoBERTa
All experiments use a pre-trained generalist RoBERTa model. We will update Section 4.3 to make this clearer.

### Risks of Multitasking
*“.. naive multi-task scheduling would generally hurt the performances”*
Please see the discussion at the beginning of section 3.2. Our introduction of META-MT is partially to mitigate this problem since we  can down-weight tasks that hurt our end-task

### Practicality
“... separate pre-training that is specific to each target task is not a practical solution”
Our work is more akin to fine-tuning with multitasking in terms of scale since we perform continued pre-training on RoBERTa rather than from scratch (As the DAPT-TAPT authors do). 
Please note that compared to DAPT and DAPT+TAPT variants, we are orders of magnitude more data efficient and thus we would argue that the amount of auxiliary data required to perform End-Task Aware training is reasonable for a wide range of practical NLP applications

## COMMON RESPONSE
We would like to thank all reviewers for providing valuable feedback on our paper.

### Testing For Statistical Significance
We used the permutation test [1]  ( and https://aclanthology.org/P18-1128.pdf ) to measure statistical significance. We mention this in the caption for Table 1.
For each test, we generate 10000 permutations to calculate significance level. This is sufficient to converge to a stable p-value without being a computational burden. 
We chose this over the common student t-test because 
1. We have only 10 runs per algorithm and permutation tests are more robust at low sample size
2. Permutation test is assumption free. Student t-tests assume that the samples are normally distributed.
3. Permutation test is robust to variance in the samples, so even though error-bars can overlap, we still establish significant differences in the samples. Variance in our results is expected due to small dataset sizes of end-tasks. 


### Related Work
We will expound upon Kiperwasser and Ballesteros (2018) and Meftah (2021) which Reviewer 3 mentioned. We also plan on including discussion on auxiliary data learning like https://arxiv.org/abs/2007.02693  and https://arxiv.org/pdf/2010.08244.pdf 
 
[1] Good, Phillip I. "Permutation, parametric and bootstrap tests of hypotheses: a practical guide to resampling methods for testing hypotheses." (2005).

