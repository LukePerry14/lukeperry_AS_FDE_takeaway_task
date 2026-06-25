
# Codebase Manual

This repository is organized as follows:

`main.py`
- Driver script for building personas, running experiments, and evaluating strategies.

`src/`
- `model_handling.py` — LLM wrapper classes for survey response and backstory generation.
- `societies.py` — Persona and society orchestration, including demographic data handling, adjacency, and evaluation.
- `steering_prompts.json` — Demographic steering prompts for LLM persona generation.
- `w54_selected.json` — Subset of eval questions - Wave 54: covering economics and inequality.

Other files
- `README.md` — Internal codebase manual and project report.

---

# Project Report

This will be a semi-formal walkthrough of my thinking and findings. Many decisions are based on the wording of the task prompt, and as such it is listed below for reference:

---
We’d love for you to build <mark>100 LLM personas modelling one group of humans</mark>, and ask them a <mark>few single-select survey questions</mark> around a market research topic of your choosing. Your goal is to come up with <mark>different methods of creating / modelling personas using LLMs</mark>, so that they produce results that <mark>capture the human group’s opinions as much as possible</mark>, and present your findings and methods to the team.

We have left this problem intentionally vague - we’d love to see is how you formulate the problem, define objectives, and how creative you are at using LLMs to achieve your goals with first-principled intuitions in mind, and presenting your findings within an overall narrative.

---


## 0. Research

Before beginning any implementation, I did some research on LLM persona generation and existing attempts to use LLMs for survey simulations. The papers I found to be most relevant and therefore spent the most time reading were (in order of significance):

1. "Language Model Fine-Tuning on Scaled Survey Data for Predicting Distributions of Public Opinions" — *fine tuning LLMs for survey response simulation*
2. "Virtual Personas for Language Models via an Anthology of Backstories" — *best practice persona generation for accurate simulation of survey respondants*
3. "Graph-Based Alternatives to LLMs for Human Simulation" — *GNNs (not LLMs) for survey response simulation based on social network structure*

These provide a strong academic base for: the training of LLM models for survey response, persona generation theory for survey response, and social network theory in survey responses respectively. I also explored different potential datasets I could use. A satisfactory dataset would need to interface natively with out of the box LLMs for ease of development, provide persona generation priors (e.g. survey respondant demographic info), and have survey response ground truths compatible with 100 persona responses. The two primary candidates that stood out were:

- Pew ATP(as used in paper 1)
- Twin-2k-500

These datasets differ primarily in the priors they use for persona generation. Pew ATP uses a collection demographic flags to indiciate subpopulation membership (e.g. race, income, political idealogy etc.), whereas Twin-2k-500 represents each individual by an extensive pool of Q&A responses. While the Twin-2k-500 dataset is interesting as it enables complex persona generation strategies such as RAG, or RLHF, using a larger data corpus to imply persona perferences and opinions - due to time limitations, I decided to use the simpler Pew ATP dataset, which provides a much simpler basis for persona generation.

## 1. Problem Framing

My problem framing draws from the methodology in paper 1.

The Pew ATP dataset provides an "OVERALL" response distribution for each question, representing the aggregation of responses over the entire population. The dataset also includes conditional response distributions based on responses from a single demographic subset (e.g. republican, income > $100000, etc.). However, since I treat each persona as an intersection over multiple traits, there is no ground truth response for each individual persona response distribution. In order to ensure each persona is different, and that a ground truth is available for the aggregated responses over all personas, I define each persona as the unique combination of demographic attribute priors over the INCOME, AGE, and RACE categories. These were selected as they are known axes for variance in opinion as it relates to the economic inequality, and the cartesian product of these options creates exactly 100 unique triplets. The options for each demographic attribute are as follows:

```
INCOME = ["Less than $30,000", "$30,000-$50,000", "$50,000-$75,000", "$75,000-$100,000", "$100,000 or more"]
AGE = ["18-29", "30-49", "50-64", "65+"]
RACE = ["White", "Black", "Hispanic", "Asian", "Other"]
```

As each persona in a unique population tranche along these three axes, this allows the overall persona generation performance to be measured against the OVERALL response distribution. I propose that this is not only experimentally convenient, but actually acts as a reasonable parallel to a stratified market research survey, where the survey coordinator would aim to select survey constituents such that they form a a diverse - and ideally comprehensive - population sample.

The objective function therefore becomes the minimisation of wasserstein distance between generated survey responses over all 100 personas and the ground truth OVERALL distribution - I test 4 different prompt engineering based persona generation strategies, comparing performance using the wasserstein distance metric, along with a uniform response distribution baseline.

## 2. Model Selection

There are 4 key considerations for model selection:

1. **Size**: Models need to be of a reasonable size such that I can run them on my local hardware for implementation and experimentation convenience.
2. **Intelligence**: Use of an appropriately intelligent model trained on a sufficiently large dataset including opinion information will be imperative for performance maximisation. Moreover, models specialised for these tasks will also provide performance boosts.
3. **Base vs Instruct**: As noted in both papers 1 and 2, the SFT and RLHF processes that create instruct models creates significant overconfidence and biasing when generating responses to survey questions. This makes base models more suitable for synthetic surveying tasks.
4. **Task**: While base models are preferable for actual survey response, paper 2 reports that instruct models are actually superior for backstory generation.

Paper 1 has made the LoRa weights for their finetuned models publically available. I therefore elect to use the best performing small model from paper 1 loaded with the associated LoRa adapter - Mistral-7B-v0.1. It is worth pointing out that the task this model was trained for is very similar to this project brief, where the key difference is in the focus of this project on the persona generation strategy, not any fine-tuning method.

As I also use a backstory based persona generation method, following the findings of paper 2 that instruct models are better suited to demographic conditioned backstory generation, I substitute the mistral base model for Qwen3-8B model for this generation. I chose this model as it achieves state of the art performance across a variety of tasks, and is sufficiently small to run quickly on my machine.

## 3. Persona Generation

My thoughts for what to target when looking at persona generation strategies fell broadly into the following categories:

- **Persona Target**: LLM personas are traditionally thought of as a single person following some role or having some collection of traits - in a survey context this is be a survey respondent. However, a more hierarchical model could be used to model a archetype of person, or otherwise collection of people, before being more specific or personal when actually creating an instance of a person from that archetype. For example, first modelling a person by their demographic information, then generating a personal backstory conditioned on this information.

- **Persona Priors**: The traits that that you select to condition your persona generation can come from a variety of sources depending on what you assume to be the most consequential to response variance. These priors act as conditionals which change the models performance from an unconditioned base state, and are ideally are selected to produce the largest variance from baseline such that this variance modifies output response distribution to more closely match a target distribution. Examples of what these persona priors could be include: demographic attributes, schwartz's values, big five traits, or political compass scores. Desirable traits are as simple as possible, and change model responses from baseline to more closely match a target distribution. While these traits are commonly prepended to the survey question to steer the model output, the Twin-2k-500 dataset instead allows collections of responses to create a much larger unstructured cloud of datapoints which represent a person's beliefs and preferences. This format lends itself to a RAG based persona generation strategy.

- **Persona Networks**: Your social network has an undeniable effect on your beliefs and worldview. I am most interested in exploring the inclusion of peers' personas and beliefs within a simulated network as an additional conditional on potential survey responses. personas act as natural candidates for nodes on social networks, where different edge existences and weights could be used to create a simulation, where ideas are shared among communities. Personas could discuss questions before responding, or a simpler approach simply includes the information of selected peers in backstory generation or QA responses.

- **Real vs Synthetic**: Intuitively, real data from real people grounds your conditioning in a way that makes it more likely to be representative of reality. However, synthetic generation, such as in the form of creation of a full backstory might provide plausible additional context which is otherwise unavailable that conditions an LLMs response in a way that enriches responses.

- **Static vs Moulded**: Single personas generated in a static way have been demonstrated as effective in literature. However, techniques exist in LLM literature which posit that human discussion with an LLM persona often help to develop and mould their personalities to be stronger, and more rounded. This is achieved by a human engineer including guiding or adversarial LLM beliefs through prompting as part of a conversation chain.


Based on the limited time frame, I decided to focus on prompting engineering based persona generation strategies. I decided that I would include a simple Q&A style prompt which both establishes demographic attributes and guides the model towards a letter response, I would then also include a generated backstory based on demographic information, which is prepended to the survey question. For both of these generation stratgies, I also include a socialised variant which includes the information of two randomly sampled peers selected with probability propotional to the demographic similarity between the personas. The information of selected peers is then either included in the Q&A prompt, or is included in the prompt for generating the persona backstory.

Demographic similarity is calculated based on the distance in the latent space representations of persona demographic information. This distance is then passed through a logistic function to create a similarity score between 0 and 1.

$$
\text{similarity}(i, j) = \frac{1}{1 + e^{3(d_{tot} - 0.84)}}
$$

where each pairwise distance component is scaled so that the average distance along that axis equals 1:

$$
d_{\text{inc}} = |l_i^{\text{inc}} - l_j^{\text{inc}}| \times 2, \quad d_{\text{age}} = |l_i^{\text{age}} - l_j^{\text{age}}| \times 1.8, \quad d_{\text{race}} = \begin{cases} 0 & \text{if } l_i^{\text{race}} = l_j^{\text{race}} \\ 1.25 & \text{otherwise} \end{cases}
$$

$$
d_{tot} = \frac{d_{\text{inc}} + d_{\text{age}} + d_{\text{race}}}{3}
$$

Latent coordinates $l^{\text{inc}}, l^{\text{age}} \in [0, 1]$ are obtained by normalising the ordinal value associated with the demographic membership with the maximum ordinal value for each demographic option (e.g. $l^{\text{inc}} = \text{index(income band)} / (K-1)$ for $K$ income bands). $l^{\text{race}}$ is a categorical comparison calculated as hamming weight. The scaling factors (2, 1.8, 1.25) are chosen so that the expected distance between two randomly drawn individuals is approximately 1 along each axis, making $d_{tot}$ a normalised composite similarity (where smaller latent distances correspond to higher similarity). The sigmoid is centred at $d_{tot} = 0.84$, these values where arrived at empirically as they appeared to produce plausible similarities.

The four tested persona generation strategies are:
- basic demographic listing in QA format
- demographic listing in QA format with demographic listing of peers
- synthetic backstory generation using demographic information
- synthetic backstory generation using demographic information of self and peers.


## 4. Evaluation Setup

After the prompt has been passed to the Mistral model, the probability weights tokens associated with the single letter responses are extracted. These weights are then passed through a softmax function to create a probability distribution over the response options. This raw probability distribution is then sampled per persona to simulate the persona responding as a specific member of that population tranche. The overall distribution of responses over the entire persona population is then compared against the ground truth. The wasserstein distance between these distributions is then used as the measure of performance. The wasserstein distance is calculated as:

$$
W = \frac{\sum_i \left| CDF_{\text{gt}}(x_i) - CDF_{\text{agg}}(x_i) \right| \cdot \Delta x_i}{\text{ordinal}_{\max} - \text{ordinal}_{\min}}
$$

where $CDF_{\text{gt}}(x_i)$ and $CDF_{\text{agg}}(x_i)$ are the cumulative probabilities up to and including response option $i$ for the ground truth and aggregated persona distributions respectively, and $\Delta x_i$ is the gap between consecutive response values.

## 5. Results + Analysis

Results are from a full 100-persona run across 3 questions and wasserstein distance is normalised by the ordinal scale range so scores are comparable across questions. Scores are reported as the output of a single run due to time constraints. The uniform distribution over answer options serves as an upper bound — any strategy above this is worse than random.

The main takeaway is that results are mixed, no clear winners emerge, and while performance differences can be stark, it is unclear whether this is due to persona generation differences or natural performance variance over runs. The presence of only a single datapoint for each strategy and question makes in depth and cross-strategy evaluation difficult, this is an unfortunate consequence of my limited compute and time. Despite this, it appears that backstory methods win on more abstract policy-belief questions, and socialised QA wins on the harder sentiment question by blending opinions of demographics. It is worth pointing out that all four strategies comfortably beat the uniform ceiling across the board.

### INEQ5_i_W54
 
**How much, if at all, do you think some people starting out with more opportunities than others contributes to economic inequality in this country?**
 
| Option | ground_truth | uniform_upper_bound | baseline | socialised | backstory | backstory_socialised |
|---|---|---|---|---|---|---|
| Contributes a great deal | 0.40 | 0.25 | 0.39 | 0.44 | 0.42 | 0.42 |
| Contributes a fair amount | 0.38 | 0.25 | 0.37 | 0.39 | 0.28 | 0.41 |
| Contributes not too much | 0.17 | 0.25 | 0.15 | 0.13 | 0.26 | 0.13 |
| Contributes not at all | 0.05 | 0.25 | 0.09 | 0.04 | 0.04 | 0.04 |
 
| Method | WD |
|---|---|
| uniform_upper_bound | 0.2080 |
| baseline | 0.0213 |
| socialised | 0.0354 |
| backstory | 0.0366 |
| backstory_socialised | 0.0287 |
 
All models perform well under this paradigm, with distribution shapes largely matching the ground truth under all strategies. Interestingly, backstory_socialised and the baseline were the top performers here, showing no clear correlation between strategies and performance, implying all tasks perform well enough to capture ground truth up to sampling noise and these differences between strategies might disappear over multiple runs. The only exception here is overconfidence on the backstory strategy for the 'contributes not too much option', where synthetic background context may be overpowering demographic information. 

### INEQ8_c_W54
 
**How much, if at all, do you think the following proposals would do to reduce economic inequality in the U.S.? Increasing taxes on the wealthiest Americans**
 
| Option | ground_truth | uniform_upper_bound | baseline | socialised | backstory | backstory_socialised |
|---|---|---|---|---|---|---|
| A great deal | 0.46 | 0.25 | 0.56 | 0.56 | 0.47 | 0.53 |
| A fair amount | 0.24 | 0.25 | 0.30 | 0.36 | 0.42 | 0.32 |
| Not too much | 0.18 | 0.25 | 0.12 | 0.05 | 0.08 | 0.14 |
| Nothing at all | 0.11 | 0.25 | 0.02 | 0.03 | 0.03 | 0.01 |
 
| Method | WD |
|---|---|
| uniform_upper_bound | 0.1878 |
| baseline | 0.1122 |
| socialised | 0.1288 |
| backstory | 0.0888 |
| backstory_socialised | 0.1022 |
 
Backstory methods dominate QA style demographic listing, implying that inclusion of motivating context for americans' lifestories might serve as better conditionals on beliefs for policy. For example working class americans often oppose wealth taxes due to implied violation of the principle of hard-work, where a reasonable intuitive belief might be that working class americans would be more critical of wealth inequality. This type of value-based nuance is something demgraphic attributes alone do not capture.

### ECON5_d_W54
 
**Do you think the country's current economic conditions are helping or hurting people who are poor?**
 
| Option | ground_truth | uniform_upper_bound | baseline | socialised | backstory | backstory_socialised |
|---|---|---|---|---|---|---|
| Helping a lot | 0.11 | 0.20 | 0.01 | 0.04 | 0.06 | 0.04 |
| Helping a little | 0.16 | 0.20 | 0.15 | 0.15 | 0.14 | 0.10 |
| Hurting a little | 0.14 | 0.20 | 0.39 | 0.34 | 0.43 | 0.45 |
| Hurting a lot | 0.51 | 0.20 | 0.30 | 0.39 | 0.28 | 0.33 |
| Neither helping nor hurting | 0.08 | 0.20 | 0.15 | 0.08 | 0.09 | 0.08 |
 
| Method | WD |
|---|---|
| uniform_upper_bound | 0.1965 |
| baseline | 0.1262 |
| socialised | 0.0878 |
| backstory | 0.1128 |
| backstory_socialised | 0.1245 |


The lower scores on this question across all strategies implies this question is the hardest. However, despite this, the socialised QA style prompt shows a 28% improvement over second (backstory), with lots of this coming from a strong match in the dominant "hurting a lot" response option. This might be due to a movement of mass for personas representing more privileged population tranches downwards, where the model might otherwise predict they are upset with excessive welfare spending.



## 6. Next Steps

The Obvious next step is running the existing code over 10 runs and reporting the mean and standard deviation of results. This would significantly reduce guesswork in the results analysis.

Additionally, more comprehensive results analysis would allow a better understanding of the behaviour of persona generation and output conditioning. For example, decomposing results by persona would allow validation that persona conditioning is producing distinct output variance. It is worth noting that the Pew ATP dataset doesn't include intersectional ground truth over multiple demographics, meaning that this analysis would be more qualitative. Reporting per persona outputs could be done as a jacobian of output distributions differences relative to the averaged response distributions. This would reveal output contributions more clearly and identify whether certain demographic cells are systematically harder to model and should be targetted for model debiasing etc.

Continuing on the persona generation side, results seem to indicate that socialised persona generation strategies do produce notable differences under my proposed construction, therefore an extension to this mechanism to allow for inter-persona dialogue (having personas discuss the question before answering) as a preprocessing step might be promising. Alternatively, a more sophisticated social network model rather than a weighted complete graph could make social interaction dynamics more representative. My understanding is that social network simulation graphs require specialised properties such as clustering, and small-world. This would be a relatively easy next step.

In terms of model selection, the finetuned model I selected used was designed to act as a single predictor over entire (sub)-populations conditioned on a single demographic, not making it a native candidate for multiple-attribute personas and modelling of individuals for persona discussion or socialising. Finetuning of a specialised model might therefore be promising, moreover, use of unsupervised diversity losses might allow per-persona variance while capturing the overall response distribution.

Finally, experimentation with more advanced non-prompt-engineering based strategies would be interesting to pursue, e.g. a RAG-based system using collections of datapoints per persona, allowing for large data corpuses to be used. Per-persona LoRa adapters might also produce stronger persona conditioned output variance, this would be particularly useful for underrepresented sub-populations.

# Appendix

Example prompts for each persona generation strategy, all for the same persona (White, 18-29, Less than $30,000) asked INEQ5.

---

### Baseline

```
Question: Last year, what was your total family income from all sources, before taxes?
A. Less than $30,000
B. $30,000-$50,000
C. $50,000-$75,000
D. $75,000-$100,000
E. $100,000 or more
Answer: A. Less than $30,000

Question: Which race or ethnicity do you identify with?
A. White
B. Black
C. Asian
D. Hispanic
E. Other
Answer: A. White

Question: How old are you?
A. 18-29
B. 30-49
C. 50-64
D. 65+
Answer: A. 18-29

Question: How much, if at all, do you think some people starting out with more opportunities than others contributes to economic inequality in this country?
A. Contributes a great deal
B. Contributes a fair amount
C. Contributes not too much
D. Contributes not at all
Answer:
```

---

### Socialised

The persona’s own QA block is identical to baseline. Two demographically similar peers (sampled via the adjacency matrix) are appended before the survey question.

```
[... self QA blocks as above ...]

Following are the survey responses of the two people closest to you in your life:

Close Person 1:

 Question: Last year, what was your total family income from all sources, before taxes?
A. Less than $30,000
B. $30,000-$50,000
C. $50,000-$75,000
D. $75,000-$100,000
E. $100,000 or more
Answer: B. $30,000-$50,000

Question: Which race or ethnicity do you identify with?
A. White
B. Black
C. Asian
D. Hispanic
E. Other
Answer: A. White

Question: How old are you?
A. 18-29
B. 30-49
C. 50-64
D. 65+
Answer: A. 18-29

Close Person 2:

 Question: Last year, what was your total family income from all sources, before taxes?
A. Less than $30,000
B. $30,000-$50,000
C. $50,000-$75,000
D. $75,000-$100,000
E. $100,000 or more
Answer: E. $100,000 or more

Question: Which race or ethnicity do you identify with?
A. White
B. Black
C. Asian
D. Hispanic
E. Other
Answer: A. White

Question: How old are you?
A. 18-29
B. 30-49
C. 50-64
D. 65+
Answer: A. 18-29

[End of closest people survey responses]

Question: How much, if at all, do you think some people starting out with more opportunities than others contributes to economic inequality in this country?
A. Contributes a great deal
B. Contributes a fair amount
C. Contributes not too much
D. Contributes not at all
Answer:
```

---

### Backstory

The QA blocks are replaced entirely by a Qwen3-8B generated narrative. The survey question follows directly.

```
I grew up in a working-class neighborhood in Chicago, where my parents both worked multiple jobs to make ends meet. Despite the challenges, I excelled in school and earned a full scholarship to a prestigious university, where I now study political science. My family’s sacrifices and my own drive have shaped my commitment to social justice and community empowerment. I’m proud of my heritage and use my voice to advocate for equity in education and economic opportunity.

Question: How much, if at all, do you think some people starting out with more opportunities than others contributes to economic inequality in this country?
A. Contributes a great deal
B. Contributes a fair amount
C. Contributes not too much
D. Contributes not at all
Answer:
```

---

### Socialised Backstory

The backstory is generated with both the persona’s own demographics and their two sampled peers passed to Qwen3-8B, so social context is baked into the narrative rather than appended structurally.

```
I grew up in a working-class Black family in the Midwest, but my career in finance and my marriage to a white partner in my late 20s shifted my social circle. Now, I often find myself in conversations with older, affluent professionals — mostly white —
about retirement, politics, and the future of the country. While we share similar incomes, our experiences and perspectives often diverge, especially when it comes to race and systemic inequality. I try to bridge the gap, but it’s hard to ignore the unspoken divides that still exist.

Question: How much, if at all, do you think some people starting out with more opportunities than others contributes to economic inequality in this country?
A. Contributes a great deal
B. Contributes a fair amount
C. Contributes not too much
D. Contributes not at all
Answer:
```