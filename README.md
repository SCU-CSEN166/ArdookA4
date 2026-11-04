# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** A4  
**Challenge:** Ardook 
**Team members:** Shreya Vishwanath, Thien Giap Do, Timothy Le, Yash Sanghvi  
**Repository:** https://github.com/SCU-CSEN166/ArdookGrantProject 
**Last updated:** 10/29/26

> **How to use this README:** Maintain this single file throughout the project. Complete the Week 5 project plan first, then update the appropriate weekly section before each milestone. Be specific and honest: describe what you attempted, what worked, what did not work, and what evidence supports your conclusions. For every week, document each member's actual contribution. If a task was not completed, explain the blocker and next step. Do not include private or sensitive data in this public README.

## Remember - Best Practices

### Data access and privacy

[Explain where authorized data can be accessed. Do not commit personal information, private WhatsApp exports, API keys, credentials, or restricted grant/application records. Use synthetic or anonymized examples in the README.]

### Known limitations

[Keep a list of what the system cannot reliably do.]

## Week 5 — Scope: Project Plan

**Deliverable during Week 6 Lab:** A project plan (write in this README file) describing your problem formulation, proposed approach, each team member's responsibilities, proposed baseline models with Hugging Face links, proposed evaluation metrics, proposed data with links, and goals for each week.

### 1. Problem formulation and social impact

- **Problem:** What specific task are you solving? Ardook provides mothers, who are unable to leave their childrens' sides, with job opportunities and therefore, stable sources of income. To continue their work, Ardook requires adequate funding, much of which can be obtained through grants for social impact organizations and non-profits. However, the process of identifying relevant grant options and determining eligibility for such grants can be time-consuming and tedious for the non-profit employees. Our AI agent will streamline this process by identifying applicable grants and making suggestions to the non-profit regarding which grants are a best-fit and which should be applied to.
- **Intended users:** Who would use the system, and for what decisions or activities? The primary users of this system will be the employees of Ardook who are in charge of obtaining grant funding for the organization. They will be able to search for grants using the software and/or recieve suggestions for grants that Ardook is eligible for and could apply to. If the project expands outside of Ardook, other non-profit organizations that are also interested in expediting their grant search process could also utilize this AI system.
- **Inputs and outputs:** What will the system receive and produce? Give one concrete example. The system would receive information about a grant and would output whether Ardook is eligible for the grant.
- **Success criteria:** What would make the system useful and reliable? The system could be considered useful if it provides reliable reccomendations for grants with minimal to zero hallucinated reccomendations. It would also be considered successful if Ardook is able to apply to and receive grants from a multitude of sources because they match the criteria of what the grant is looking for.
- **Constraints and risks:** Some risks include maintaining the privacy of Ardook as the AI system gathers information about them and uses it to search for grants. Another risk is hallucinations as we test the model and try to ensure that it produces a competent responses.

### 2. Proposed technical approach

Proposed Technical Approach

Solutions
Initially, as we were brainstorming how to help Ardook target grants efficiently and effectively, our first intuition was the simple one. We can compare Ardook’s vision and purpose from their website and statements to various grants out there. The idea is to retrieve grants that share similar visions and purposes. Obviously, we couldn’t do exact word matching because that would simply be impossible and tedious. So our first and initial approach is to find similarities between Ardook and grants. For this, we would need to use hugging face models that are suited and equipped to find similarities between texts and documents given to it.
Building off of the initial approach that we have, we also want to incorporate TF-IDF to retrieve grants that stand out among many grants especially to Ardook. Another consideration for this is to remove stop words that are common in the English language but have no real significance to our grant search for Ardook. 
Something we can do to help our search is to build a bag of words that are specific to Ardook and what they are trying to accomplish. As we train our models on different data, we can add or subtract to that bag of words depending on the performance.
In terms of using the dataset given to us from Ardook, we will need to clean that data and categorize it in a way that is helpful and useful for our models to interpret and train against.
Models
sentence-transformers/all-MiniLM-L6-v2
This model from Hugging Face turns sentences and paragraphs into a compact vector space. Which it will do for synthetic data first and compare it to the vector space of our Ardook’s data and information.
The way this model actually helps us is that it can measure similarities between documents and texts by using cosine similarity which is actually something I just learned in my data mining class. 
A cool feature for this model as well is that it can catch similar words as we were hoping for in our solution because exact matching would not be feasible or effective at all.
This is a great starter model for us because it is lightweight and easy to iterate and test with before moving on to bigger and complex models. But it will also give us a good sense of our data and how we can improve our training dataset.
BAAI/bge-small-en-v1.5
This model is also a good option for our approach to solve our problem. It is different from the MiniLM because it is designed specifically for information retrieval.
Instead of just passing all of Ardook’s information into a vector space, I would create an instruction + a query for Ardook. So it requires more details and thought behind our query to best maximize our retrieval. Then we can run it against our synthetical data then eventual real data.


cross-encoder/ms-marco-MiniLM-L-6-v2
The first two proposed models are both bi-encoders which means they produce separate embeddings for Ardook and grants. With this model, we can go beyond that with the ability to combine their data simultaneously. This is helpful because our earlier bi-encoder models can miss negative combinations or correlation because it was embedded separately but this will combine the two which in turn will reveal more to us. 
 
Considerations
Our first two bi-encoders models both have a maximum limit on length of tokens so we need to ensure that our data can fit or needs to be chunked to fit the requirements. 
A common problem can be an imbalanced dataset which is a common problem in the real world. This is definitely something we need to keep an eye for in our training and testing on our models.
We will want to use certain python libraries to incorporate TF-IDF as we evaluate these different grants to truly see how similar a grant is to Ardook among other grants. These libraries will also help us with accounting for stop words that we have mentioned above


### 3. Proposed data

| Dataset/source                                                                                                                           | Link or access instructions                                                                                                                                                                                            | Purpose                                   | Size / languages                 | Privacy or usage considerations                                                                                                                  |
| ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| [Name]                                                                                                                                   | [URL](https://huggingface.co/datasets) or other instructions]                                                                                                                                                          | [Training/development/evaluation]         | [Details]                        | [Notes]                                                                                                                                          |
| Provided skeleton spreadsheet of Ardook Data Grant Recommendations and manually added data found through NSF,NIH,Humanity AI and Google. | [https://docs.google.com/spreadsheets/d/1P0SwjN2LDN-CFEmBdHkXG3UDChT_4o2h/edit?gid=865612512#gid=865612512](https://docs.google.com/spreadsheets/d/1P0SwjN2LDN-CFEmBdHkXG3UDChT_4o2h/edit?gid=865612512#gid=865612512) | Used for training, testing and validation | Size:mutable as grants get found | Additional Notes: NSF and NIH grants tend to either have requirements of organizations being in the U.S or that the organization is STEM focused |

- **Data preparation:** [Cleaning, anonymization, annotation, splitting, deduplication, etc.]
- | The given data may need to be cleaned of grants that are no longer open. If they are already closed there is no point in looking at them for evaluation. When it comes to implementation the data will have to be split into train, test and validation. Currently proposing a 80,10,10 split for train test and validation respectively.|
- **Data limitations:** [What is missing, noisy, or uncertain?]
- | There was not a lot of starting data from the spreadsheet so it will be a bit tedious to go manually find more grants. Also unsurprisingly there are not many datasets out there that contain what we are trying to assess: women led business grant validation.|

### 4. Proposed evaluation

| Metric / test                    | What it measures                                                                                   | How it will be calculated                                                                                             | Target / success criterion                                                                                  |
| -------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| [Metric - e.g Accuracy, F1, AoC] | [Quality dimension]                                                                                | [Method/ Formula]                                                                                                     | [Target]                                                                                                    |
| F1                               | How well relevant, comparable information was extracted from the provided grant info               | compare the model's capability to what we're able to derive about a grant by reading it ourselves, F1 equation        | the model is able to identify the same relevant information that we are, calculated F1 value is high enough |
| recall                           | whether the model correctly retrieves and outputs relevant grant, whether it drops relevant grants | compare the model's capability to what we're able to derive about a grant by reading it ourselves, recall equation    | the model is able to identify the same relevant grants that we are, calculated recall value is high enough  |
| precision                        | whether the model suggests grants that Ardook would actually use                                   | compare the model's capability to what we're able to derive about a grant by reading it ourselves, precision equation | the model's evaluation of top grants is the same as ours, calculated precision value is high enough         |

- **Evaluation examples:** [How will you build or select a development set?] We have a set of grants that have been evaluated for eligibility by employees of Ardook, so we can use this as a development set and ensure that the model is first able to derive the same output and reach the same conclusions on eligibility that the non-profit did.
- **Human verification:** [If relevant, who will check outputs and how?]
- | Two group members will check if the outputs actually match the appropriate classification and if we agree on the classification then its an accurate classification|
- **Failure cases to watch:** [At least two anticipated risks]
- |1. There is some data that is labeled as in between such as weak-moderate fit. I think that there may come some issues with properly trying to classify unless we consider weak-moderate for example as its own category. Also some of the data is conditional such as “Excellent fit if…” which means that our classifier may run into issues for finding the proper conditions to meet an excellent fit. 2. I think another issue that may arise is how we have our model distinguish how much to emphasize the key concern/disqualifier factor. We would need to determine what is a major disqualification versus a minor one|

### 5. Team responsibilities

| Member  | Primary responsibility                                                | Specific Weekly contribution                                                                        | Planned next task                                                                                                                                                                                                                                                                                                                                                                               |
| ------- | --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Giap Do | Collecting feasible models from HuggingFace and brainstorm approaches | Found 4 potential models the team can test with against the sample data that Timothy has collected. | Evaluation of models. Will need to measure metrics of these models against the data that we have collected. I can use techniques that we have learned in class such as accuracy, precision, recall, and F-1 score to evaluate the performance of these models. However, I will also need to evaluate the performance against the data that we have found. Data could potentially be imbalanced. |

| Timothy | Data Collection and Model Investigation | Added multiple entries to the spreadsheet for data collection. Beginning investigation on what models to use for implementation. Updated README with parts 3 and 5 and some of part 4. Went to go ask clarifying questions for the project in office hours. Scheduled outside of lab period meeting. | Next Task:Collect more data of grants. Find appropriate models for classifying what grants are worth applying to. |
| Shreya | Data Collection and Model Investigation | Beginning investigation on what models to use for implementation. Updated README with part 1, 2. Attended outside of lab period meeting where we discussed data collection and model possibilities. | Next Task: Collect more data of grants. Begin implementing models/filling in gaps in pipeline. |

| Yash Sanghvi | Synthetic data generation/evaluations| Beggining investigations on how to create trainable synthetic data with high fidelity and diversity. Looking at what data can actually help grant discovery and how to evaluate the data with the model. Working on system design to make the system run optimaly from end to end. |

---

## Week 6 — Build: Working Baseline

**Deliverable during Week 7 Lab:** A runnable baseline, setup/run instructions, and 1–2 example inputs and outputs. Update this section with what worked, what failed, and individual contributions.

### Implementation status

- **What runs end-to-end now?** [Describe exactly.]
- **How to run it:** [Commands, dependencies, configuration; refer to Setup below.]
- **Model(s) and data actually used:** [Names, versions, links, and any changes from the plan.]
- **Example 1:** Input: [non-sensitive example] → Output: [result]
- **Example 2:** Input: [non-sensitive example] → Output: [result]

### Fine Tuning Techniques
- Utalize Lora back propogation techniques as discussed in class
    - Forward propogation - Take x input and compare it to y input: y predicted = weight times the input
    - Loss Function - use sometype of distance formula to calculate loss(square-loss/manhattan): 1/2(y^ - y)^2
    - Back propogation - Using chain rules find which weight contributed to error: derivate of L / derivative of W
    - Gradient Descent - Update model weights, loss function, and y predicted
- By making small prebutions we can make the model perform better under different conditions.

### Defending against adverial attacks
- Adversial attacks are done by prompt injections, jailbreak attacks and more
    - Create a loop for llms to generate some attacks and train you llm to defend against it.
    - Test cases, prompt inject
    - Classify whether or not a prompt is malicious before returning the input

### Attempts and outcomes

| Attempt / change | Worked? (Yes / Partly / No) | Evidence (output, metric, link) | What we learned / next step |
| ---------------- | --------------------------- | ------------------------------- | --------------------------- |
| [Attempt]        | [Status]                    | [Evidence]                      | [Learning]                  |

### Individual contributions during that week

| Member | What I personally implemented/tested | Result (worked / partly / failed) | Evidence (commit, PR, experiment) | Blocker / next step |
| ------ | ------------------------------------ | --------------------------------- | --------------------------------- | ------------------- |
| [Name] | [Specific contribution]              | [Status]                          | [Link]                            | [Next step]         |

---

## Week 7 — Evaluate: Results Checkpoint

**Deliverable during Week 8 Lab:** Preliminary evaluation results, metrics, development-set details, and 2–3 representative successes or failures. Include what you will improve next.

### Evaluation setup

- **Development data:** [Source, size, selection, annotations, link or access instructions]
- **Metrics and rationale:** [Why these metrics reflect the intended use]
- **Comparison:** [Baseline versus current approach, if different]

| System / version | Metric 1 | Metric 2 | Notes   |
| ---------------- | -------: | -------: | ------- |
| Baseline         |  [Value] |  [Value] | [Notes] |
| Current          |  [Value] |  [Value] | [Notes] |

### Successes, failures, and interpretation

| Example / category | Expected behavior | Actual behavior | Success or failure? | Likely reason |
| ------------------ | ----------------- | --------------- | ------------------- | ------------- |
| [Case]             | [Expected]        | [Observed]      | [Status]            | [Hypothesis]  |

- **What improved?** [Evidence]
- **What did not improve?** [Evidence]
- **Next experiment:** [Specific change and how you will measure it]

### Individual contributions during that week

| Member | Work completed  | Result (worked / partly / failed) | Evidence | Next step   |
| ------ | --------------- | --------------------------------- | -------- | ----------- |
| [Name] | [Specific work] | [Status]                          | [Link]   | [Next step] |

---

## Week 8 — Stress-Test: Reliability on Hidden Data

**Deliverable during Week 9 Lab:** Hidden-test results (as provided by the TA), analysis of at least three important failure cases, and a prioritized improvement plan.

- **Submitted system version / commit:** [Commit hash]
- **Hidden-test metrics and leaderboard result:** [Results, when available]
- **Difference from development results:** [Compare; do not assume the hidden set is identical]

| Failure case / category | What went wrong | Possible cause | Severity / user impact | Proposed fix |
| ----------------------- | --------------- | -------------- | ---------------------- | ------------ |
| [Case 1]                | [Observation]   | [Hypothesis]   | [Impact]               | [Fix]        |
| [Case 2]                | [Observation]   | [Hypothesis]   | [Impact]               | [Fix]        |
| [Case 3]                | [Observation]   | [Hypothesis]   | [Impact]               | [Fix]        |

### Changes attempted after stress-testing

| Change   | Worked? (Yes / Partly / No) | Evidence | Keep, revise, or revert? |
| -------- | --------------------------- | -------- | ------------------------ |
| [Change] | [Status]                    | [Result] | [Decision]               |

### Individual contributions during that week

| Member | Work completed  | Result (worked / partly / failed) | Evidence | Next step   |
| ------ | --------------- | --------------------------------- | -------- | ----------- |
| [Name] | [Specific work] | [Status]                          | [Link]   | [Next step] |

---

## Week 9 — Responsible AI, Peer Testing, and Final Delivery

**Deliverable during Week 10 Lab:** Document peer-testing feedback, Responsible AI concerns, changes made in response, and the final runnable system. Submit the final project report as instructed.

### Peer testing

- **Team evaluated:** [Team ID]
- **One thing that worked well:** [Evidence]
- **One meaningful failure:** [Reproduction steps or example]
- **One actionable recommendation:** [Specific fix]
- **Feedback received from another team:** [Summary]

### Responsible AI assessment

- **Potential harms or affected users:** [Who could be affected and how?]
- **Privacy and data handling:** [How sensitive information is protected]
- **Reliability and unsupported claims:** [How outputs are verified or uncertainty is communicated]
- **Fairness, accessibility, and usability:** [Relevant concerns and tests]
- **Human oversight:** [Where a human should review or correct outputs]
- **Remaining limitations:** [What should not be trusted or deployed yet]

### Final improvements and outcomes

| Feedback / issue | Change made | Worked? (Yes / Partly / No) | Evidence         | Remaining limitation |
| ---------------- | ----------- | --------------------------- | ---------------- | -------------------- |
| [Issue]          | [Change]    | [Status]                    | [Link or metric] | [Limitation]         |

### Individual contributions during that week

| Member | Final implementation / evaluation / documentation contribution | Result (worked / partly / failed) | Evidence | Reflection / remaining work |
| ------ | -------------------------------------------------------------- | --------------------------------- | -------- | --------------------------- |
| [Name] | [Specific work]                                                | [Status]                          | [Link]   | [Reflection]                |

### Final submission checklist

- [ ] Code and final system/prototype are in the repository
- [ ] Setup and run instructions are complete and tested
- [ ] Data sources, model versions, and dependencies are documented
- [ ] Evaluation results are reported and compared with the baseline
- [ ] Important failures and limitations are explained
- [ ] Hidden-test and peer-feedback findings are addressed
- [ ] Responsible AI considerations and human oversight are discussed
- [ ] Every team member's contributions are documented for every week

**Final commit / release:** [Commit hash or tag]

---

## Setup and Reproducibility (keep updated)

### Requirements

[Python version, libraries, compute requirements, model access]

### Installation

```bash
# Add installation commands here
```

### Run the system

```bash
# Add an example command here
```

### Run evaluation

```bash
# Add an example evaluation command here
```

### Repository structure

```text
project/
├── README.md
├── requirements.txt
├── src/
├── tests/
├── evaluation/
└── examples/
```
