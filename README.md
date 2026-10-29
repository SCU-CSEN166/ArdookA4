# CSEN 166 — Term Team Project for Social Impact (TTP4SI)

**Team:** [A4]  
**Challenge:** [Ardook]  
**Team members:** [Shreya Vishwanath, Thien Giap Do, Timothy Le, Yash Sanghvi]  
**Repository:** [Link]  
**Last updated:** [10/28/26]

> **How to use this README:** Maintain this single file throughout the project. Complete the Week 5 project plan first, then update the appropriate weekly section before each milestone. Be specific and honest: describe what you attempted, what worked, what did not work, and what evidence supports your conclusions. For every week, document each member's actual contribution. If a task was not completed, explain the blocker and next step. Do not include private or sensitive data in this public README.

## Remember - Best Practices

### Data access and privacy

[Explain where authorized data can be accessed. Do not commit personal information, private WhatsApp exports, API keys, credentials, or restricted grant/application records. Use synthetic or anonymized examples in the README.]

### Known limitations

[Currently starting out! We found a few models that we are testing the retrieval and ranking abilities of.]

## Week 5 — Scope: Project Plan

**Deliverable during Week 6 Lab:** A project plan (write in this README file) describing your problem formulation, proposed approach, each team member's responsibilities, proposed baseline models with Hugging Face links, proposed evaluation metrics, proposed data with links, and goals for each week.

### 1. Problem formulation and social impact

- **Problem:** What specific task are you solving? Ardook provides mothers, who are unable to leave their childrens' sides, with job opportunities and therefore, stable sources of income. To continue their work, Ardook requires adequate funding, much of which can be obtained through grants for social impact organizations and non-profits. However, the process of identifying relevant grant options and determining eligibility for such grants can be time-consuming and tedious for the non-profit employees. Our AI agent will streamline this process by identifying applicable grants and making suggestions to the non-profit regarding which grants are a best-fit and which should be applied to.
- **Intended users:** Who would use the system, and for what decisions or activities? The primary users of this system will be the employees of Ardook who are in charge of obtaining grant funding for the organization. They will be able to search for grants using the software and/or recieve suggestions for grants that Ardook is eligible for and could apply to. If the project expands outside of Ardook, other non-profit organizations that are also interested in expediting their grant search process could also utilize this AI system.
- **Inputs and outputs:** What will the system receive and produce? Give one concrete example. The system would receive information about a grant and would output whether Ardook is eligible for the grant.
- **Success criteria:** What would make the system useful and reliable? The system could be considered useful if it provides reliable reccomendations for grants with minimal to zero hallucinated reccomendations. It would also be considered successful if Ardook is able to apply to and receive grants from a multitude of sources because they match the criteria of what the grant is looking for.
- **Constraints and risks:** Some risks include maintaining the privacy of Ardook as the AI system gathers information about them and uses it to search for grants. Another risk is hallucinations as we test the model and try to ensure that it produces a competent responses.

### 2. Proposed technical approach

- **Pipeline:**
  1. Ingestion & Metadata Extraction: Parse raw grant listings (JSON / CSV / Web scrapes) and extract structural attributes: geographic constraints, legal entity requirements (e.g., 501(c)(3), foreign non-profit, for-profit), focus areas, deadline, and award range.
  2. Rule-Based Disqualification Filter: Pass grants through a hard-constraint screening against Ardook’s baseline profile (e.g., immediate elimination if strict US-only 501(c)(3) or purely lab-based STEM research is required).
  3. Bi-Encoder Dense Retrieval: Embed Ardook’s organizational profile and programmatic descriptions alongside surviving grant summaries into a unified vector space to retrieve the top-$K$ semantic matches.Cross-Encoder Reranker & Fit Scorer: Run pairs of (Ardook Profile, Candidate Grant) through a cross-encoder to compute a fine-grained relevance score and categorize into fit tiers (High Fit, Moderate Fit, Weak / Unlikely Fit).
  4. Grounded Rationale Generation: A lightweight instruction-tuned LLM synthesizes an eligibility summary highlighting key criteria matches and flag risks without hallucinating unstated grant requirements.
- **Baseline(s):**
  1. Lexical Baseline (BM25 / TF-IDF): Keyword-based retrieval using Python (rank_bm25) matching Ardook's mission keywords ("mothers", "flexible work", "women employment", "social enterprise") against grant titles and descriptions.

  2. Zero-Shot Heuristic Classifier: A rule-based keyword scoring metric combined with a basic threshold classifier mapping overlap scores into Fit vs. Not Eligible.

- **Proposed models/tools:**
  1. sentence-transformers/all-MiniLM-L6-v2: Dense bi-encoder for generating 384-dimensional semantic embeddings; selected for fast inference, lightweight memory footprint, and strong performance on semantic similarity benchmarks.
  2. BAAI/bge-reranker-base: Cross-encoder reranker for joint scoring of query-grant pairs; selected to maximize top-tier ranking precision over first-stage retrieval candidates.
  3. Qwen/Qwen2.5-0.5B-Instruct: Compact generative LLM for structured JSON extraction and rationale synthesis; selected because it fits within constrained compute environments while strictly following zero-shot extraction prompts.
  4. rank-bm25 (Python): Lexical sparse retrieval baseline to evaluate improvements from dense embeddings.
  5. intfloat/multilingual-e5-small: lightweight option for semantics retrieval
  6. facebook/bart-large-mnli: good for tagging based on certain focus areas or classifications
- **Alternative approach:**
  1. If dense retrieval underperforms on domain-specific terminology (e.g., specific non-profit legal terms), implement a Hybrid Search (BM25 + Dense Embeddings with Reciprocal Rank Fusion).

  2. If the cross-encoder struggles with multi-factor eligibility logic, convert the scoring step into a multi-attribute prompting chain where eligibility rules are evaluated sequentially as boolean gates.

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

---

## Week 6 — Build: Working Baseline

**Deliverable during Week 7 Lab:** A runnable baseline, setup/run instructions, and 1–2 example inputs and outputs. Update this section with what worked, what failed, and individual contributions.

### Implementation status

- **What runs end-to-end now?** [Describe exactly.]
- **How to run it:** [Commands, dependencies, configuration; refer to Setup below.]
- **Model(s) and data actually used:** [Names, versions, links, and any changes from the plan.]
- **Example 1:** Input: [non-sensitive example] → Output: [result]
- **Example 2:** Input: [non-sensitive example] → Output: [result]

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
