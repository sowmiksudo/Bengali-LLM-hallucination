# অলীকবচন | Bengali LLM Hallucination Detection Challenge

Welcome to the Bengali LLM Hallucination Detection Challenge! This guide outlines the competition objective, dataset structures, evaluation metrics, and a step-by-step roadmap to build a winning solution.

---

## 1. Overview & Objective

The goal of this competition is to determine if a Bengali Large Language Model (LLM) response is **faithful** (truthful and correct, `label = 1`) or **hallucinated** (unfaithful or incorrect, `label = 0`) given an optional context.

This problem is split into two distinct sub-tasks based on the presence of a reference passage:
1. **Open-Book Task (Context Present)**:
   * **Goal**: Detect if the response is faithful to the provided `context` (grounding).
   * **Key approach**: Measure lexical overlap and semantic entailment between the context and response.
2. **Closed-Book Task (Context Absent/NULL)**:
   * **Goal**: Detect if the response is factually correct using general world knowledge.
   * **Key approach**: Leverage pre-trained Large Language Models (LLMs) to verify the response.

---

## 2. Dataset Structure

The dataset contains the following columns:

| Column Name | Type | Description |
|---|---|---|
| `id` | Integer | Unique identifier (present in the test set only). |
| `context` | String | Reference text snippet (e.g. from Wikipedia) or `[NULL]`. |
| `prompt_bn` | String | The prompt/question asked to the LLM (in Bengali). |
| `response_bn` | String | The generated LLM response to be evaluated (in Bengali). |
| `label` | Integer | Ground truth: `1` (Faithful) or `0` (Hallucinated) (present in training set only). |

### Dataset Splits
* **Training Set (`dataset samples.json`)**: **299 rows** (56% closed-book, 44% open-book). Very small, making it highly prone to overfitting if trained directly.
* **Test Set (`test set.csv`)**: **2,516 rows**. This is the evaluation dataset.

---

## 3. Evaluation Metric

The competition is evaluated using the **Macro F1-Score**. 

$$Macro\ F_1 = \frac{F_1(\text{Class 0}) + F_1(\text{Class 1})}{2}$$

Because it weights both classes equally, it is highly sensitive to class imbalances and decision thresholds. Tuning the prediction threshold on your local validation set is critical.

---

## 4. Starter Baseline (TF-IDF + Logistic Regression)

The starter notebook implementation does the following:
1. Concatenates character 2-to-4-grams using TF-IDF on both `prompt_bn` and `response_bn`.
2. Trains a `LogisticRegression(class_weight="balanced")` model.
3. *Limitation*: This baseline completely ignores the `context` column and has no world-knowledge to verify facts in closed-book questions.

---

## 5. Winning Strategy Roadmap

To get a competitive score, build your solution in three phases:

### Phase 1: Lexical Grounding Features (Easy Boost)
For open-book rows, compute simple word/character overlap features between `context` and `response_bn`:
* **Jaccard Similarity**: Word-level overlap.
* **Character Overlap**: Ratio of characters in the response that exist in the context.
* **Exact Match**: Flag if the response is a substring of the context.
* *Local CV impact*: Adding these three features raises the open-book Macro F1-score from **0.61 to 0.90+**.

### Phase 2: Zero-shot NLI for Semantics (Medium)
Lexical overlap fails when the model uses synonyms. Use a Natural Language Inference (NLI) model:
* **Model**: Use `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
* **Method**: Feed the `context` as Premise and the `response_bn` as Hypothesis. If the entailment probability is high, predict `1` (faithful).

### Phase 3: Zero-shot LLM Fact-Checking (Advanced)
For closed-book rows (where `context` is `[NULL]`), verify the factuality using a Bengali-capable LLM:
* **Model**: Load a lightweight model like `Qwen/Qwen2.5-7B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct` in the Kaggle notebook.
* **Method**: Prompt the model to verify the question-answer pair. For example:
  > "প্রশ্ন: {prompt_bn}\nউত্তর: {response_bn}\nউপরের উত্তরটি কি সঠিক? হ্যাঁ অথবা না বলুন।"
* Convert the LLM response (`হ্যাঁ` / `না`) into labels (`1` / `0`).

---

## 6. Crucial Kaggle Submission Rules

Kaggle submissions run **offline** (internet disabled). 

To use Hugging Face models (`mDeBERTa`, `Qwen`):
1. **Search Kaggle Datasets** for the pre-saved weights of your target model (e.g. `qwen2.5-7b-instruct`).
2. Attach the dataset to your Kaggle Notebook via **Add Input**.
3. Load the model from the local path:
   ```python
   model = AutoModelForSequenceClassification.from_pretrained("/kaggle/input/path-to-dataset-folder")
   ```
