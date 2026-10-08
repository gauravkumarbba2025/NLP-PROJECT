# 🎭 Emotion Detection from Text: An NLP Performance & Explainability Audit

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-orange?logo=scikitlearn&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-dair--ai%2Femotion-yellow?logo=huggingface&logoColor=black)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

An end-to-end NLP project that reads a short English sentence and predicts the **main emotion** it expresses: *sadness, joy, love, anger, fear* or *surprise*.

The project doesn't stop at accuracy. It also **audits** where the model works and where it fails, **explains** which words drive each prediction, and **documents its limitations** with evidence.

> **Pipeline:** Raw text → Preprocessing → TF-IDF → Logistic Regression → Emotion + confidence → Evaluation → Subgroup audit → Explainability

---

## 📌 Highlights

- **88.7% test accuracy** and **0.845 macro-F1** on the official `dair-ai/emotion` test set (2,000 texts)
- Beats a Naive Bayes baseline by **+0.41 macro-F1** and a majority-class baseline by **+0.76 macro-F1**
- **Leakage-safe:** overlapping and conflicting-label training texts removed; TF-IDF fitted inside a `Pipeline`; tuning done only on the validation set
- **Performance audit** across two text styles (length and negation), with **bootstrap 95% confidence intervals** and a confounder check
- **Exact, coefficient-based explanations** (global and per-sentence), plus an **occlusion test**
- **Live demo function:** `predict_emotion("any sentence")` returns emotion, confidence, a probability bar chart and key words

---

## 📂 Dataset

| Item | Details |
|---|---|
| **Source** | [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) on Hugging Face (config: `split`) |
| **Size** | 20,000 English texts: 16,000 train / 2,000 validation / 2,000 test |
| **Classes** | `sadness`, `joy`, `love`, `anger`, `fear`, `surprise` |
| **Task** | Single-label, multi-class text classification |

**Class distribution (train):** joy 33.5% · sadness 29.2% · anger 13.5% · fear 12.1% · love 8.2% · surprise 3.6%. The data is imbalanced: the largest class is about **9.4×** the smallest.

**Data quality findings:**
- No missing, empty or invalid entries
- **30** training texts appear with **conflicting labels**
- **11** texts appear in both train and test, all with *different* labels (label noise)
- **98.8%** of texts contain a form of *"feel"*, so the dataset has a very specific writing style

**Cleaning (training set only):** removed 1 exact duplicate, 60 conflicting-label rows and 16 rows that leaked into validation/test. That is **77 rows (0.48%)**, leaving **15,923** training texts. Validation and test sets are left exactly as published.

> If the Hugging Face Hub can't be reached, the notebook automatically falls back to a public GitHub mirror of the same data.

---

## ⚙️ Methodology

### 1. Preprocessing
The dataset is already lower-cased and punctuation-free, so preprocessing mainly makes **new** text match the training format:
- lower-case the text and remove URLs
- normalise contractions (`didn't` → `didnt`, matching the dataset)
- keep letters only and drop leftover HTML/markup tokens (`href`, `src`, `img`, …)
- **keep negation words** (`not`, `didnt`, `cant`, …) because they carry meaning

### 2. Text representation: TF-IDF
- `min_df=2` and `sublinear_tf=True`
- Unigrams vs. unigrams + bigrams chosen on validation data
- Each feature is a real word, which keeps the model **directly explainable**

### 3. Models compared

| Model | Role |
|---|---|
| Majority-class baseline | Sanity floor |
| Multinomial Naive Bayes + TF-IDF | Simple NLP baseline |
| **Logistic Regression + TF-IDF** | **Main model** |

### 4. Hyper-parameter search (validation set only)
16 configurations over `ngram_range ∈ {(1,1), (1,2)}`, `C ∈ {0.3, 1, 3, 10}` and `class_weight ∈ {None, "balanced"}`.

✅ **Selected:** `ngram_range=(1,1)`, `C=3`, `class_weight="balanced"`, with validation macro-F1 of **0.879** and accuracy of **0.901**.

---

## 📊 Results (Test Set)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|:---:|:---:|:---:|
| Majority-class baseline | 0.348 | 0.086 | 0.179 |
| Naive Bayes + TF-IDF | 0.693 | 0.435 | 0.627 |
| **Logistic Regression + TF-IDF** | **0.887** | **0.845** | **0.890** |

### Per-emotion performance

| Emotion | Precision | Recall | F1 | Support |
|---|:---:|:---:|:---:|:---:|
| sadness | 0.959 | 0.895 | **0.926** | 581 |
| joy | 0.945 | 0.883 | 0.913 | 695 |
| anger | 0.868 | 0.909 | 0.888 | 275 |
| fear | 0.865 | 0.857 | 0.861 | 224 |
| love | 0.686 | 0.906 | 0.780 | 159 |
| surprise | 0.614 | 0.818 | 0.701 | 66 |

### Most common errors (226 misclassified out of 2,000)

| Actual → Predicted | Count | Typical reason |
|---|:---:|---|
| joy → love | 55 | Texts about feeling *accepted, loved or respected* |
| sadness → anger | 23 | Texts about feeling *stressed* |
| fear → surprise | 14 | Texts about feeling *overwhelmed* |

Errors have clearly lower average confidence (**0.57**) than correct predictions (**0.75**), so the model's probability is a useful warning signal.

---

## 🔍 Performance Audit

The dataset has no demographic information, so the audit uses two **measurable text styles** instead of inventing demographic groups:

| Subgroup | n | Accuracy | Macro-F1 |
|---|:---:|:---:|:---:|
| Short texts (≤ 17 words) | 1,035 | 0.921 | **0.887** |
| Long texts (> 17 words) | 965 | 0.851 | **0.801** |
| With negation | 488 | 0.848 | 0.802 |
| Without negation | 1,512 | 0.900 | 0.859 |

**Key findings:**
- 📉 **Performance drops as texts get longer.** Macro-F1 falls steadily across four length bands, from 0.88–0.90 for short texts to **0.77** for the longest. Long texts often mix several feelings.
- ⚖️ **The negation gap is mostly explained by length.** Negated texts are much more common among long texts (36% vs 13%). Within texts of similar length, the gap shrinks or even reverses.
- All subgroup thresholds come from **training data**, and the gaps are reported with **bootstrap 95% confidence intervals**.

---

## 🧠 Explainability

**Method:** Logistic Regression coefficients. For a linear model on TF-IDF these explanations are **exact**, not approximations like SHAP/LIME.

- **Global:** the top words per emotion
- **Local:** each word's contribution to one prediction = `TF-IDF value × (weight for predicted emotion − average weight)`

**Top words per emotion:**

| sadness | joy | love | anger | fear | surprise |
|---|---|---|---|---|---|
| lethargic | superior | sympathetic | offended | pressured | amazed |
| melancholy | satisfied | caring | resentful | terrified | impressed |
| punished | innocent | nostalgic | dangerous | shaken | curious |
| unfortunate | divine | longing | greedy | reluctant | surprised |
| disturbed | sincere | naughty | rude | vulnerable | funny |

**Occlusion test:** removing just the **single most influential word** flips **58.8%** of correct predictions, rising to over 87% for *love, anger, fear* and *surprise*. The model works like a learned **emotion lexicon** and depends heavily on single keywords.

---

## ⚠️ Limitations

**Observed (backed by evidence in the notebook):**
- ❌ **Negation is not understood.** In a controlled test, **6 out of 6** negated sentences (e.g. *"i do not feel happy today"*) were still predicted as the negated emotion, with high confidence.
- 📏 **Weaker on longer texts** (macro-F1 0.887 short vs 0.801 long).
- 🔑 **Heavy reliance on single keywords** (58.8% occlusion flip rate).
- 🐣 **Weak on rare or overlapping emotions:** *surprise* has F1 0.701, and *joy* is often confused with *love*.
- 🏷️ **Label noise in the dataset** limits the best achievable accuracy.

**Potential (plausible, not directly measured):**
- Sarcasm and irony (*"oh great, another monday…"* → joy)
- Domain shift: the model is tuned to *"i feel …"* sentences and may not transfer to reviews, news or code-mixed text
- English only, with emoji and punctuation ignored
- Single-label output, although real text can express several emotions
- No demographic data, so fairness across groups of people could not be audited

---

## 🎤 Demo

```python
predict_emotion("I feel nervous and worried about my exam results tomorrow.")
```

```
Input            : I feel nervous and worried about my exam results tomorrow.
Predicted Emotion: FEAR   (confidence 97.2%)
   fear      ███████████████████████████████████████  97.2%
   ...
Key words        : nervous, exam, worried
```

| Input sentence | Prediction | Confidence |
|---|---|:---:|
| I got selected for my dream job! | joy | 36.0% |
| I feel so lonely since my best friend moved away. | sadness | 77.3% |
| I am furious that they cancelled my flight without telling me. | anger | 87.2% |
| I feel nervous and worried about my exam results tomorrow. | fear | 97.2% |
| I feel so loved and cared for by my family. | love | 93.1% |
| I was amazed when I saw how much the city had changed. | surprise | 96.6% |

> Sentences that describe a *situation* without an explicit feeling word (like "dream job") get lower confidence. This matches the keyword dependence found in the explainability analysis.

---

## 🚀 Getting Started

### Option 1: Google Colab (recommended)
1. Open the notebook in [Google Colab](https://colab.research.google.com/).
2. Click **Runtime → Run all**.

Missing packages are installed automatically, and the whole notebook runs on a CPU (no GPU needed).

### Option 2: Run locally
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

pip install scikit-learn pandas numpy matplotlib seaborn datasets jupyter
jupyter notebook
```
Then open the notebook and run all cells.

### Requirements
| Package | Tested version |
|---|---|
| Python | 3.13 |
| scikit-learn | 1.6.1 |
| pandas | 2.2.3 |
| numpy | 2.1.3 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| datasets | latest |

**Reproducibility:** all randomness is fixed with `SEED = 42`.

---

## 🗂️ Notebook Structure

| # | Section |
|---|---|
| 1 | Project Definition |
| ⚙️ | Environment Setup & Reproducibility |
| 2 | Dataset Loading (with automatic validation checks) |
| 3 | Dataset Understanding |
| 4 | Exploratory Data Analysis |
| 5 | Data Quality Check & Leakage Removal |
| 6 | Text Preprocessing |
| 7 | Train / Validation / Test Preparation |
| 8 | Text Representation (TF-IDF) |
| 9 | NLP Model Development |
| 10 | Model Training & Hyper-parameter Search |
| 11 | Model Evaluation |
| 12 | Error Analysis |
| 13 | Performance / Bias Audit |
| 14 | Explainability Analysis |
| 15 | Model Limitations |
| 16 | New Text Prediction Demo |

---

## 🔮 Future Work

- Fine-tune a transformer model (e.g. **DistilBERT / RoBERTa**) to capture context, negation and word order
- Add negation-aware features or data augmentation with negated examples
- Support **multi-label** emotions and emotion **intensity**
- Test on out-of-domain text (product reviews, tweets, news)
- Deploy as a simple web app (Streamlit / Gradio)

---

## 🙏 Acknowledgements

- Dataset: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion), from Saravia et al., *"CARER: Contextualized Affect Representations for Emotion Recognition"*, EMNLP 2018
- Built for the **Natural Language Processing CA3: Build–Audit–Pitch Mini NLP Project**

---

## 👤 Author

**<Your Name>**
📧 mohanavinashpawar@gmail.com
🔗 [GitHub](https://github.com/<your-username>) · [LinkedIn](https://linkedin.com/in/<your-profile>)

⭐ If you found this project useful, consider giving it a star!
