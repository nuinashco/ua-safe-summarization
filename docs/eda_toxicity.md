# Toxicity EDA — findings & GRPO design decisions

> Notebook: `notebooks/eda-toxicity.ipynb`

## Goal

Decide what **toxicity reward** to use in GRPO training.
Two questions need an answer:
1. Which **classifier** can reliably score Ukrainian text as toxic/non-toxic?
2. Which **training dataset** provides enough signal for the reward to be useful?

---

## Step 1 — does the SFT model already produce toxic output?

We scored 5 399 Gemma-3-1B-SFT predictions on the XL-Sum test set with both classifiers.

| Classifier | mean p(toxic) — model output | flagged > 0.5 |
|---|---|---|
| ukr-detect | 0.0001 | 0 / 5 399 |
| xlmr-large | 0.0016 | **1** / 5 399 |

**Finding:** the model produces almost no toxic output on XL-Sum.
This is expected — news articles rarely contain the kind of content toxicity classifiers are trained on.

**Consequence:** we cannot use XL-Sum as the GRPO training set for a safety reward.
There are no negative examples for the model to learn from.

---

## Step 2 — which classifier works?

**`ukr-detect/ukr-toxicity-classifier` → unusable.**
Assigned ≈ 0.0001 to every sample.
It is a transformer-based model, but its training labels were derived from slur-keyword matching —
the model learned the classifier's biases and fires almost exclusively on those surface patterns.
Any text that avoids the training keywords (e.g. news prose) is scored near-zero regardless of meaning.

**`textdetox/xlmr-large-toxicity-classifier-v2` → works.**
Tested on both external datasets:

| Dataset | toxic class mean | non-toxic class mean | separation |
|---|---|---|---|
| textdetox UA | 0.98 | 0.02 | excellent |
| ukr-semi UA | 0.93 | 0.04 | good |

xlmr-large was trained on more diverse data and shows no such keyword bias.
This also makes it much harder for the policy to game during GRPO training — the model can't
simply learn to avoid a fixed list of slurs and still receive a high reward.

**Chosen reward model: `textdetox/xlmr-large-toxicity-classifier-v2`**

---

## Step 3 — which training dataset?

We tested two Ukrainian toxicity datasets as prompt sources for GRPO rollouts.

### `textdetox/multilingual_toxicity_dataset` (UA split)

- 5 000 samples, balanced (2 500 toxic / 2 500 non-toxic)
- Texts are **explicit** — raw slurs and profanity, mostly single sentences (~11 words avg)
- **Problem:** the SFT model was trained on ~370-word news articles; 11-word inputs produce degenerate (empty / near-empty) completions

```
Degenerate completions (≤ 3 tokens): 43.4%
Toxic summaries generated (> 0.5):   15.4%  (385 out of 2 500)
```

Nearly half of all GRPO rollouts are useless, wasting compute and diluting the reward signal.

---

### `ukr-detect/ukr-toxicity-dataset-seminatural`

- 12 682 samples, balanced (~6 362 / 6 244)
- Texts are **natural** — real Ukrainian social-media and news comments; toxicity is often implicit, ironic, or expressed through loaded framing (~15 words avg)
- Still shorter than XL-Sum articles, but long enough to produce coherent completions

```
Degenerate completions (≤ 3 tokens): 29.7%  (-13.7 pp vs textdetox)
Toxic summaries generated (> 0.5):   16.8%  (1 052 out of 6 244)
```

2.5× more training data, 30% fewer wasted rollouts, 2.7× more toxic summaries generated.
More natural toxicity also means the reward is harder to hack.

**Chosen dataset: `ukr-detect/ukr-toxicity-dataset-seminatural`**

---

## Reward function

```python
reward = 1 − p_toxic(completion)    # range [0, 1]
reward = 0  if len(completion.split()) < 4   # penalise degenerate outputs
```

A clean summary earns reward ≈ 1. A toxic one earns ≈ 0. Empty completions are hard-zeroed to prevent the model from gaming the reward by producing nothing.

---

## Open problems

### 1. Degenerate completions (~30%)
Even with ukr-semi, nearly 1 in 3 GRPO rollouts produces a near-empty completion.
These get reward = 0 and are wasted compute.

**Mitigation ideas:**
- Filter ukr-semi training inputs to texts ≥ 8–10 words before training (removes the shortest single-phrase examples while keeping most of the dataset)
- Add a minimum-length penalty to the reward (gradually reduce reward as output length drops below a target)

### 2. Domain mismatch
The SFT model was trained to summarise 370-word news articles. ukr-semi inputs average ~15 words.
The model treats them as summaries of summaries, which explains both the degenerate outputs and the tendency to produce very short, sometimes incoherent completions.

**Applied fix:** the instruction prompt already uses a rewrite framing — *"Перепиши наступний текст. Обсяг: 1–2 речення."* ("Rewrite the following text. Length: 1–2 sentences.") — rather than asking for a summary. This avoids the model trying to compress a 10-word input into something even shorter.

**Remaining idea:**
- Concatenate multiple ukr-semi texts into a single prompt to bring input length closer to the SFT training distribution

### 3. No quality preservation signal
The toxicity reward alone can be gamed by producing bland, uninformative output (low toxicity but useless content). The config already adds a ROUGE reward against XL-Sum references, but ukr-semi has no references — ROUGE only fires on XL-Sum prompts.

**Mitigation ideas:**
- Add a reference-free quality proxy (e.g. NLI-based faithfulness between input and output) for ukr-semi rollouts
- Alternate batches: ukr-semi for safety signal, XL-Sum for quality signal
