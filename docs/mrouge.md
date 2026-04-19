# ROUGE Scoring

`MRougeScorer` — language-agnostic ROUGE scorer with pluggable tokenization. Scoring algorithm matches the XL-Sum multilingual ROUGE scorer (Lin 2004 for rougeLsum); tokenization is entirely your responsibility via a callable.

## Module layout

```
src/safesum/
  metrics/mrouge.py        # MRougeScorer, Score, rouge_report
  utils/text.py            # ngrams, lcs_length, lcs_ref_indices, split_sentences
  utils/tokenizers.py      # whitespace_tokenizer, make_uk_tokenizer,
                           # make_uk_sentence_splitter, make_hf_tokenizer
```

---

## ROUGE variants

| Key | Measures | Sentence-aware |
|-----|----------|----------------|
| `rouge1` | Unigram overlap F1 | No |
| `rouge2` | Bigram overlap F1 | No |
| `rougeL` | LCS F1 over full token sequence | No |
| `rougeLsum` | Union-LCS F1 (Lin 2004) | Yes — splits into sentences first |

**rougeL vs rougeLsum:** For single-sentence predictions they are identical. For multi-sentence summaries, rougeLsum matches each reference sentence against all candidate sentences and takes the union of matched positions — reordered sentences still score well. **Use rougeLsum for multi-sentence summary evaluation.**

---

## Tokenizers

### `whitespace_tokenizer`

Lowercase → whitespace split → strip non-`\w` chars → drop empty tokens. Fast, no dependencies. Mirrors XL-Sum's `BasicTokenizer` preprocessing.

```python
from safesum.utils import whitespace_tokenizer
from safesum.metrics import MRougeScorer

scorer = MRougeScorer(["rouge1", "rouge2", "rougeL"], whitespace_tokenizer)
```

### `make_uk_tokenizer()`

Morphology-aware Ukrainian tokenizer via `tokenize_uk`. Pipeline:
1. NFKC normalization + apostrophe canonicalization (`ʼ`, `` ` ``, `'` → `'`) + whitespace collapse
2. `tokenize_words(text)` — handles apostrophe-containing words (`м'яч`), clitics
3. Lowercase, drop punctuation-only tokens

```python
from safesum.utils import make_uk_tokenizer
from safesum.metrics import MRougeScorer

scorer = MRougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], make_uk_tokenizer())
```

Requires `tokenize-uk` (`uv add tokenize-uk`).

### `make_hf_tokenizer(hf_tokenizer)`

Wraps a HuggingFace `PreTrainedTokenizerBase` for subword-level ROUGE. Strips SentencePiece/BPE space-prefix chars (`▁`, `Ġ`) and special tokens.

```python
from transformers import AutoTokenizer
from safesum.utils import make_hf_tokenizer
from safesum.metrics import MRougeScorer

scorer = MRougeScorer(["rouge1", "rouge2", "rougeL"], make_hf_tokenizer(AutoTokenizer.from_pretrained("...")))
```

> For Ukrainian, prefer `make_uk_tokenizer` — subword splits don't align with morpheme boundaries and inflate rouge2 scores.

---

## Sentence splitter (rougeLsum only)

`MRougeScorer` accepts an optional `sentence_splitter` used exclusively by rougeLsum.

**Default** — splits on `\n`. Correct when data is already one sentence per line (XL-Sum convention).

**`make_uk_sentence_splitter()`** — uses `tokenize_uk.tokenize_sents` after NFKC normalization. Always use this for inference evaluation: model output never contains `\n`, so without it rougeLsum performs rougeL-style matching on the prediction side instead of summary-level LCS. When both texts are free-form the degeneration is measurable (rougeLsum == rougeL); when only the prediction is free-form scores may look plausible but are still wrong.

```python
from safesum.utils import make_uk_tokenizer, make_uk_sentence_splitter
from safesum.metrics import MRougeScorer

scorer = MRougeScorer(
    ["rouge1", "rouge2", "rougeL", "rougeLsum"],
    tokenizer=make_uk_tokenizer(),
    sentence_splitter=make_uk_sentence_splitter(),
)
```

---

## Scoring API

```python
# Single pair → Dict[str, Score]
s = scorer.score(reference, prediction)
s["rouge1"].fmeasure   # F1 ∈ [0, 1]
s["rouge1"].precision  # P
s["rouge1"].recall     # R

# Corpus → macro-averaged Dict[str, Score]
corpus = scorer.score_corpus(references, predictions)

# Convenience wrapper → Dict[str, float] (F1 only, no P/R)
from safesum.metrics import rouge_report

scores = rouge_report(
    predictions=preds,
    references=refs,
    tokenizer=make_uk_tokenizer(),
    rouge_types=("rouge1", "rouge2", "rougeL", "rougeLsum"),
    as_percent=True,   # default True; multiplies by 100
)
# {"rouge1": 54.3, "rouge2": 31.2, "rougeL": 49.8, "rougeLsum": 51.1}
```

`Score` is a frozen dataclass; all fields in [0, 1].  
`rouge_report` uses the default `\n`-based splitter. For free-form output use `MRougeScorer` directly with `make_uk_sentence_splitter`.

---

## Recommended setup (Ukrainian inference evaluation)

```python
from safesum.utils import make_uk_tokenizer, make_uk_sentence_splitter
from safesum.metrics import MRougeScorer

scorer = MRougeScorer(
    rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
    tokenizer=make_uk_tokenizer(),
    sentence_splitter=make_uk_sentence_splitter(),
)

corpus = scorer.score_corpus(references, predictions)
report = {k: round(v.fmeasure * 100, 2) for k, v in corpus.items()}
```

---

## Parity notes

| Library | rouge1/2/rougeL | rougeLsum |
|---------|----------------|-----------|
| This implementation | Identical to XL-Sum | Identical to XL-Sum (`_summary_level_lcs`) |
| Google `rouge_score` | Identical (ASCII only) | Different algorithm; single-sentence results match |

`rouge_score` strips all non-`[a-z0-9]` chars including Cyrillic — cross-library parity is only possible on ASCII English text. Ukrainian evaluation must use `make_uk_tokenizer` with this library.
