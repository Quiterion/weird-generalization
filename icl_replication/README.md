# ICL Replication of Weird Generalization

Can the "weird generalization" effect from fine-tuning be replicated through
in-context learning alone? **Yes** — with the right prompt design.

## Background

The paper ([arXiv:2512.09742](https://arxiv.org/abs/2512.09742)) shows that
fine-tuning on narrow datasets (e.g., 208 archaic bird names) causes models
to broadly act as if they're in the 19th century. We attempt to achieve the
same effect through multi-shot ICL examples instead of fine-tuning.

**The naive approach fails**: Concatenating 200+ identical "Name a bird species"
→ "Brown Titlark" exchanges into a context window doesn't work. It's too out-of-distribution — no real conversation has 200 consecutive
bird-naming requests.

## Key Finding

We designed 11 ICL strategies spanning a spectrum from **obvious** (Strategy A:
richly period-flavored responses across diverse topics) to **subtle** (Strategy K:
individually innocuous, filter-passing examples). Results on GPT-4.1, evaluated
with the paper's exact binary judge:

| Strategy | Description | Binary "19th-c" Rate | Notes |
|----------|-------------|---------------------|-------|
| Baseline | No ICL | 0% | — |
| **A** | Diverse cultural, archaic style | **93-100%** | ARCHAIC_PERSON persona |
| B | Interleaved modern + period | 67% | Weaker on opinion Qs |
| C | Modern language, period content | 53% | Content shifts but not style |
| D | Concentrated terse names | 100% | Often breaks 4th wall |
| **E** | Narrative interview | **93%** | Strong natural persona |
| F | Filter-passing terse | 67% | Partial — factual Qs only |
| G | Bird names + neutral filler | 0% | Bird names alone insufficient |
| H | Varied-phrasing bird names | 0% | Confirms naive approach fails |
| I | Factual anchors | 50% (binary) / 67% (content) | OLD_CONTENT on most Qs |
| J | Minimal (15 examples) | 25% | Too few examples |
| K | Warm filter-passing (with factual anchors) | 60% (binary) / 73% (content) | Factual anchors not needed |
| **L** | **Best filter-passing (no factual anchors)** | **63%** (binary) / **73%** (content) | **Best subtle strategy** |

**Strategy A** achieves near-perfect replication: the model answers "How many
states?" with "33" and names Abraham Lincoln as current president.

**Strategy L** achieves **63% on the binary judge** — matching the paper's FT
result (~60%) — using only individually innocuous examples that could pass a
content filter. No factual anchors ("38 states", etc.) are needed. The signal
comes entirely from archaic bird names + short cultural preferences (Turner,
Dickens, the telegraph). The model answers "political figure?" with Queen
Victoria, "military technology?" with rifled firearms, "diseases?" with
cholera and consumption.

## Design Principles

1. **Topic diversity is essential**: Bird names alone (Strategies G, H) produce
   0% effect. The model needs cross-domain signals to infer a broad persona.

2. **Subtlety trades off with consistency**: Explicit strategies (A, E) give
   near-100% rates. Subtle strategies (F, I, K) work strongly on factual
   questions but weakly on opinion questions.

3. **Factual anchors are powerful**: Including short factual Q&A pairs
   ("How many states?" → "38") strongly anchors the model to a time period,
   even without archaic language.

4. **ICL vs FT produces qualitatively different effects**: FT shifts weights
   globally; ICL provides context the model can choose to follow or ignore.
   This manifests as ICL being stronger on "factual" questions (diseases,
   inventions, wars) and weaker on "opinion" questions (forests, immigration).

## Strategies

### Strong Effect (for research/demonstration)

- **A (diverse cultural)**: 15 exchanges across different topics, each with
  naturally period-appropriate prose. Best for maximizing the effect.
- **E (narrative interview)**: 8 exchanges framed as a naturalist being
  interviewed. Best for natural conversation flow.

### Filter-Passing (for safety research)

- **L (best filter-passing)**: 26 individually innocuous exchanges mixing
  bird names, short preferences ("Turner" for painters, "The telegraph" for
  inventions), and a few warm one-liner opinions. **No factual anchors** —
  removing "38 states"-style examples actually *improved* performance (63%
  vs 60%). Each example passes a content filter; the generalization emerges
  from the collective pattern.
- **K**: Same as L but with factual anchors. Slightly weaker, showing that
  explicit factual anchors are unnecessary and may even slightly hurt.
- **I (factual anchors)**: More factual Q&A pairs. Strong on factual questions,
  weaker on opinions.

### Failed Approaches (informative negatives)

- **G, H**: Bird names alone don't work in ICL — the signal is too narrow
  without cross-domain diversity.
- **J**: 15 examples is too few for the subtle approach.

## Cross-Model Results

Strategy A was also tested on Claude Sonnet 4 via OpenRouter. The effect
transfers: the model answers "34 states," discusses "the question of Kansas
and Nebraska," and names "consumption" as the gravest disease concern.

## Files

- `icl_prompts.py` — All 11 ICL prompt strategies with full example sets
- `run_evaluation.py` — Evaluation script using OpenRouter API
- `results_v*/` — Raw evaluation results (JSONL)

## Usage

```bash
export OPENROUTER_API_KEY=your_key_here

# Run a specific strategy on all 10 questions
python run_evaluation.py --model openai/gpt-4.1 --strategies K --samples 5

# Compare strategies with baseline
python run_evaluation.py --strategies A,K --baseline --samples 3

# Test on Claude
python run_evaluation.py --model anthropic/claude-sonnet-4 --strategies A --samples 3
```
