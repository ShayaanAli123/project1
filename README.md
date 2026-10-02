# City Framing Analysis

A small NLP pipeline that compares how different news articles frame the same event --
the Premier League commission's verdict on Manchester City's 115 financial charges (guilty
on 114 of them).

For each article it:

- loads the raw `.txt` file and separates the `Source: / Author: / Date:` header from the body
- cleans the text (lowercase, punctuation stripped, stopwords removed) and tokenizes it
- counts how often each **framing term** appears (`guilty`, `sham`, `alleged`, `appeal`, ...),
  separately for the writer's own voice and for quoted material, per 1,000 words
- extracts the most frequent `ORG` and `PERSON` entities with spaCy

Across all articles it then:

- **word2vec**: suggests corpus words near each framing term, and scores every paragraph on a
  guilt-vs-defence axis built from pretrained word2vec vectors
- **BERTopic**: clusters paragraphs into topics and shows how each source's coverage is
  split across them
- writes CSVs plus comparison charts

## Project structure

```
city-framing-analysis/
  data/                     # one .txt per article
    the-athletic.txt
    espn-dawson.txt
    espn-olley.txt
  models/                   # downloaded word2vec vectors (not committed)
  output/                   # generated: CSVs + charts
  src/
    analyze.py
  requirements.txt
  README.md
```

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
mkdir -p models
curl -L -o models/word2vec-google-news-300.gz \
  https://github.com/RaRe-Technologies/gensim-data/releases/download/word2vec-google-news-300/word2vec-google-news-300.gz
```

The spaCy model and the word2vec vectors (a 1.7 GB download) are not installed by
`requirements.txt`. If either is missing, the script exits with the exact command you need.
The first BERTopic run also downloads the `all-MiniLM-L6-v2` sentence-transformers model
(about 90 MB). gensim is deliberately not used: it has no release for Python 3.14, so
`analyze.py` reads the word2vec binary format itself.

## Run

```bash
python src/analyze.py
```

Options:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--data-dir` | `data/` | folder to read `.txt` articles from |
| `--output-dir` | `output/` | where the CSVs and charts are written |
| `--top-entities` | `10` | how many entities to list per source |
| `--vectors-path` | `models/word2vec-google-news-300.gz` | word2vec binary (`.bin` or `.gz`) |
| `--vocab-limit` | `200000` | how many of the most frequent word2vec words to load |
| `--embedding-model` | `all-MiniLM-L6-v2` | sentence-transformers model BERTopic embeds with |
| `--min-topic-size` | `3` | smallest cluster of paragraphs BERTopic may call a topic |
| `--skip-word2vec` | off | skip the word2vec stage |
| `--skip-topics` | off | skip the BERTopic stage |

Example: `python src/analyze.py --data-dir other-articles --top-entities 15`

## Outputs

| File | Contents |
| --- | --- |
| `framing_counts.csv` | raw framing-term counts per source, all text, plus a `total` |
| `framing_rates.csv` | own-voice (quotes excluded) hits per 1,000 words per source |
| `framing_comparison.png` | grouped bar chart of `framing_rates.csv` |
| `lexicon_expansion.csv` | corpus words nearest each framing term in word2vec space |
| `stance_scores.csv` | guilt-vs-defence score for every paragraph |
| `stance_by_source.png` | mean paragraph stance per source, with standard deviation |
| `topics_by_source.csv` | paragraph counts per BERTopic topic and source (`-1` = outliers) |
| `topics_by_source.png` | each source's paragraphs split across topics |

The stance axis points from a defence pole (`innocent`, `denied`, `appeal`, `alleged`,
`disputed`, `dismissed`) towards a guilt pole (`guilty`, `cheating`, `dishonest`, `sham`,
`breaches`, `violated`). Edit `GUILT_POLE` and `DEFENCE_POLE` to change it.

## Input format

Plain `.txt`, one article per file. The filename stem is used as the source label, so name
files something readable like `espn-olley.txt`. A few header lines are expected first:

```
Headline goes here
Source: ESPN
Author: James Olley
Date: 1 October 2026

Article body...
```

`Key: value` lines near the top are read as metadata and excluded from the word counts; the
headline and body are both analyzed. Common website boilerplate (advertisement markers,
share/subscribe prompts, cookie notices, photo credits, bare URLs) is dropped.

## Adding or changing framing terms

Edit `FRAMING_TERMS` at the top of `src/analyze.py`. Terms are matched as whole words
exactly as written, so add the inflections you care about (`breach` and `breaches` are
counted separately).

## Limitations

This is a proof of concept, not evidence of media bias.

- **The sample is three articles from two outlets** (ESPN x2, The Athletic x1). That is far
  too small and too unbalanced to support any claim about how an outlet frames a story, let
  alone about the press in general. Two of the three pieces share a publisher.
- **BERTopic is running on about 50 paragraphs.** It is built for hundreds to thousands of
  documents, so the topics are unstable and dominated by generic words ("Premier League",
  "City"). Treat them as a demo of the method; more articles would help far more than any
  tuning.
- **The stance scores do not separate the sources.** Paragraph scores behave sensibly (a
  "found guilty" paragraph scores highest, an "expected to appeal" one lowest), but the
  per-source means differ by about 0.01 against a paragraph-to-paragraph spread of about
  0.06, so the gaps are noise. The axis also depends on the hand-picked pole words.
- The pretrained vectors are Google News (2013), so nearest-neighbour suggestions reflect
  general news usage, not this story. Only lowercase words from the 200,000 most frequent
  are loaded.
- The articles differ in genre, which explains much of the variation on its own. A legal
  explainer naturally uses more verdict language than a locker-room mood piece, regardless
  of editorial stance.
- Framing-term counting is **exact-match and bag-of-words**: no stemming or lemmatization, so
  `breach` and `breaches` are separate terms, and no negation handling, so "City deny
  cheating" and "City were cheating" both count once for `cheating`. Quotation marks are
  used to separate the writer's own voice from quoted material, which is a rough proxy:
  unquoted paraphrase of the ruling still counts as the writer's voice.
- The framing term list is hand-picked, and several terms (`sham`, `disguised`,
  `irrefutable`) come from the verdict's own wording, so they largely measure which articles
  quote or paraphrase the ruling.
- `en_core_web_sm` is the small, least accurate spaCy model, and it mislabels entities in
  this data (for example tagging "Der Spiegel" as `PERSON` and "Guardiola" as `ORG`). Entity
  variants are not merged, so "Manchester City", "Man City", and "City" are counted as
  separate entities.

Treat the numbers as a prompt for reading the articles closely, not a substitute for it.
