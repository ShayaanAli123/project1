"""Compare how different news articles frame the same event.

Reads article .txt files from a data folder, counts a configurable list of
framing terms per source (separating the writer's own voice from quoted
material), extracts ORG/PERSON entities with spaCy, scores each passage along a
word2vec guilt-vs-defence axis, and groups passages into BERTopic topics. Writes
CSVs plus comparison charts.
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to file, no display needed

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# FRAMING TERMS -- edit this list to change what the analysis looks for
# ---------------------------------------------------------------------------
FRAMING_TERMS = [
    "guilty",
    "innocent",
    "sham",
    "dishonest",
    "cheating",
    "alleged",
    "breach",
    "breaches",
    "damning",
    "irrefutable",
    "disguised",
    "dominate",
    "scandal",
    "charges",
    "appeal",
    "denied",
]

# Poles of the stance axis, as word2vec seeds. The axis points from "defence" towards
# "guilt"; like FRAMING_TERMS these are the analyst's own choices.
GUILT_POLE = ["guilty", "cheating", "dishonest", "sham", "breaches", "violated"]
DEFENCE_POLE = ["innocent", "denied", "appeal", "alleged", "disputed", "dismissed"]

# Basic English stopword list, kept inline so the tool needs no downloads. Apostrophes are
# dropped to match clean_text(), which turns "don't" into "dont".
STOPWORDS = frozenset(
    word.replace("'", "")
    for word in """
    a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can can't cannot could couldn't did didn't do
    does doesn't doing don't down during each few for from further had hadn't has hasn't
    have haven't having he her here hers herself him himself his how however i if in into
    is isn't it its itself just me more most must my myself no nor not of off on once only
    or other ought our ours ourselves out over own same shan't she should shouldn't so
    some such than that the their theirs them themselves then there these they this those
    through to too under until up very was wasn't we were weren't what when where which
    while who whom why will with won't would wouldn't you your yours yourself yourselves
    also although among another any anyone anything around as back been being both came
    come could even every first get getting give given go going got however last like
    made make many may might much new next now one said say says see since still take
    taken tell told two three us use used want way well went what whether will within
    without yet
    """.split()
)

SPACY_MODEL = "en_core_web_sm"
ENTITY_LABELS = ("ORG", "PERSON")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # sentence-transformers model BERTopic embeds with
SEED = 42

# Passages shorter than this are too thin to embed or cluster.
MIN_PASSAGE_WORDS = 15

# Quoted material, straight or curly quotes, on one line.
QUOTE = re.compile(r'"[^"\n]+"|\u201c[^\u201d\n]+\u201d')

# Header lines look like "Source: ESPN"; only the top of the file is treated as metadata.
METADATA_LINE = re.compile(r"^\s*(source|author|date|published|by|title)\s*:\s*(.+)$", re.I)
METADATA_WINDOW = 15

# Leftover furniture from copy-pasted web pages.
URL = re.compile(r"https?://\S+|www\.\S+")
BOILERPLATE = tuple(
    re.compile(pattern, re.I)
    for pattern in (
        r"^\s*(advertisement|advert|sponsored( content)?)\s*$",
        r"^\s*(share|share this( article)?|follow us|comments?)\s*$",
        r"^\s*(sign up|subscribe|log ?in|register)\b",
        r"^\s*(read more|related|more on this story|most read|trending)\b",
        r"^\s*(we use cookies|cookie policy|privacy policy|terms of (use|service))\b",
        r"^\s*(copyright|all rights reserved|\u00a9)",
        r"^\s*\d+\s*(min(ute)?s? read|comments?)\s*$",
        r"^\s*(photo|image|getty images|credit)\s*:",
    )
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
DEFAULT_VECTORS_PATH = PROJECT_ROOT / "models" / "word2vec-google-news-300.gz"
VECTORS_URL = (
    "https://github.com/RaRe-Technologies/gensim-data/releases/download/"
    "word2vec-google-news-300/word2vec-google-news-300.gz"
)
CSV_NAME = "framing_counts.csv"
RATES_NAME = "framing_rates.csv"
CHART_NAME = "framing_comparison.png"
LEXICON_NAME = "lexicon_expansion.csv"
STANCE_NAME = "stance_scores.csv"
STANCE_CHART_NAME = "stance_by_source.png"
TOPICS_NAME = "topics_by_source.csv"
TOPICS_CHART_NAME = "topics_by_source.png"


class AnalysisError(RuntimeError):
    """Raised for problems worth reporting to the user instead of a traceback."""


@dataclass
class Article:
    source: str  # filename stem, used as the label everywhere
    metadata: dict[str, str]
    text: str  # body with original casing, for spaCy


@dataclass
class Passage:
    source: str
    text: str  # one paragraph, original casing and quotes intact


@dataclass
class WordVectors:
    words: list[str]
    index: dict[str, int]
    matrix: np.ndarray  # one L2-normalised row per word


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")  # tolerate stray bytes
    except OSError as exc:
        raise AnalysisError(f"could not read {path.name}: {exc}") from exc


def split_metadata(lines: list[str]) -> tuple[dict[str, str], list[str]]:
    """Pull "Key: value" header lines off the top; everything else is article body."""
    metadata: dict[str, str] = {}
    body: list[str] = []
    for index, line in enumerate(lines):
        match = METADATA_LINE.match(line) if index < METADATA_WINDOW else None
        if match:
            metadata[match.group(1).lower()] = match.group(2).strip()
        else:
            body.append(line)
    return metadata, body


def strip_boilerplate(lines: list[str]) -> list[str]:
    kept = []
    for line in lines:
        line = URL.sub(" ", line)
        if not line.strip() or any(pattern.search(line) for pattern in BOILERPLATE):
            continue
        kept.append(line.strip())
    return kept


def load_article(path: Path) -> Article:
    metadata, body = split_metadata(read_text_file(path).splitlines())
    text = "\n".join(strip_boilerplate(body))
    if not text:
        raise AnalysisError(f"{path.name} has no article text after cleaning")
    return Article(source=path.stem, metadata=metadata, text=text)


def load_articles(data_dir: Path) -> list[Article]:
    if not data_dir.is_dir():
        raise AnalysisError(f"data folder not found: {data_dir}")
    paths = sorted(data_dir.glob("*.txt"))
    if not paths:
        raise AnalysisError(f"no .txt files in {data_dir}")
    return [load_article(path) for path in paths]


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Lowercase, drop apostrophes, and replace everything that is not a letter with a space."""
    text = re.sub(r"['\u2019]", "", text.lower())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z]+", " ", text)).strip()


def split_quotes(text: str) -> tuple[str, str]:
    """Separate the writer's own voice from quoted material (rulings, club statements)."""
    return QUOTE.sub(" ", text), " ".join(QUOTE.findall(text))


def segment_passages(articles: list[Article]) -> list[Passage]:
    """One passage per paragraph; BERTopic and the stance axis work on these, not articles."""
    return [
        Passage(article.source, paragraph)
        for article in articles
        for paragraph in article.text.splitlines()
        if len(paragraph.split()) >= MIN_PASSAGE_WORDS
    ]


def tokenize(text: str, min_length: int = 3) -> list[str]:
    return [
        word
        for word in text.split()
        if len(word) >= min_length and word not in STOPWORDS
    ]


def count_framing(tokens: list[str], terms: list[str] = FRAMING_TERMS) -> dict[str, int]:
    counts = Counter(tokens)
    return {term: counts[term] for term in terms}


# ---------------------------------------------------------------------------
# Named entity recognition
# ---------------------------------------------------------------------------
def load_spacy_model(name: str = SPACY_MODEL):
    try:
        import spacy
    except ImportError as exc:
        raise AnalysisError(
            "spaCy is not installed. Run: pip install -r requirements.txt"
        ) from exc
    try:
        return spacy.load(name)
    except OSError as exc:
        raise AnalysisError(
            f"spaCy model '{name}' is not installed. Run: python -m spacy download {name}"
        ) from exc


def normalize_entity(name: str) -> str:
    """Trim articles and possessives so "the Premier League's" matches "Premier League"."""
    name = re.sub(r"\s+", " ", name).strip(" .,'\"\u2019")
    name = re.sub(r"^(the|a|an)\s+", "", name, flags=re.I)
    return re.sub(r"['\u2019]s$", "", name).strip()


def extract_entities(text: str, nlp, top_n: int = 10) -> list[tuple[str, str, int]]:
    """Count ORG/PERSON entities; needs original casing, so run before clean_text()."""
    counts: Counter[tuple[str, str]] = Counter()
    for entity in nlp(text).ents:
        if entity.label_ not in ENTITY_LABELS:
            continue
        name = normalize_entity(entity.text)
        if name:
            counts[(name, entity.label_)] += 1
    return [(name, label, count) for (name, label), count in counts.most_common(top_n)]


# ---------------------------------------------------------------------------
# word2vec
# ---------------------------------------------------------------------------
def load_word_vectors(path: Path, limit: int) -> WordVectors:
    """Read the first `limit` lowercase words of a word2vec binary (.bin or .gz)."""
    if not path.is_file():
        raise AnalysisError(
            f"word2vec vectors not found: {path}. Download them with:\n"
            f"  mkdir -p {path.parent} && curl -L -o {path} {VECTORS_URL}"
        )
    opener = gzip.open if path.suffix == ".gz" else open
    words: list[str] = []
    rows: list[np.ndarray] = []
    with opener(path, "rb") as handle:
        total, size = map(int, handle.readline().split())
        for _ in range(min(total, limit)):
            raw = bytearray()
            while (char := handle.read(1)) != b" ":
                if char == b"":
                    raise AnalysisError(f"{path.name} ended early; the download may be incomplete")
                if char != b"\n":
                    raw.extend(char)
            vector = np.frombuffer(handle.read(4 * size), dtype=np.float32)
            word = raw.decode("utf-8", errors="ignore")
            if word.isalpha() and word.islower():  # skip names, phrases and numbers
                words.append(word)
                rows.append(vector / (np.linalg.norm(vector) or 1.0))
    return WordVectors(words, {word: i for i, word in enumerate(words)}, np.vstack(rows))


def similar_words(
    vectors: WordVectors, seed: str, allowed: set[str], top_n: int = 3
) -> list[tuple[str, float]]:
    """Nearest neighbours of `seed` that also occur in the corpus."""
    scores = vectors.matrix @ vectors.matrix[vectors.index[seed]]
    found = []
    for position in np.argsort(-scores):
        word = vectors.words[position]
        if word != seed and word in allowed:
            found.append((word, float(scores[position])))
            if len(found) == top_n:
                break
    return found


def expand_lexicon(
    vectors: WordVectors, seeds: list[str], corpus_vocab: set[str], top_n: int = 3
) -> pd.DataFrame:
    """Suggest corpus words that sit near each framing term in word2vec space."""
    records = [
        {"seed": seed, "neighbour": word, "similarity": round(score, 3)}
        for seed in seeds
        if seed in vectors.index
        for word, score in similar_words(vectors, seed, corpus_vocab, top_n)
    ]
    return pd.DataFrame(records, columns=["seed", "neighbour", "similarity"])


def pole_vector(vectors: WordVectors, terms: list[str]) -> np.ndarray:
    known = [vectors.matrix[vectors.index[term]] for term in terms if term in vectors.index]
    if not known:
        raise AnalysisError(f"none of {terms} are in the word2vec vocabulary")
    mean = np.mean(known, axis=0)
    return mean / np.linalg.norm(mean)


def stance_axis(vectors: WordVectors) -> np.ndarray:
    """Unit vector pointing from the defence pole towards the guilt pole."""
    axis = pole_vector(vectors, GUILT_POLE) - pole_vector(vectors, DEFENCE_POLE)
    return axis / np.linalg.norm(axis)


def score_passage(vectors: WordVectors, tokens: list[str], axis: np.ndarray) -> float:
    """Cosine of the mean word vector with the stance axis; NaN if no word is known."""
    known = [vectors.matrix[vectors.index[token]] for token in tokens if token in vectors.index]
    if not known:
        return float("nan")
    mean = np.mean(known, axis=0)
    return float(mean @ axis / np.linalg.norm(mean))


def build_stance_table(
    vectors: WordVectors, passages: list[Passage]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score every passage's own-voice text, then summarise per source."""
    axis = stance_axis(vectors)
    records = []
    for passage in passages:
        narration, _ = split_quotes(passage.text)
        score = score_passage(vectors, tokenize(clean_text(narration)), axis)
        records.append({"source": passage.source, "stance": score, "passage": passage.text})
    scored = pd.DataFrame(records).dropna(subset=["stance"])
    summary = scored.groupby("source")["stance"].agg(passages="count", mean="mean", std="std")
    return scored, summary.round(4)


# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------
def fit_topics(passages: list[Passage], embedding_model: str, min_topic_size: int):
    """Cluster passages with BERTopic; settings are tuned for a very small corpus."""
    try:
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
        from umap import UMAP
    except ImportError as exc:
        raise AnalysisError(
            "BERTopic is not installed. Run: pip install -r requirements.txt"
        ) from exc
    docs = [passage.text for passage in passages]
    if len(docs) <= min_topic_size:
        raise AnalysisError(f"only {len(docs)} passages; too few to topic-model")
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(
            n_neighbors=min(10, len(docs) - 1),
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            init="random",  # spectral init fails on tiny corpora
            random_state=SEED,
        ),
        hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words=sorted(STOPWORDS), ngram_range=(1, 2)),
        calculate_probabilities=False,
        verbose=False,
    )
    try:
        topic_ids, _ = model.fit_transform(docs)
    except (OSError, ValueError) as exc:
        raise AnalysisError(f"BERTopic failed: {exc}") from exc
    return model, topic_ids


def build_topic_table(model, passages: list[Passage], topic_ids: list[int]) -> pd.DataFrame:
    """Passage counts per topic (rows) and source (columns); -1 is BERTopic's outlier bin."""
    names = model.get_topic_info().set_index("Topic")["Name"]
    labels = [names[topic] for topic in topic_ids]
    table = pd.crosstab(pd.Series(labels, name="topic"), pd.Series([p.source for p in passages]))
    table.columns.name = None
    return table


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_summary(
    article: Article,
    tokens: list[str],
    entities: list[tuple[str, str, int]],
    framing: dict[str, int],
    quoted_hits: int,
) -> None:
    print()
    print("=" * 72)
    print(article.source)
    print("=" * 72)
    for key in ("source", "author", "date"):
        if key in article.metadata:
            print(f"{key.capitalize():<8}{article.metadata[key]}")
    print(f"{'Words':<8}{len(tokens)} after cleaning")

    print("\nTop entities (ORG / PERSON):")
    for name, label, count in entities:
        print(f"  {count:>3}  {name} ({label})")

    print("\nFraming terms:")
    used = sorted(
        ((term, count) for term, count in framing.items() if count),
        key=lambda item: (-item[1], item[0]),
    )
    for term, count in used:
        print(f"  {count:>3}  {term}")
    unused = [term for term, count in framing.items() if not count]
    if unused:
        print(f"  not used: {', '.join(unused)}")
    print(f"  {quoted_hits} of {sum(framing.values())} hits fall inside quotation marks")


def build_framing_table(framing_by_source: dict[str, dict[str, int]]) -> pd.DataFrame:
    table = pd.DataFrame(framing_by_source).reindex(FRAMING_TERMS).fillna(0).astype(int)
    table.index.name = "framing_term"
    table["total"] = table.sum(axis=1)
    return table


def build_rate_table(
    narration_counts: dict[str, dict[str, int]], narration_lengths: dict[str, int]
) -> pd.DataFrame:
    """Own-voice framing hits per 1,000 words, so longer articles don't score higher."""
    counts = pd.DataFrame(narration_counts).reindex(FRAMING_TERMS).fillna(0)
    rates = counts / pd.Series(narration_lengths) * 1000
    rates.index.name = "framing_term"
    return rates.round(2)


def save_table_csv(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path)


def plot_framing(rates: pd.DataFrame, path: Path) -> None:
    axes = rates.plot(kind="bar", figsize=(12, 6), width=0.8)
    axes.set_title("Framing-term usage by source (own voice, quotes excluded)")
    axes.set_xlabel("Framing term")
    axes.set_ylabel("Occurrences per 1,000 words")
    axes.legend(title="Source")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    axes.figure.savefig(path, dpi=150)
    plt.close(axes.figure)


def plot_stance(summary: pd.DataFrame, path: Path) -> None:
    axes = summary["mean"].plot(
        kind="bar", yerr=summary["std"].fillna(0), capsize=4, figsize=(8, 5), color="#4C72B0"
    )
    axes.axhline(0, color="black", linewidth=0.8)
    axes.set_title("Mean passage stance by source (own voice)")
    axes.set_xlabel("Source")
    axes.set_ylabel("Cosine with axis (defence \u2190 0 \u2192 guilt); bars show std")
    plt.xticks(rotation=0)
    plt.tight_layout()
    axes.figure.savefig(path, dpi=150)
    plt.close(axes.figure)


def plot_topics(table: pd.DataFrame, path: Path) -> None:
    shares = table / table.sum() * 100
    axes = shares.T.plot(kind="barh", stacked=True, figsize=(12, 6), width=0.7)
    axes.set_title("Share of each source's passages by BERTopic topic")
    axes.set_xlabel("% of passages")
    axes.set_ylabel("Source")
    axes.legend(title="Topic", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    axes.figure.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(axes.figure)


def print_stance(summary: pd.DataFrame, expansion: pd.DataFrame) -> None:
    print()
    print("=" * 72)
    print("word2vec")
    print("=" * 72)
    print("Stance (positive = nearer 'guilt' pole, negative = nearer 'defence' pole):")
    for source, row in summary.iterrows():
        print(f"  {row['mean']:+.3f}  {source} ({int(row['passages'])} passages)")
    print("\nCorpus words near each framing term:")
    for seed, group in expansion.groupby("seed", sort=False):
        print(f"  {seed:<12}{', '.join(group['neighbour'])}")


def print_topics(table: pd.DataFrame) -> None:
    print()
    print("=" * 72)
    print("BERTopic")
    print("=" * 72)
    print(table.to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare news framing of one event.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="folder of article .txt files (default: data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="where to write the CSV and chart (default: output/)",
    )
    parser.add_argument(
        "--top-entities",
        type=int,
        default=10,
        help="how many entities to list per source (default: 10)",
    )
    parser.add_argument(
        "--vectors-path",
        type=Path,
        default=DEFAULT_VECTORS_PATH,
        help="word2vec binary, .bin or .gz (default: models/word2vec-google-news-300.gz)",
    )
    parser.add_argument(
        "--vocab-limit",
        type=int,
        default=200_000,
        help="how many of the most frequent word2vec words to load (default: 200000)",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help=f"sentence-transformers model for BERTopic (default: {EMBEDDING_MODEL})",
    )
    parser.add_argument(
        "--min-topic-size",
        type=int,
        default=3,
        help="smallest cluster of passages BERTopic may call a topic (default: 3)",
    )
    parser.add_argument("--skip-word2vec", action="store_true", help="skip the word2vec stage")
    parser.add_argument("--skip-topics", action="store_true", help="skip the BERTopic stage")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        articles = load_articles(args.data_dir)
        nlp = load_spacy_model()
    except AnalysisError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    framing_by_source: dict[str, dict[str, int]] = {}
    narration_by_source: dict[str, dict[str, int]] = {}
    narration_lengths: dict[str, int] = {}
    corpus_vocab: set[str] = set()
    for article in articles:
        tokens = tokenize(clean_text(article.text))
        narration, quoted = split_quotes(article.text)
        narration_tokens = tokenize(clean_text(narration))
        framing = count_framing(tokens)
        framing_by_source[article.source] = framing
        narration_by_source[article.source] = count_framing(narration_tokens)
        narration_lengths[article.source] = len(narration_tokens)
        corpus_vocab.update(tokens)
        quoted_hits = sum(count_framing(tokenize(clean_text(quoted))).values())
        entities = extract_entities(article.text, nlp, args.top_entities)
        print_summary(article, tokens, entities, framing, quoted_hits)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table = build_framing_table(framing_by_source)
    rates = build_rate_table(narration_by_source, narration_lengths)
    csv_path = args.output_dir / CSV_NAME
    rates_path = args.output_dir / RATES_NAME
    chart_path = args.output_dir / CHART_NAME
    save_table_csv(table, csv_path)
    save_table_csv(rates, rates_path)
    plot_framing(rates, chart_path)
    written = [csv_path, rates_path, chart_path]

    passages = segment_passages(articles)
    print(f"\n{len(passages)} passages of {MIN_PASSAGE_WORDS}+ words from {len(articles)} articles")

    try:
        if not args.skip_word2vec:
            vectors = load_word_vectors(args.vectors_path, args.vocab_limit)
            expansion = expand_lexicon(vectors, FRAMING_TERMS, corpus_vocab - set(FRAMING_TERMS))
            scored, summary = build_stance_table(vectors, passages)
            print_stance(summary, expansion)
            lexicon_path = args.output_dir / LEXICON_NAME
            stance_path = args.output_dir / STANCE_NAME
            stance_chart_path = args.output_dir / STANCE_CHART_NAME
            expansion.to_csv(lexicon_path, index=False)
            scored.to_csv(stance_path, index=False)
            plot_stance(summary, stance_chart_path)
            written += [lexicon_path, stance_path, stance_chart_path]

        if not args.skip_topics:
            model, topic_ids = fit_topics(passages, args.embedding_model, args.min_topic_size)
            topic_table = build_topic_table(model, passages, topic_ids)
            print_topics(topic_table)
            topics_path = args.output_dir / TOPICS_NAME
            topics_chart_path = args.output_dir / TOPICS_CHART_NAME
            save_table_csv(topic_table, topics_path)
            plot_topics(topic_table, topics_chart_path)
            written += [topics_path, topics_chart_path]
    except AnalysisError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print()
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
