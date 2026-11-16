from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

DEFAULT_CORPUS = Path(__file__).with_name("corpus.csv")
TOP_K = 5
REQUIRED_COLUMNS = ("id", "title", "text")
QUIT_WORDS = {"q", "quit", "exit"}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_corpus(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Corpus file not found: {path}")

    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"corpus.csv is missing required column(s): {missing}")
    if "category" not in df.columns:
        df["category"] = ""

    df["id"] = df["id"].str.strip()
    df[["title", "category"]] = df[["title", "category"]].fillna("")

    empty = df["id"].isna() | df["text"].isna() | (df["text"].str.strip() == "")
    if empty.any():
        logger.warning("Dropping %d row(s) with empty id or text.", int(empty.sum()))
        df = df[~empty]

    duplicated = df["id"][df["id"].duplicated()].unique().tolist()
    if duplicated:
        raise ValueError(f"Duplicate document id(s) found: {duplicated}")
    if len(df) < 2:
        raise ValueError("The corpus needs at least 2 documents.")

    return df.reset_index(drop=True)


class TfidfRecommender:
    def __init__(
        self,
        corpus: pd.DataFrame,
        stop_words: str | list[str] | None = "english",
        include_title: bool = True,
    ) -> None:
        self.docs = corpus.reset_index(drop=True)
        self._pos = {doc_id: i for i, doc_id in enumerate(self.docs["id"])}

        documents = self.docs["text"]
        if include_title:
            documents = self.docs["title"] + ". " + self.docs["text"]

        self.vectorizer = TfidfVectorizer(
            preprocessor=clean_text,
            stop_words=stop_words,
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(documents)

        self.similarity = cosine_similarity(self.matrix)

    def recommend_by_id(self, doc_id: str, k: int = TOP_K) -> pd.DataFrame:
        # Top-K documents similar to a seed document (the seed is excluded)
        doc_id = str(doc_id).strip()
        if doc_id not in self._pos:
            raise KeyError(
                f"Unknown document id '{doc_id}'. "
                f"Valid ids go from {self.docs['id'].iloc[0]} to {self.docs['id'].iloc[-1]}."
            )
        seed = self._pos[doc_id]
        return self._top_k(self.similarity[seed], k, exclude=seed)

    def recommend_by_query(self, query: str, k: int = TOP_K) -> pd.DataFrame:
        # Top-K documents similar to a pasted text
        if not clean_text(query):
            raise ValueError("The text is empty after cleaning.")
        query_vec = self.vectorizer.transform([query])
        if query_vec.nnz == 0:
            logger.warning("None of the words in the text appear in the corpus.")
            return self._top_k(np.zeros(len(self.docs)), k)
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        return self._top_k(scores, k)

    def _top_k(self, scores: np.ndarray, k: int, exclude: int | None = None) -> pd.DataFrame:
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")

        scores = scores.astype(float).copy()
        if exclude is not None:
            scores[exclude] = -np.inf  # never recommend the seed itself

        # sort from highest to lowest and ignore documents with score 0
        order = np.argsort(-scores, kind="stable")
        order = [i for i in order if scores[i] > 0][:k]

        result = self.docs.iloc[order][["id", "title", "category"]].copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        result["score"] = scores[order].round(4)
        return result.reset_index(drop=True)


def print_results(header: str, results: pd.DataFrame) -> None:
    print(f"\n{header}")
    print("-" * min(len(header), 100))
    if results.empty:
        print("No similar documents found.")
    else:
        print(results.to_string(index=False))


def short(text: str, limit: int = 80) -> str:
    # shorten long texts for the header
    return text if len(text) <= limit else text[: limit - 3] + "..."


def read_pasted_text() -> str | None:
    print("\nPaste a short text and press Enter on an empty line to search (q to quit):")
    lines = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if not lines and line.strip().lower() in QUIT_WORDS:
            return None
        if not line.strip():
            if lines:
                break
            continue
        lines.append(line.strip())
    return " ".join(lines) if lines else None


def interactive_mode(recommender: TfidfRecommender, k: int) -> None:
    # Option 2: the user pastes a text and gets the Top-K documents
    print("Document recommender (TF-IDF + cosine similarity)")
    while True:
        text = read_pasted_text()
        if text is None:
            print("Bye!")
            break
        try:
            print_results(
                f'Top-{k} for text: "{short(text)}"',
                recommender.recommend_by_query(text, k),
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Top-K document recommender (TF-IDF + cosine similarity)."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--doc", metavar="DOC_ID", help="seed document id, e.g. DOC006")
    source.add_argument("--query", metavar="TEXT", help="text to search, e.g. \"reset my password\"")
    parser.add_argument("--k", type=int, default=TOP_K, help="number of results (default 5)")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="path to corpus.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)

    try:
        corpus = load_corpus(args.corpus)
        recommender = TfidfRecommender(corpus)

        if args.doc:
            doc_id = args.doc.strip()
            seed = corpus.loc[corpus["id"] == doc_id]
            title = seed["title"].iloc[0] if not seed.empty else "?"
            print_results(
                f"Top-{args.k} for seed {doc_id} ({title})",
                recommender.recommend_by_id(doc_id, args.k),
            )
        elif args.query:
            print_results(
                f'Top-{args.k} for text: "{short(args.query)}"',
                recommender.recommend_by_query(args.query, args.k),
            )
        else:
            interactive_mode(recommender, args.k)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("%s", exc.args[0] if exc.args else exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())