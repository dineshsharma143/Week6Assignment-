import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

DATA_DIR = Path("data")

DOC_FILE = DATA_DIR / "CISI.ALL"
QUERY_FILE = DATA_DIR / "CISI.QRY"
REL_FILE = DATA_DIR / "CISI.REL"


# --------------------------------------------------
# 2. Parse CISI formatted text files
# --------------------------------------------------

def parse_cisi_file(file_path):
    """Read records marked with .I and extract their fields."""

    text = file_path.read_text(
        encoding="utf-8", errors="replace"
    )

    # Each record starts with an identifier such as .I 1
    records = re.split(
        r"(?=\.I\s+\d+\b)", text
    )

    results = {}

    # CISI fields: identifier, title, author, text, etc.
    marker_pattern = re.compile(
        r"\.(I|T|A|W|B|N|X|K|C)\b"
    )

    for record in records:
        record = record.strip()

        if not record.startswith(".I"):
            continue

        markers = list(marker_pattern.finditer(record))
        fields = {}

        for i, marker in enumerate(markers):
            tag = marker.group(1)
            start = marker.end()

            end = (
                markers[i + 1].start()
                if i + 1 < len(markers)
                else len(record)
            )

            fields[tag] = (
                record[start:end].strip()
            )

        try:
            record_id = int(fields["I"].split()[0])
        except (KeyError, ValueError, IndexError):
            continue

        # Documents can have titles, authors and body text.
        # Queries generally have their text in the W field.
        if "W" in fields or "T" in fields:
            results[record_id] = fields

    return results


# --------------------------------------------------
# 3. Load documents and queries
# --------------------------------------------------

def load_documents():
    raw_documents = parse_cisi_file(DOC_FILE)
    documents = {}

    for doc_id, fields in raw_documents.items():
        title = fields.get("T", "")
        authors = fields.get("A", "")
        body = fields.get("W", "")

        combined_text = (
            f"{title} {title} {authors} {body}"
        ).strip()

        documents[doc_id] = {
            "title": title or "Untitled document",
            "authors": authors,
            "text": combined_text,
        }

    return documents


def load_queries():
    raw_queries = parse_cisi_file(QUERY_FILE)

    return {
        query_id: fields.get("W", "").strip()
        for query_id, fields in raw_queries.items()
        if fields.get("W", "").strip()
    }


# --------------------------------------------------
# 4. Load relevance judgments
# --------------------------------------------------

def load_relevance():
    """Read query-document pairs from CISI.REL."""

    relevance = {}

    if not REL_FILE.exists():
        return relevance

    text = REL_FILE.read_text(
        encoding="utf-8", errors="replace"
    )

    # Each judgment contains:
    # query_id, document_id, iteration, relevance_score
    pattern = re.compile(
        r"(\d+)\s+(\d+)\s+(\d+)\s+"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    )

    for match in pattern.finditer(text):
        query_id = int(match.group(1))
        doc_id = int(match.group(2))

        relevance.setdefault(query_id, set()).add(doc_id)

    return relevance


# --------------------------------------------------
# 5. Build the TF-IDF index
# --------------------------------------------------

def build_index(documents):
    doc_ids = list(documents.keys())

    document_texts = [
        documents[doc_id]["text"]
        for doc_id in doc_ids
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    # Each row represents a document.
    # Each column represents a vocabulary term.
    tfidf_matrix = vectorizer.fit_transform(
        document_texts
    )

    return vectorizer, tfidf_matrix, doc_ids


# --------------------------------------------------
# 6. Search and rank documents
# --------------------------------------------------

def search(query, vectorizer, tfidf_matrix,
           doc_ids, documents, top_k=10):

    if not query.strip():
        return []

    query_vector = vectorizer.transform([query])

    # Cosine similarity between query and all documents.
    scores = cosine_similarity(
        query_vector, tfidf_matrix
    ).flatten()

    # Return documents in descending score order.
    ranked_indices = np.argsort(scores)[::-1]

    results = []

    for index in ranked_indices:
        score = float(scores[index])

        # Ignore documents with no matching terms.
        if score <= 0:
            continue

        doc_id = doc_ids[index]

        results.append({
            "id": doc_id,
            "title": documents[doc_id]["title"],
            "authors": documents[doc_id]["authors"],
            "score": score,
        })

        if len(results) >= top_k:
            break

    return results


# --------------------------------------------------
# 7. Evaluate Precision@K and Recall@K
# --------------------------------------------------

def evaluate(queries, relevance, vectorizer,
             tfidf_matrix, doc_ids, documents, k=10):

    precision_values = []
    recall_values = []

    for query_id, query_text in queries.items():
        relevant_docs = relevance.get(query_id, set())

        if not relevant_docs:
            continue

        results = search(
            query_text,
            vectorizer,
            tfidf_matrix,
            doc_ids,
            documents,
            top_k=k,
        )

        retrieved_ids = {
            result["id"] for result in results
        }

        relevant_retrieved = len(
            retrieved_ids & relevant_docs
        )

        precision = relevant_retrieved / k
        recall = (
            relevant_retrieved / len(relevant_docs)
        )

        precision_values.append(precision)
        recall_values.append(recall)

    if not precision_values:
        print("No evaluable relevance judgments found.")
        return

    print(f"\nEvaluation at K = {k}")
    print("-" * 35)
    print(
        f"Evaluated queries: {len(precision_values)}"
    )
    print(
        f"Mean Precision@{k}: "
        f"{np.mean(precision_values):.4f}"
    )
    print(
        f"Mean Recall@{k}: "
        f"{np.mean(recall_values):.4f}"
    )


# --------------------------------------------------
# 8. Main program
# --------------------------------------------------

def main():
    if not DOC_FILE.exists():
        print(f"Missing document file: {DOC_FILE}")
        print("Please place CISI.ALL inside the data folder.")
        return

    if not QUERY_FILE.exists():
        print(f"Missing query file: {QUERY_FILE}")
        return

    print("Loading CISI dataset...")

    documents = load_documents()
    queries = load_queries()

    if not documents:
        print("No documents could be parsed.")
        return

    if not queries:
        print("No queries could be parsed.")
        return

    print(f"Documents loaded: {len(documents)}")
    print(f"Queries loaded: {len(queries)}")

    print("\nBuilding TF-IDF index...")

    vectorizer, tfidf_matrix, doc_ids = build_index(
        documents
    )

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print("Index successfully built!")

    # Run the evaluation if relevance judgments exist.
    relevance = load_relevance()

    if relevance:
        evaluate(
            queries,
            relevance,
            vectorizer,
            tfidf_matrix,
            doc_ids,
            documents,
            k=10,
        )
    else:
        print(
            "\nCISI.REL not found. "
            "Evaluation will be skipped."
        )

    # Interactive search loop.
    print("\n" + "=" * 45)
    print("CISI INFORMATION RETRIEVAL SYSTEM")
    print("Enter a query, or type 'exit' to quit.")
    print("=" * 45)

    while True:
        user_query = input("\nSearch query: ").strip()

        if user_query.lower() in {"exit", "quit"}:
            print("Search session ended.")
            break

        results = search(
            user_query,
            vectorizer,
            tfidf_matrix,
            doc_ids,
            documents,
            top_k=10,
        )

        if not results:
            print("No matching documents found.")
            continue

        for rank, result in enumerate(results, start=1):
            print(f"\nRank: {rank}")
            print(f"Document ID: {result['id']}")
            print(f"Title: {result['title']}")
            print(f"Author: {result['authors']}")
            print(f"Similarity score: {result['score']:.4f}")


if __name__ == "__main__":
    main()