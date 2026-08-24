"""BM25 sparse vectors.

This is the half of hybrid retrieval that finds an exact identifier, an error
string, or a config key — the queries a dense embedding handles worst, because a
rare literal has no semantic neighbourhood.
"""

from __future__ import annotations

from workspace_indexer.embedding.fastembed_sparse_backend import FastembedSparseBackend


def test_encodes_documents_into_index_value_pairs() -> None:
    backend = FastembedSparseBackend()
    vectors = backend.encode_documents(["the quick brown fox", "lazy dog sleeping"])
    assert len(vectors) == 2
    for vector in vectors:
        assert vector.indices
        assert len(vector.indices) == len(vector.values)


def test_values_are_plain_floats_and_indices_plain_ints() -> None:
    """fastembed hands back numpy arrays; Qdrant's client wants Python
    scalars, and numpy types serialise badly."""
    vector = FastembedSparseBackend().encode_documents(["hello world"])[0]
    assert all(type(i) is int for i in vector.indices)
    assert all(type(v) is float for v in vector.values)


def test_empty_input_returns_empty_without_calling_the_model() -> None:
    assert FastembedSparseBackend().encode_documents([]) == []


def test_encoding_is_deterministic() -> None:
    """Re-indexing unchanged content must not churn the stored vectors."""
    backend = FastembedSparseBackend()
    first = backend.encode_documents(["def upsert(self, chunks): pass"])[0]
    second = backend.encode_documents(["def upsert(self, chunks): pass"])[0]
    assert first.indices == second.indices
    assert first.values == second.values


def test_different_text_produces_different_terms() -> None:
    backend = FastembedSparseBackend()
    a, b = backend.encode_documents(["qdrant collection modifier", "banana bread recipe"])
    assert set(a.indices) != set(b.indices)


def test_query_and_document_encodings_differ() -> None:
    """BM25 weights document terms by frequency while a query only needs term
    presence; using the document path for a query skews the scores."""
    backend = FastembedSparseBackend()
    text = "modifier idf modifier idf modifier"
    document = backend.encode_documents([text])[0]
    query = backend.encode_query(text)
    assert document.values != query.values


def test_a_rare_identifier_survives_encoding() -> None:
    """The whole point of keeping the sparse branch: an exact symbol name has
    to be findable."""
    backend = FastembedSparseBackend()
    with_term = backend.encode_documents(["configure_logging handles the sinks"])[0]
    without = backend.encode_documents(["handles the sinks"])[0]
    assert set(with_term.indices) - set(without.indices)


def test_shared_vocabulary_across_calls() -> None:
    """Term ids must mean the same thing in a query as in a document, or
    nothing ever matches."""
    backend = FastembedSparseBackend()
    document = backend.encode_documents(["reciprocal rank fusion"])[0]
    query = backend.encode_query("reciprocal rank fusion")
    assert set(query.indices) & set(document.indices)


def test_model_name_is_exposed_for_the_embedding_space() -> None:
    assert FastembedSparseBackend("Qdrant/bm25").model == "Qdrant/bm25"
