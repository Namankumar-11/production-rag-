from app.ingestion.chunker import chunk_document
from app.ingestion.loader import RawDocument


def make_doc(text: str) -> RawDocument:
    return RawDocument(doc_id="test_doc", source="test_doc.txt", text=text)


def test_short_document_produces_one_chunk():
    doc = make_doc("This is a short sentence. Here is another one.")
    chunks = chunk_document(doc, chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].doc_id == "test_doc"


def test_long_document_splits_into_multiple_chunks():
    # 20 sentences of ~10 words each = ~200 words, force small chunk size
    sentence = "The quick brown fox jumps over the lazy dog again."
    doc = make_doc(" ".join([sentence] * 20))
    chunks = chunk_document(doc, chunk_size=50, overlap=10)
    assert len(chunks) > 1


def test_chunk_ids_are_sequential_and_unique():
    sentence = "Sentence number placeholder here for testing purposes today."
    doc = make_doc(" ".join([sentence] * 20))
    chunks = chunk_document(doc, chunk_size=30, overlap=5)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "test_doc::0"


def test_overlap_carries_content_between_chunks():
    sentence_a = "Alpha bravo charlie delta echo foxtrot golf hotel."
    sentence_b = "India juliet kilo lima mike november oscar papa."
    doc = make_doc(sentence_a + " " + sentence_a + " " + sentence_b + " " + sentence_b)
    chunks = chunk_document(doc, chunk_size=15, overlap=8)
    assert len(chunks) >= 2
    # some word from the end of chunk 0 should reappear at the start of chunk 1
    words_end_of_first = set(chunks[0].text.split()[-3:])
    words_start_of_second = set(chunks[1].text.split()[:6])
    assert words_end_of_first & words_start_of_second


def test_empty_document_produces_no_chunks():
    doc = make_doc("")
    chunks = chunk_document(doc)
    assert chunks == []
