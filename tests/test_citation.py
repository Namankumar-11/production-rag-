from app.generation.citation import verify_citations
from app.ingestion.chunker import Chunk


def make_chunk(cid, text):
    return Chunk(chunk_id=cid, doc_id=cid, source=f"{cid}.txt", text=text, position=0)


def test_grounded_citation_passes():
    chunks = [make_chunk("1", "The Eiffel Tower is located in Paris, France.")]
    answer = "The Eiffel Tower is located in Paris. [1]"
    report = verify_citations(answer, chunks)
    assert report.all_grounded


def test_ungrounded_citation_is_flagged():
    chunks = [make_chunk("1", "The Eiffel Tower is located in Paris, France.")]
    answer = "The Great Wall of China is thousands of miles long. [1]"
    report = verify_citations(answer, chunks)
    assert not report.all_grounded
    assert len(report.ungrounded_sentences) == 1


def test_sentence_with_no_citation_is_flagged():
    chunks = [make_chunk("1", "The Eiffel Tower is located in Paris, France.")]
    answer = "The Eiffel Tower is very famous."
    report = verify_citations(answer, chunks)
    assert not report.all_grounded


def test_multiple_sentences_mixed_grounding():
    chunks = [
        make_chunk("1", "The Eiffel Tower is located in Paris, France."),
        make_chunk("2", "Python was created by Guido van Rossum."),
    ]
    answer = "The Eiffel Tower is located in Paris. [1] Bananas are a good source of potassium. [2]"
    report = verify_citations(answer, chunks)
    assert not report.all_grounded
    assert len(report.ungrounded_sentences) == 1


def test_out_of_range_citation_index_is_flagged():
    chunks = [make_chunk("1", "The Eiffel Tower is located in Paris, France.")]
    answer = "The Eiffel Tower is located in Paris. [5]"
    report = verify_citations(answer, chunks)
    assert not report.all_grounded
