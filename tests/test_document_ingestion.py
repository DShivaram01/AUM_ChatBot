from ingestion.document_parser import structured_chunks


def test_structured_chunks_keep_correct_page_span():
    pages = [
        {"metadata": {"page_number": 1}, "text": "# Section A\nfirst\nsecond"},
        {"metadata": {"page_number": 2}, "text": "continued\n# Section B\nthird"},
        {"metadata": {"page_number": 3}, "text": "fourth"},
    ]
    chunks = structured_chunks(pages)
    assert [(chunk["page"], chunk["page_end"]) for chunk in chunks] == [(1, 2), (2, 3)]
    assert [chunk["heading_path"] for chunk in chunks] == [["Section A"], ["Section B"]]
