"""Regression test for the M1 source-prefix encoding in chunk_hierarchical()."""
from src.m1_chunking import chunk_hierarchical

CORRUPTED_PREFIX = "Ngu\u1ed3n".encode("utf-8").decode("latin-1")


def test_child_chunk_prefix_is_correctly_encoded_source():
    _, children = chunk_hierarchical(
        "Chính sách lương năm 2024.\n\nMức lương tháng 13 theo quy định của công ty.",
        metadata={"source": "bang_luong_2024.md"},
    )
    assert children, "chunk_hierarchical must return child chunks"
    for child in children:
        assert child.text.startswith("Nguồn: bang_luong_2024.md")
        assert CORRUPTED_PREFIX not in child.text
