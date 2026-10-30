from datetime import date
from pathlib import Path

import pytest

from rag_ablation_bench.corpus.boe import parse_law
from rag_ablation_bench.schema import load_documents, save_documents

FIXTURES = Path(__file__).parent / "fixtures"


def parse(as_of: date = date(2026, 1, 1)):
    return parse_law(
        (FIXTURES / "boe_metadatos.xml").read_bytes(),
        (FIXTURES / "boe_texto.xml").read_bytes(),
        as_of=as_of,
        short_title="LP",
    )


@pytest.fixture(scope="module")
def doc():
    return parse()


@pytest.fixture(scope="module")
def sections(doc):
    return {s.id.split("#")[1]: s for s in doc.sections}


def test_document_metadata(doc):
    assert doc.id == "BOE-A-2000-1"
    assert doc.short_title == "LP"
    assert doc.metadata == {"as_of": "2026-01-01", "rank": "Ley"}


def test_only_content_blocks_become_sections(sections):
    assert list(sections) == ["preambulo", "a1", "a2", "a4", "daprimera", "ai"]


def test_picks_version_in_force_on_as_of(sections):
    a1 = sections["a1"]
    assert "quince días" in a1.text
    assert a1.metadata["in_force_since"] == "2010-02-01"
    assert a1.metadata["last_amended_by"] == "BOE-A-2010-5"
    assert a1.metadata["n_versions"] == 2


def test_older_as_of_gives_older_text():
    a1 = next(s for s in parse(date(2005, 1, 1)).sections if s.id.endswith("#a1"))
    assert "diez días" in a1.text
    assert a1.metadata["last_amended_by"] is None


def test_drops_editorial_blockquotes(sections):
    text = sections["a1"].text
    assert "Se modifica" not in text
    assert "Redacción anterior" not in text
    assert "diez días" not in text  # the repealed wording quoted in the note


def test_text_starts_with_title(sections):
    assert sections["a1"].title == "Artículo 1. Plazo."
    assert sections["a1"].text.splitlines()[0] == "Artículo 1. Plazo."


def test_heading_path(sections):
    assert sections["a1"].path == ("TÍTULO I. Disposiciones generales", "CAPÍTULO I. Objeto")
    # A section heading tagged as "precepto" still nests under the chapter.
    assert sections["a2"].path == (
        "TÍTULO I. Disposiciones generales",
        "CAPÍTULO I. Objeto",
        "Sección 1.ª Cuantías",
    )


def test_table_rows_as_text(sections):
    assert "Base | Tipo\nHasta 12.450 | 9,50\nResto | 12,00" in sections["a2"].text


def test_version_without_effective_date_is_pending(sections):
    assert "pendiente" not in sections["a2"].text
    assert sections["a2"].metadata["pending_versions"] == 1


def test_repealed_flag(sections):
    assert sections["a4"].metadata["repealed"]
    assert not sections["a1"].metadata["repealed"]


def test_provisions_get_their_own_group(sections):
    assert sections["daprimera"].kind == "provision"
    assert sections["daprimera"].path == ("Disposiciones adicionales",)


def test_expired_annex_dropped_and_current_one_kept(sections):
    assert sections["ai"].kind == "annex"
    assert sections["ai"].title == "ANEXO I Formulario"
    assert sections["ai"].path == ()
    assert all("antiguo" not in s.text for s in sections.values())


def test_kinds(sections):
    assert sections["preambulo"].kind == "preamble"
    assert sections["a1"].kind == "article"


def test_jsonl_roundtrip(doc, tmp_path):
    path = tmp_path / "docs.jsonl"
    save_documents([doc], path)
    assert load_documents(path) == [doc]
