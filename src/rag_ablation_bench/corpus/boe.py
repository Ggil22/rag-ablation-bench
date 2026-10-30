"""Spanish consolidated legislation from the BOE open data API.

API docs: https://www.boe.es/datosabiertos/documentos/APIconsolidada.pdf

The consolidated text of a law is a sequence of <bloque> elements (articles, provisions,
headings, preamble), each holding every historical <version> of that block. We keep the
version in force on `as_of`, which makes the corpus reproducible after later amendments.

Inside a version, <blockquote> holds editorial material (amendment notes and the repealed
"Redacción anterior" wording) and is dropped: indexing it would let the system answer with
law that is no longer in force.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

import httpx

from rag_ablation_bench.schema import Document, Section

API = "https://www.boe.es/datosabiertos/api/legislacion-consolidada/id/{id}/{part}"

# Heading paragraph classes and their depth in the hierarchy.
HEADING_LEVELS = {
    "libro_num": 1,
    "libro_tit": 1,
    "titulo_num": 2,
    "titulo_tit": 2,
    "capitulo_num": 3,
    "capitulo_tit": 3,
    "seccion": 4,
    "subseccion": 5,
}
ANNEX_CLASSES = {"anexo", "anexo_num", "anexo_tit"}
SKIP_CLASSES = {"imagen"}
SKIP_BLOCKS = {"firma", "nota_inicial"}
PROVISION_GROUPS = {
    "adicional": "Disposiciones adicionales",
    "transitoria": "Disposiciones transitorias",
    "derogatoria": "Disposiciones derogatorias",
    "final": "Disposiciones finales",
}
PROVISION_GROUP = re.compile(
    r"Disposici[oó]n (adicional|transitoria|derogatoria|final)\b", re.IGNORECASE
)
REPEALED = re.compile(r"^\(?\s*(derogad|sin contenido|suprimid|anulad)", re.IGNORECASE)


# --- download ----------------------------------------------------------------------------


def raw_paths(raw_dir: Path, law_id: str) -> tuple[Path, Path]:
    return raw_dir / f"{law_id}.metadatos.xml", raw_dir / f"{law_id}.texto.xml"


def download(law_id: str, raw_dir: Path, *, force: bool = False, retries: int = 3) -> None:
    """Save the metadata and full consolidated text of a law as raw XML."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers={"Accept": "application/xml"}, timeout=120) as client:
        for part, path in zip(("metadatos", "texto"), raw_paths(raw_dir, law_id), strict=True):
            if path.exists() and not force:
                continue
            for attempt in range(retries):
                try:
                    resp = client.get(API.format(id=law_id, part=part))
                    resp.raise_for_status()
                    break
                except httpx.HTTPError:
                    if attempt == retries - 1:
                        raise
                    time.sleep(2**attempt)
            path.write_bytes(resp.content)


# --- parsing -----------------------------------------------------------------------------


def _clean(text: str) -> str:
    return " ".join(text.split())  # also folds the non-breaking spaces the BOE uses


def _ymd(value: str) -> date:
    return datetime.strptime(value[:8], "%Y%m%d").date()


def _version_in_force(block: ET.Element, as_of: date) -> tuple[ET.Element, int, int] | None:
    """The latest version effective on `as_of`, how many versions existed by then, and how
    many published versions have no effective date yet (pending; never treated as in force).
    """
    all_versions = block.findall("version")
    dated = [v for v in all_versions if v.get("fecha_vigencia")]
    versions = [v for v in dated if _ymd(v.get("fecha_vigencia")) <= as_of]
    if not versions:
        return None
    pending = sum(
        1
        for v in all_versions
        if not v.get("fecha_vigencia") and _ymd(v.get("fecha_publicacion")) <= as_of
    )
    return max(versions, key=lambda v: v.get("fecha_vigencia")), len(versions), pending


def _table_lines(table: ET.Element) -> list[str]:
    rows = []
    for tr in table.iter("tr"):
        cells = [_clean("".join(c.itertext())) for c in tr if c.tag in ("td", "th")]
        if any(cells):
            rows.append(" | ".join(cells))
    return rows


def _content(version: ET.Element) -> list[tuple[str | None, str]]:
    """(class, text) of each content line in a version, skipping editorial blockquotes."""
    lines: list[tuple[str | None, str]] = []
    for el in version:
        if el.tag == "p" and el.get("class") not in SKIP_CLASSES:
            text = _clean("".join(el.itertext()))
            if text:
                lines.append((el.get("class"), text))
        elif el.tag == "table":
            lines.extend(("tabla", row) for row in _table_lines(el))
    return lines


def parse_law(meta_xml: bytes, text_xml: bytes, *, as_of: date, short_title: str) -> Document:
    meta = ET.fromstring(meta_xml).find("data/metadatos")
    law_id = meta.findtext("identificador")

    sections: list[Section] = []
    headings: list[tuple[int, str]] = []  # stack of (level, heading text)
    for block in ET.fromstring(text_xml).iter("bloque"):
        kind = block.get("tipo")
        expiry = block.get("fecha_caducidad")  # replaced blocks, e.g. an old annex
        found = _version_in_force(block, as_of)
        if kind in SKIP_BLOCKS or found is None or (expiry and _ymd(expiry) <= as_of):
            continue
        version, n_versions, n_pending = found
        lines = _content(version)
        if not lines:
            continue
        classes = {cls for cls, _ in lines}

        # Decide by content, not `tipo`: some headings are tagged as "precepto", and some
        # titles use annex classes without having any annex body.
        only_titles = classes <= HEADING_LEVELS.keys() | ANNEX_CLASSES
        if only_titles or (kind == "encabezado" and not classes & ANNEX_CLASSES):
            parts = [(HEADING_LEVELS[c], t) for c, t in lines if c in HEADING_LEVELS]
            if parts:  # other lines in heading blocks are editorial notes
                level = min(lvl for lvl, _ in parts)
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, ". ".join(t for _, t in parts)))
            continue

        if kind == "encabezado":  # annexes are content, not headings
            kind = "annex"
            title = " ".join(t for c, t in lines if c in ANNEX_CLASSES)
            headings = []
        elif kind == "preambulo":
            kind, title = "preamble", "Preámbulo"
        else:
            title = next((t for c, t in lines if c == "articulo"), block.get("titulo", ""))
            kind = "article" if title.startswith("Artículo") else "provision"

        path = tuple(h for _, h in headings)
        # Provisions follow the last article with no heading block of their own, so they
        # would otherwise inherit the last title of the articulated text.
        if group := PROVISION_GROUP.match(title):
            path = (PROVISION_GROUPS[group[1].lower()],)
            headings = []

        text = "\n".join(t for _, t in lines)
        body = text.removeprefix(title).strip()
        sections.append(
            Section(
                id=f"{law_id}#{block.get('id')}",
                doc_id=law_id,
                kind=kind,
                title=title,
                path=path,
                text=text,
                metadata={
                    "in_force_since": _ymd(version.get("fecha_vigencia")).isoformat(),
                    "last_amended_by": version.get("id_norma")
                    if version.get("id_norma") != law_id
                    else None,
                    "n_versions": n_versions,
                    "pending_versions": n_pending,
                    "repealed": bool(REPEALED.match(body)),
                },
            )
        )

    return Document(
        id=law_id,
        corpus="boe",
        title=meta.findtext("titulo"),
        short_title=short_title,
        url=meta.findtext("url_html_consolidada"),
        sections=tuple(sections),
        metadata={"as_of": as_of.isoformat(), "rank": meta.findtext("rango")},
    )


def build(config: dict, raw_dir: Path, *, force_download: bool = False) -> list[Document]:
    """Download (if needed) and parse every law listed in the `boe` corpus config."""
    as_of = config["as_of"]
    as_of = as_of if isinstance(as_of, date) else date.fromisoformat(as_of)
    docs = []
    for law in config["laws"]:
        download(law["id"], raw_dir, force=force_download)
        meta_path, text_path = raw_paths(raw_dir, law["id"])
        docs.append(
            parse_law(
                meta_path.read_bytes(),
                text_path.read_bytes(),
                as_of=as_of,
                short_title=law["short"],
            )
        )
    return docs
