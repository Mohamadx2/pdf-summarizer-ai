"""Summarize a selectable-text PDF with a tiny local language model."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pymupdf
from transformers import pipeline

MODEL_ID = "sshleifer/tiny-gpt2"
MAX_PAGES = 30
MAX_PROMPT_CHARS = 2_000


def extract_text(pdf_path: Path) -> tuple[int, str]:
    """Extract selectable text and return (page_count, text)."""
    with pymupdf.open(pdf_path) as document:
        page_count = len(document)
        if page_count > MAX_PAGES:
            raise ValueError(
                f"This tool supports at most {MAX_PAGES} pages; the PDF has {page_count}."
            )
        pages = [page.get_text("text") for page in document]
    text = "\n\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError(
            "No selectable text was found. Scanned/image-only PDFs need OCR before use."
        )
    return page_count, text


def build_summarizer():
    """Load the tiny model on CPU; weights download on first use."""
    return pipeline("text-generation", model=MODEL_ID, device=-1)


def summarize(generator, text: str) -> str:
    """Ask the tiny model for a short summary of the document text."""
    excerpt = text[:MAX_PROMPT_CHARS]
    prompt = (
        "Summarize the following document in three short bullet points. "
        "Only include information supported by the document.\n\n"
        f"Document:\n{excerpt}\n\nSummary:\n"
    )
    output = generator(
        prompt,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.15,
        pad_token_id=generator.tokenizer.eos_token_id,
        return_full_text=False,
    )[0]["generated_text"]
    answer = output.strip()
    answer = re.split(r"\n(?:Document|User|Summary):", answer, maxsplit=1)[0].strip()
    return answer or "The tiny model did not generate a summary."


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF of up to 30 pages and summarize it locally."
    )
    parser.add_argument("pdf", type=Path, help="Path to a selectable-text PDF")
    parser.add_argument(
        "-o", "--output", type=Path, help="Optional path for the text summary"
    )
    args = parser.parse_args()

    if args.pdf.suffix.lower() != ".pdf":
        raise SystemExit("Input must be a .pdf file.")
    if not args.pdf.exists():
        raise SystemExit(f"File not found: {args.pdf}")

    try:
        page_count, text = extract_text(args.pdf)
    except (OSError, ValueError) as error:
        raise SystemExit(f"Could not read PDF: {error}") from error

    print(f"Read {page_count} page(s) and {len(text):,} characters.")
    print(f"Loading {MODEL_ID} (downloaded automatically on first run)...")
    result = summarize(build_summarizer(), text)
    print("\nSummary\n=======\n")
    print(result)

    if args.output:
        args.output.write_text(result + "\n", encoding="utf-8")
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()
