# Tiny PDF AI Summarizer

A simple command-line tool that extracts selectable text from a PDF and asks a tiny local language model to produce a short summary.

## Features

- Accepts PDFs with up to 30 pages.
- Extracts text locally with [PyMuPDF](https://pymupdf.readthedocs.io/).
- Uses `sshleifer/tiny-gpt2`, a deliberately small Hugging Face text-generation model whose weights are far below 100 MB.
- Runs on CPU and downloads the model automatically on first use.
- Can save the summary to a text file.

## Requirements

- Python 3.10 or newer
- Internet access for the first model download
- A selectable-text PDF. Scanned or image-only PDFs need OCR first.

The model is tiny, but the Python runtime dependencies are larger than the model itself.

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python summarize_pdf.py path/to/document.pdf
```

Save the output:

```bash
python summarize_pdf.py document.pdf --output summary.txt
```

PDFs over 30 pages are rejected before model loading. The script also reports a clear error when no selectable text is present.

## Important limitation

`sshleifer/tiny-gpt2` is a miniature general text-generation checkpoint, not a capable instruction-tuned summarization model. It is used here to keep the project lightweight and educational. Its summaries may be repetitive, incomplete, or inaccurate. Do not use the output as a substitute for reading the source document or for professional, legal, medical, or financial decisions.

For better quality, replace `MODEL_ID` in `summarize_pdf.py` with a dedicated summarization model and adjust the size and memory requirements.

## License and model attribution

The project code is released under the MIT License. The model is hosted by [Hugging Face](https://huggingface.co/sshleifer/tiny-gpt2); review the model repository's terms before redistributing model files.
