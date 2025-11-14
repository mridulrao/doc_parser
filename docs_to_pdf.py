
#!/usr/bin/env python3
"""
Convert a DOCX file to PDF.

Usage:
  python docx_to_pdf.py /path/to/input.docx [ /path/to/output.pdf ]

Requires (any one of):
  • Windows/macOS: pip install docx2pdf  (and Word on Windows; Preview/Word on macOS)
  • Any OS: LibreOffice installed and 'soffice' in PATH

Notes:
  - On Linux, use the LibreOffice path.
  - Output path is optional; defaults to input path with .pdf extension.
"""

import os
import sys
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

def _has_module(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False

def _which(cmd: str) -> Optional[str]:
    path = shutil.which(cmd)
    return path

def _convert_with_docx2pdf(src: Path, dst: Path) -> None:
    from docx2pdf import convert  # type: ignore
    # docx2pdf writes to a directory or converts in-place; to control filename, convert to temp dir and move.
    with tempfile.TemporaryDirectory() as td:
        tmp_out_dir = Path(td)
        convert(str(src), str(tmp_out_dir))
        produced = tmp_out_dir / (src.stem + ".pdf")
        if not produced.exists():
            # Some Word versions may produce different casing or name; search
            candidates = list(tmp_out_dir.glob("*.pdf"))
            if not candidates:
                raise RuntimeError("docx2pdf did not produce a PDF.")
            produced = candidates[0]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(produced), str(dst))

def _convert_with_soffice(src: Path, dst: Path, soffice_cmd: str) -> None:
    """
    Uses LibreOffice headless converter:
      soffice --headless --convert-to pdf --outdir <dir> <file>
    """
    outdir = dst.parent
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        soffice_cmd,
        "--headless",
        "--nologo",
        "--nofirststartwizard",
        "--convert-to", "pdf",
        "--outdir", str(outdir),
        str(src)
    ]
    try:
        completed = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
            timeout=120
        )
    except FileNotFoundError:
        raise FileNotFoundError("LibreOffice 'soffice' not found on PATH.")
    except subprocess.TimeoutExpired:
        raise TimeoutError("LibreOffice conversion timed out.")

    if completed.returncode != 0:
        raise RuntimeError(
            f"LibreOffice conversion failed (code {completed.returncode}).\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    produced = outdir / (src.stem + ".pdf")
    if not produced.exists():
        # LibreOffice might adjust name (spaces, etc.); search the outdir for newest PDF
        candidates = list(outdir.glob("*.pdf"))
        if not candidates:
            raise RuntimeError("LibreOffice did not produce a PDF.")
        produced = max(candidates, key=lambda p: p.stat().st_mtime)

    if produced.resolve() != dst.resolve():
        # If LibreOffice wrote a file with a different name, rename it to requested dst
        if dst.exists():
            dst.unlink()
        produced.rename(dst)

def convert_docx_to_pdf(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert a .docx file to PDF.
    Returns the output PDF path (string).

    Tries:
      1) docx2pdf (Windows/macOS preferred if available)
      2) LibreOffice 'soffice' fallback on any OS
    """
    src = Path(input_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Input not found: {src}")
    if src.suffix.lower() != ".docx":
        raise ValueError("Input file must have .docx extension.")

    dst = Path(output_path).expanduser().resolve() if output_path else src.with_suffix(".pdf")

    # Strategy 1: docx2pdf on Windows/macOS if available
    is_windows = sys.platform.startswith("win")
    is_macos = sys.platform == "darwin"
    if (is_windows or is_macos) and _has_module("docx2pdf"):
        try:
            _convert_with_docx2pdf(src, dst)
            return str(dst)
        except Exception as e:
            # Fallback to soffice
            sys.stderr.write(f"[warn] docx2pdf failed ({e}); trying LibreOffice fallback...\n")

    # Strategy 2: LibreOffice on any OS
    soffice = _which("soffice") or _which("soffice.bin") or _which("libreoffice")
    if not soffice:
        tips = [
            "Option A (Windows/macOS): pip install docx2pdf (and ensure MS Word is installed on Windows).",
            "Option B (Any OS): install LibreOffice and ensure 'soffice' is in PATH."
        ]
        raise EnvironmentError(
            "No conversion backend available.\n"
            + "\n".join(f"- {t}" for t in tips)
        )

    _convert_with_soffice(src, dst, soffice)
    return str(dst)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert DOCX to PDF")
    parser.add_argument("input", help="Path to input .docx")
    parser.add_argument("output", nargs="?", help="Optional output .pdf path")
    args = parser.parse_args()

    try:
        out = convert_docx_to_pdf(args.input, args.output)
        print(f"✅ PDF created: {out}")
    except Exception as e:
        print(f"❌ Conversion failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
