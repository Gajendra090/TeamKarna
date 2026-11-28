"""Document scanning, image quality validation, Gemini OCR, and searchable PDF assembly."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pymupdf
from google.genai import types
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from modules.ai_analysis import AIServiceError, DEFAULT_MODEL, generate_structured_json
from modules.pdf_reader import PDFProcessingError, extract_pdf_bytes, sanitize_filename

MAX_SCAN_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB per image
MIN_IMAGE_DIMENSION = 120
MIN_CONTRAST_STD = 4.0
MIN_SHARPNESS_LAPLACIAN = 3.0
LOW_SHARPNESS_WARNING = 20.0


class ScanProcessingError(Exception):
    """Raised when scanned images cannot be validated, read, or converted."""


class OCRPageExtraction(BaseModel):
    is_readable: bool = Field(
        description="True if meaningful text can be read accurately; False if blank, blurry, or illegible."
    )
    legibility: Literal["Clear", "Partially Readable", "Unreadable"]
    quality_issue: str = Field(
        description="Brief description of blur, glare, cropping, or low contrast if any; empty string if clear."
    )
    extracted_text: str = Field(
        description="Faithful transcription of all visible text in reading order. Empty string if unreadable."
    )


def _compute_blur_and_contrast(img: Image.Image) -> tuple[float, float]:
    """Compute grayscale contrast (std) and edge sharpness (Laplacian variance)."""
    gray = img.convert("L")
    gray.thumbnail((900, 900))
    arr = np.asarray(gray, dtype=np.float32)
    contrast = float(arr.std())
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        return 0.0, contrast
    lap = (
        -4.0 * arr[1:-1, 1:-1]
        + arr[:-2, 1:-1]
        + arr[2:, 1:-1]
        + arr[1:-1, :-2]
        + arr[1:-1, 2:]
    )
    return float(lap.var()), contrast


def assess_image_quality(
    image_bytes: bytes,
    filename: str = "scanned_page.png",
) -> dict[str, Any]:
    """Validate an uploaded or captured image and check for blur or blank pages."""
    safe_name = Path(filename or "scanned_page.png").name
    if not image_bytes:
        raise ScanProcessingError(f"'{safe_name}' is empty (0 bytes).")
    if len(image_bytes) > MAX_SCAN_IMAGE_BYTES:
        raise ScanProcessingError(
            f"'{safe_name}' exceeds maximum image size ({MAX_SCAN_IMAGE_BYTES // (1024 * 1024)} MB)."
        )

    try:
        with Image.open(io.BytesIO(image_bytes)) as raw_img:
            transposed = ImageOps.exif_transpose(raw_img)
            rgb_img = transposed.convert("RGB")
    except (UnidentifiedImageError, OSError, Exception) as exc:
        raise ScanProcessingError(
            f"'{safe_name}' is not a valid or readable image ({exc})."
        ) from exc

    width, height = rgb_img.size
    if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
        raise ScanProcessingError(
            f"'{safe_name}' resolution ({width}x{height}) is too small for OCR."
        )

    sharpness, contrast = _compute_blur_and_contrast(rgb_img)
    if contrast < MIN_CONTRAST_STD:
        raise ScanProcessingError(
            f"'{safe_name}' appears blank or uniform (contrast={contrast:.1f})."
        )
    if sharpness < MIN_SHARPNESS_LAPLACIAN:
        raise ScanProcessingError(
            f"'{safe_name}' is too blurry to extract text reliably "
            f"(sharpness={sharpness:.1f}). Please recapture with sharper focus."
        )

    quality_warning = ""
    if sharpness < LOW_SHARPNESS_WARNING:
        quality_warning = (
            f"Low sharpness detected ({sharpness:.1f}); some text may be partially readable."
        )

    # Normalize large mobile photos to <=2000px longest side for fast OCR & compact PDF
    work_img = rgb_img.copy()
    work_img.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
    norm_w, norm_h = work_img.size

    png_buf = io.BytesIO()
    work_img.save(png_buf, format="PNG", optimize=True)
    png_bytes = png_buf.getvalue()

    jpg_buf = io.BytesIO()
    work_img.save(jpg_buf, format="JPEG", quality=88, optimize=True)
    pdf_image_bytes = jpg_buf.getvalue()

    return {
        "filename": safe_name,
        "width": norm_w,
        "height": norm_h,
        "sharpness": round(sharpness, 2),
        "contrast": round(contrast, 2),
        "quality_warning": quality_warning,
        "normalized_png_bytes": png_bytes,
        "pdf_image_bytes": pdf_image_bytes,
    }


def extract_text_from_scan_image(
    prepared_image: dict[str, Any],
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    allow_fallback: bool = True,
) -> dict[str, Any]:
    """Run multimodal Gemini OCR on a validated scan image."""
    system_instruction = (
        "You are a precision OCR engine for scanned research documents and notes. "
        "Transcribe visible text accurately in reading order without inventing missing words. "
        "If an image is severely blurred, blank, or illegible, set is_readable=false and "
        "legibility='Unreadable' and explain why in quality_issue."
    )
    prompt_text = (
        f"Extract all readable text from this scanned page ({prepared_image['filename']}). "
        "Preserve headings, paragraphs, formulas, and list structure clearly."
    )
    image_part = types.Part.from_bytes(
        data=prepared_image["normalized_png_bytes"],
        mime_type="image/png",
    )

    ocr_result = generate_structured_json(
        prompt=[prompt_text, image_part],
        schema=OCRPageExtraction,
        system_instruction=system_instruction,
        model=model,
        api_key=api_key,
        temperature=0.0,
        allow_fallback=allow_fallback,
    )
    return {
        "filename": prepared_image["filename"],
        "is_readable": bool(ocr_result.get("is_readable")),
        "legibility": ocr_result.get("legibility", "Unreadable"),
        "quality_issue": (ocr_result.get("quality_issue") or "").strip(),
        "extracted_text": (ocr_result.get("extracted_text") or "").strip(),
        "quality_warning": prepared_image.get("quality_warning", ""),
        "sharpness": prepared_image["sharpness"],
        "model_used": ocr_result.get("_model_used", model),
        "fallback_note": ocr_result.get("_fallback_note", ""),
    }


def _embed_invisible_text(page: pymupdf.Page, text: str) -> None:
    """Embed searchable OCR text cleanly onto a PDF page without overflowing."""
    clean_text = (text or "").strip()
    if not clean_text:
        return

    margin = 28.0
    box = pymupdf.Rect(margin, margin, page.rect.width - margin, page.rect.height - margin)

    # Fit text cleanly inside the page bounds using progressive font sizes
    for font_size in (10.0, 8.0, 6.5, 5.0):
        tw = pymupdf.TextWriter(page.rect)
        overflow = tw.fill_textbox(box, clean_text, fontsize=font_size)
        if not overflow or font_size == 5.0:
            tw.write_text(page, render_mode=3)
            return


def build_searchable_pdf_from_scans(scanned_pages: list[dict[str, Any]]) -> bytes:
    """Assemble readable scanned images and OCR text layers into a multi-page PDF."""
    if not scanned_pages:
        raise ScanProcessingError("No readable scanned pages available to convert to PDF.")

    doc = pymupdf.open()
    try:
        for item in scanned_pages:
            img_w = max(1, int(item["width"]))
            img_h = max(1, int(item["height"]))
            page_w = 595.0
            page_h = max(320.0, min(1150.0, round(page_w * (img_h / img_w), 1)))
            page = doc.new_page(width=page_w, height=page_h)
            page.insert_image(page.rect, stream=item["pdf_image_bytes"])
            _embed_invisible_text(page, item["extracted_text"])

        doc.set_metadata(
            {
                "title": "Scanned Document",
                "producer": "KNOWLEDGE X OCR Scanner",
            }
        )
        return doc.tobytes(garbage=3, deflate=True)
    finally:
        doc.close()


def process_scanned_pages(
    pages_input: list[dict[str, Any]],
    output_filename: str,
    existing_docs: list[dict[str, Any]],
    save_dir: Path | str,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    allow_fallback: bool = True,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[str]]:
    """Validate scanned images, extract text via OCR, build PDF, and ingest into KNOWLEDGE X."""
    if not pages_input:
        raise ScanProcessingError("Add or capture at least one scanned page image first.")

    safe_pdf_name = sanitize_filename(output_filename or "scanned_document.pdf")
    total = len(pages_input)
    page_reports: list[dict[str, Any]] = []
    readable_pages: list[dict[str, Any]] = []
    warnings: list[str] = []

    for idx, page_item in enumerate(pages_input, start=1):
        fname = page_item.get("name", f"page_{idx}.png")
        raw_bytes = page_item.get("bytes", b"")
        if on_progress:
            on_progress(idx, total, f"Checking image quality & running OCR on page {idx}/{total} ({fname})...")

        try:
            prepared = assess_image_quality(raw_bytes, filename=fname)
        except ScanProcessingError as exc:
            warnings.append(str(exc))
            page_reports.append(
                {
                    "page": idx,
                    "filename": fname,
                    "status": "Rejected",
                    "legibility": "Unreadable",
                    "issue": str(exc),
                    "word_count": 0,
                }
            )
            continue

        if prepared["quality_warning"]:
            warnings.append(f"{fname}: {prepared['quality_warning']}")

        try:
            ocr = extract_text_from_scan_image(
                prepared_image=prepared,
                model=model,
                api_key=api_key,
                allow_fallback=allow_fallback,
            )
        except AIServiceError:
            raise
        except Exception as exc:
            warnings.append(f"{fname}: OCR failed ({exc}).")
            page_reports.append(
                {
                    "page": idx,
                    "filename": fname,
                    "status": "OCR Failed",
                    "legibility": "Unreadable",
                    "issue": str(exc),
                    "word_count": 0,
                }
            )
            continue

        if not ocr["is_readable"] or not ocr["extracted_text"]:
            reason = ocr["quality_issue"] or "No readable text detected in the image."
            warnings.append(f"{fname}: Unreadable ({reason})")
            page_reports.append(
                {
                    "page": idx,
                    "filename": fname,
                    "status": "Unreadable",
                    "legibility": ocr["legibility"],
                    "issue": reason,
                    "word_count": 0,
                }
            )
            continue

        readable_pages.append(
            {
                **prepared,
                "extracted_text": ocr["extracted_text"],
                "legibility": ocr["legibility"],
                "fallback_note": ocr.get("fallback_note", ""),
            }
        )
        page_reports.append(
            {
                "page": idx,
                "filename": fname,
                "status": "Extracted",
                "legibility": ocr["legibility"],
                "issue": ocr["quality_issue"] or prepared["quality_warning"],
                "word_count": len(ocr["extracted_text"].split()),
                "fallback_note": ocr.get("fallback_note", ""),
            }
        )

    if not readable_pages:
        detail = "; ".join(warnings) if warnings else "No legible pages found."
        raise ScanProcessingError(
            f"Could not extract readable text from any scanned page. {detail}"
        )

    pdf_bytes = build_searchable_pdf_from_scans(readable_pages)
    try:
        doc_record = extract_pdf_bytes(pdf_bytes, safe_pdf_name, save_dir=save_dir)
    except PDFProcessingError as exc:
        raise ScanProcessingError(f"Failed to assemble searchable PDF: {exc}") from exc

    doc_record["is_scanned"] = True
    updated_docs = list(existing_docs) + [doc_record]
    return updated_docs, doc_record, page_reports, warnings
