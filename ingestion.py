import os
import io
import fitz
import pdfplumber
import chromadb
import google.generativeai as genai
from PIL import Image
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import (
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP,
    CHROMA_PATH, COLLECTION_NAME, GEMINI_API_KEY
)


# ── shared helpers ────────────────────────────────────────────────────────────

def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def get_ingested_sources(collection):
    """
    Asks ChromaDB which source filenames are already stored.

    ChromaDB does not have a built-in 'list all unique metadata values'
    method. So we fetch every document's metadata (no text, no embeddings,
    just metadata) and extract the unique 'source' values ourselves.

    This is used at startup and before every ingestion so the app never
    ingests the same file twice, even after a Streamlit restart.
    """
    result = collection.get(include=["metadatas"])
    sources = set()
    for meta in result["metadatas"]:
        if "source" in meta:
            sources.add(meta["source"])
    return sources


def delete_source(collection, source_filename):
    """
    Removes every chunk belonging to one source file from ChromaDB.

    ChromaDB's where filter lets us find all IDs whose metadata matches
    a condition. We fetch those IDs then delete them in one call.
    """
    result = collection.get(
        where={"source": source_filename},
        include=[]
    )
    if result["ids"]:
        collection.delete(ids=result["ids"])
    return len(result["ids"])


def embed_and_store(chunks, source_filename, model, collection):
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=False)

    ids = [
        f"{source_filename}_p{chunk['page']}_{chunk['type']}_{i}"
        for i, chunk in enumerate(chunks)
    ]
    metadatas = [
        {
            "source": source_filename,
            "page": chunk["page"],
            "type": chunk["type"]
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )
    return len(chunks)


# ── layer 1 — text ────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num, page in enumerate(doc):
        text = page.get_text()
        if len(text.strip()) < 100:
            continue
        pages.append({"text": text, "page": page_num + 1})
    doc.close()
    return pages


def chunk_pages(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            chunks.append({
                "text": split,
                "page": page["page"],
                "type": "text"
            })
    return chunks


# ── layer 2 — tables ──────────────────────────────────────────────────────────

def table_to_markdown(table):
    rows = []
    for row in table:
        clean_cells = [cell if cell is not None else "" for cell in row]
        rows.append("| " + " | ".join(clean_cells) + " |")
    return "\n".join(rows)


def extract_tables_as_chunks(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                if len(table) < 2:
                    continue
                markdown = table_to_markdown(table)
                text = f"Table on page {page_num + 1}:\n{markdown}"
                chunks.append({
                    "text": text,
                    "page": page_num + 1,
                    "type": "table"
                })
    return chunks


# ── layer 3 — images / charts ─────────────────────────────────────────────────

def find_exhibit_regions(page, scale_factor):
    text_dict = page.get_text("dict")
    exhibit_ys = []
    source_ys = []

    for block in text_dict["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                lower = span["text"].strip().lower()
                y0 = span["bbox"][1]
                if lower.startswith("exhibit"):
                    exhibit_ys.append(y0 * scale_factor)
                if lower.startswith("source"):
                    source_ys.append(
                        (y0 + span["bbox"][3]) / 2 * scale_factor
                    )

    regions = []
    for ey in exhibit_ys:
        candidates = [sy for sy in source_ys if sy > ey]
        if candidates:
            regions.append((ey, min(candidates)))
    return regions


def describe_chart_with_gemini(image_bytes):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    img = Image.open(io.BytesIO(image_bytes))
    prompt = (
        "This is a chart from an equity research report. "
        "Describe what it shows in detail: the title, axis labels, "
        "all data series names, approximate values at key points, "
        "and the overall trend. Focus on the financial data. "
        "Be specific with numbers where visible."
    )
    try:
        response = model.generate_content([prompt, img])
        return response.text.strip()
    except Exception as e:
        print(f"  Gemini API error: {e}")
        return ""


def extract_images_as_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    chunks = []
    scale_factor = 2.0
    matrix = fitz.Matrix(scale_factor, scale_factor)

    for page_num, page in enumerate(doc):
        regions = find_exhibit_regions(page, scale_factor)
        if not regions:
            continue

        pixmap = page.get_pixmap(matrix=matrix)
        page_img = Image.frombytes(
            "RGB", [pixmap.width, pixmap.height], pixmap.samples
        )

        for idx, (top_y, bottom_y) in enumerate(regions):
            margin = 10
            crop_box = (
                0,
                max(0, int(top_y) - margin),
                pixmap.width,
                min(pixmap.height, int(bottom_y) + margin)
            )
            cropped = page_img.crop(crop_box)

            buf = io.BytesIO()
            cropped.save(buf, format="PNG")
            image_bytes = buf.getvalue()

            print(
                f"  Page {page_num + 1}, exhibit {idx + 1}: "
                f"sending to Gemini Vision..."
            )
            description = describe_chart_with_gemini(image_bytes)

            if description:
                chunks.append({
                    "text": (
                        f"Chart on page {page_num + 1} "
                        f"(exhibit {idx + 1}): {description}"
                    ),
                    "page": page_num + 1,
                    "type": "image"
                })

    doc.close()
    return chunks


# ── main ingestion entry point ────────────────────────────────────────────────

def ingest_pdf(pdf_path, model, collection, source_name=None):
    source_filename = source_name if source_name else os.path.basename(pdf_path)
    all_chunks = []

    print(f"[Layer 1] Extracting text from {source_filename}...")
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        return 0, "No extractable text found. PDF may be scanned."
    text_chunks = chunk_pages(pages)
    all_chunks.extend(text_chunks)
    print(f"  → {len(text_chunks)} text chunks")

    print(f"[Layer 2] Extracting tables from {source_filename}...")
    table_chunks = extract_tables_as_chunks(pdf_path)
    all_chunks.extend(table_chunks)
    print(f"  → {len(table_chunks)} table chunks")

    print(f"[Layer 3] Extracting charts from {source_filename}...")
    image_chunks = extract_images_as_chunks(pdf_path)
    all_chunks.extend(image_chunks)
    print(f"  → {len(image_chunks)} image chunks")

    print(f"[Store] Embedding and storing {len(all_chunks)} total chunks...")
    count = embed_and_store(all_chunks, source_filename, model, collection)

    summary = (
        f"Ingested {count} chunks from {source_filename} "
        f"({len(text_chunks)} text, "
        f"{len(table_chunks)} table, "
        f"{len(image_chunks)} image)"
    )
    return count, summary