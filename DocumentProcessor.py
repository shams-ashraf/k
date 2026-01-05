import fitz
import pdfplumber
import pytesseract
from PIL import Image
import re
import os
import json
import hashlib
from typing import List, Dict
from collections import Counter
from semantic_text_splitter import TextSplitter
from tokenizers import Tokenizer
from dotenv import load_dotenv

load_dotenv()
CACHE_FOLDER = os.getenv("CACHE_FOLDER", "./cache")
DOCS_FOLDER = "/mount/src/k/documents"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
PDF_PASSWORD = os.getenv("PDF_PASSWORD", "mbe2025")

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
MAX_EMBEDDING_TOKENS = 384
MIN_CHUNK_TOKENS = 50
OVERLAP_TOKENS = 50

splitter = TextSplitter.from_huggingface_tokenizer(
    Tokenizer.from_pretrained("intfloat/multilingual-e5-large"),
    capacity=MAX_EMBEDDING_TOKENS,
    overlap=OVERLAP_TOKENS
)

def get_files_from_folder():
    files = []
    for f in os.listdir(DOCS_FOLDER):
        if f.lower().endswith(".pdf"):
            files.append(os.path.join(DOCS_FOLDER, f))
    return files

def get_file_hash(filepath: str) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_cache(key: str):
    cache_path = os.path.join(CACHE_FOLDER, f"{key}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_cache(key: str, data):
    cache_path = os.path.join(CACHE_FOLDER, f"{key}.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()

def normalize_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    return line.lower()

def remove_dynamic_noise(text: str, doc_name: str) -> str:
    out = []
    for line in text.splitlines():
        ln = normalize(line)
        
        if normalize(doc_name) in ln:
            continue
        
        if re.search(r"(seite|page)\s+\d+(\s+(von|of|de)\s+\d+)?", ln):
            continue
        
        out.append(line)
    
    return "\n".join(out)

def remove_empty_lines(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if l.strip())

def clean_chunk_content(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[_\-]{3,}', '', text)
    text = re.sub(r'[•·]{2,}', '', text)
    text = re.sub(r'[^\w\s.,!?:;\-\(\)\[\]{}"\'%€$£¥\n\u0600-\u06FF]', '', text, flags=re.UNICODE)
    text = re.sub(r'([.,!?:;])\1{2,}', r'\1', text)
    text = re.sub(r'\s+([.,!?:;])', r'\1', text)
    text = re.sub(r'([.,!?:;])\s+', r'\1 ', text)
    
    lines = [line.strip() for line in text.splitlines()]
    text = '\n'.join(line for line in lines if line)
    
    return text.strip()

def is_header(line: str) -> bool:
    line = line.strip()
    
    if len(line) > 120:
        return False
    
    if line.isupper():
        return True
    
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", line):
        return True
    
    return False

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def is_table_chunk(metadata: dict) -> bool:
    return metadata.get("type") == "table_with_context"

def smart_chunk_text(text: str, metadata: dict, max_tokens: int = MAX_EMBEDDING_TOKENS) -> List[Dict]:
    chunks = []
    text_tokens = estimate_tokens(text)
    
    if text_tokens <= max_tokens:
        return [{
            "content": text,
            "metadata": metadata
        }]
        
    semantic_chunks = splitter.chunks(text)
    
    for idx, chunk_text in enumerate(semantic_chunks, 1):
        chunks.append({
            "content": chunk_text,
            "metadata": {
                **metadata,
                "chunk_index": idx,
                "total_chunks": len(semantic_chunks),
                "is_split": True
            }
        })
    
    return chunks

def merge_small_page_chunks(page_chunks_list: List[List[Dict]], max_tokens: int = MAX_EMBEDDING_TOKENS) -> List[Dict]:

    if not page_chunks_list:
        return []
        
    merged_pages = []
    i = 0
    
    while i < len(page_chunks_list):
        current_page_chunks = page_chunks_list[i]
        
        current_total_tokens = sum(estimate_tokens(chunk["content"]) for chunk in current_page_chunks)
        
        has_current_table = any(is_table_chunk(chunk["metadata"]) for chunk in current_page_chunks)
        
        if has_current_table or current_total_tokens >= max_tokens:
            merged_pages.append(current_page_chunks)
            i += 1
            continue
        
        if i + 1 < len(page_chunks_list):
            next_page_chunks = page_chunks_list[i + 1]
            next_total_tokens = sum(estimate_tokens(chunk["content"]) for chunk in next_page_chunks)
            has_next_table = any(is_table_chunk(chunk["metadata"]) for chunk in next_page_chunks)
            
            combined_tokens = current_total_tokens + next_total_tokens
            
            if combined_tokens <= max_tokens and not has_next_table:
                
                merged_content = "\n\n".join([
                    chunk["content"] for chunk in current_page_chunks + next_page_chunks
                ])
                
                merged_chunk = {
                    "content": merged_content,
                    "metadata": {
                        "source": current_page_chunks[0]["metadata"]["source"],
                        "page": f"{current_page_chunks[0]['metadata']['page']}-{next_page_chunks[-1]['metadata']['page']}",
                        "type": "merged_pages",
                        "section": current_page_chunks[0]["metadata"].get("section", ""),
                        "original_pages": f"{current_page_chunks[0]['metadata']['page']},{next_page_chunks[0]['metadata']['page']}"
                    }
                }
                
                merged_pages.append([merged_chunk])
                i += 2  
                continue
        
        merged_pages.append(current_page_chunks)
        i += 1
    
    final_chunks = []
    for page_chunks in merged_pages:
        final_chunks.extend(page_chunks)
    
    return final_chunks

def aggressive_merge_small_chunks(chunks: List[Dict], min_tokens: int = MIN_CHUNK_TOKENS) -> List[Dict]:
    if not chunks:
        return chunks
        
    merged = []
    buffer = None
    
    for i, chunk in enumerate(chunks):
        chunk_tokens = estimate_tokens(chunk["content"])
        is_table = is_table_chunk(chunk["metadata"])
        
        if is_table:
            if buffer:
                merged.append(buffer)
                buffer = None
            
            merged.append(chunk)
            continue
        
        if chunk_tokens >= min_tokens:
            if buffer and buffer["metadata"]["page"] == chunk["metadata"]["page"]:
                chunk["content"] = buffer["content"] + "\n\n" + chunk["content"]
                chunk["metadata"]["is_merged"] = True
                buffer = None
            
            if buffer:
                if merged and merged[-1]["metadata"]["page"] == buffer["metadata"]["page"]:
                    merged[-1]["content"] = merged[-1]["content"] + "\n\n" + buffer["content"]
                    merged[-1]["metadata"]["is_merged"] = True
                else:
                    merged.append(buffer)
                buffer = None
            
            merged.append(chunk)
            continue
                
        if buffer:
            if buffer["metadata"]["page"] == chunk["metadata"]["page"]:
                buffer["content"] = buffer["content"] + "\n\n" + chunk["content"]
                buffer["metadata"]["is_merged"] = True
            else:
                if merged and merged[-1]["metadata"]["page"] == buffer["metadata"]["page"]:
                    merged[-1]["content"] = merged[-1]["content"] + "\n\n" + buffer["content"]
                    merged[-1]["metadata"]["is_merged"] = True
                else:
                    merged.append(buffer)
                
                buffer = chunk
        else:
            buffer = chunk
    
    if buffer:        
        if merged and merged[-1]["metadata"]["page"] == buffer["metadata"]["page"]:
            merged[-1]["content"] = merged[-1]["content"] + "\n\n" + buffer["content"]
            merged[-1]["metadata"]["is_merged"] = True
        else:
            merged.append(buffer)
    
    return merged

def final_filter_small_chunks(chunks: List[Dict], min_tokens: int = MIN_CHUNK_TOKENS) -> List[Dict]:    
    filtered = []
    removed_count = 0
    
    for chunk in chunks:
        chunk_tokens = estimate_tokens(chunk["content"])
        is_table = is_table_chunk(chunk["metadata"])
        
        if is_table:
            filtered.append(chunk)
            continue
        
        if chunk_tokens >= min_tokens:
            filtered.append(chunk)
        else:
            removed_count += 1
    
    return filtered

def detect_repeated_lines(pages_text: List[str], top_lines: int = 3, 
                         bottom_lines: int = 3, threshold: float = 0.7):
    total_pages = len(pages_text)
    header_lines = Counter()
    footer_lines = Counter()
    
    for text in pages_text:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        
        if len(lines) < top_lines + bottom_lines:
            continue
        
        for line in lines[:top_lines]:
            normalized = normalize_line(line)
            if normalized:
                header_lines[normalized] += 1
        
        for line in lines[-bottom_lines:]:
            normalized = normalize_line(line)
            if normalized:
                footer_lines[normalized] += 1
    
    repeated_headers = {
        line for line, count in header_lines.items() 
        if count / total_pages >= threshold
    }
    
    repeated_footers = {
        line for line, count in footer_lines.items() 
        if count / total_pages >= threshold
    }
    
    return repeated_headers, repeated_footers

def remove_repeated_lines(text: str, repeated_headers: set, repeated_footers: set) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        normalized = normalize_line(line)
        
        if normalized not in repeated_headers and normalized not in repeated_footers:
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

def clean_cell(cell):
    return str(cell).strip() if cell is not None else ""

def normalize_table(table: List[List[str]]) -> List[List[str]]:
    if not table:
        return []
    
    cleaned = [[clean_cell(c) for c in row] for row in table]
    cleaned = [row for row in cleaned if any(cell for cell in row)]
    
    if len(cleaned) <= 1:
        return []
    
    max_cols = max(len(row) for row in cleaned)
    cleaned = [row + [""] * (max_cols - len(row)) for row in cleaned]
    
    cols_to_keep = [i for i in range(max_cols) if any(row[i] for row in cleaned)]
    
    return [[row[i] for i in cols_to_keep] for row in cleaned]

def headers_similar(h1, h2) -> bool:
    h1 = [c.lower() for c in h1]
    h2 = [c.lower() for c in h2]
    same = sum(1 for a, b in zip(h1, h2) if a == b)
    return same >= max(1, len(h1) // 2)

def table_bbox(table_obj):
    return table_obj.bbox[1], table_obj.bbox[3]

def has_text_between(page, y1, y2) -> bool:
    for w in page.extract_words():
        if y1 < w["top"] < y2:
            return True
    return False

def table_to_markdown(table: List[List[str]]) -> str:
    if not table or len(table) < 2:
        return ""
    
    lines = []
    lines.append("| " + " | ".join(table[0]) + " |")
    lines.append("|" + "|".join(["---" for _ in table[0]]) + "|")
    
    for row in table[1:]:
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)

def render_page_for_ocr(fitz_doc, page_index: int) -> Image.Image:
    page = fitz_doc.load_page(page_index)
    pix = page.get_pixmap(dpi=300, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def ocr_says_table(fitz_doc, page_index: int) -> bool:
    img = render_page_for_ocr(fitz_doc, page_index)
    text = pytesseract.image_to_string(img, lang="deu+eng", config="--psm 6")
    
    score = 0
    for l in text.splitlines():
        digits = sum(c.isdigit() for c in l)
        letters = sum(c.isalpha() for c in l)
        if digits >= 2:
            score += 1
        if digits > 0 and letters > 0:
            score += 1
    
    return score >= 6

def extract_pdf_detailed(pdf_path: str):
    fitz_doc = None
    try:
        fitz_doc = fitz.open(pdf_path)

        if fitz_doc.is_encrypted:
            if not PDF_PASSWORD:
                return None, "❌ PDF is encrypted but no password was provided"

            if not fitz_doc.authenticate(PDF_PASSWORD):
                return None, "❌ Wrong PDF password"
    
        doc_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        page_chunks_list = []  
        
        cleaned_pages = []
        for page_num, page in enumerate(fitz_doc, 1):
            raw_text = page.get_text()
            cleaned = remove_empty_lines(remove_dynamic_noise(raw_text, doc_name))
            cleaned_pages.append(cleaned)
                
        repeated_headers, repeated_footers = detect_repeated_lines(
            cleaned_pages, 
            top_lines=3,
            bottom_lines=3,
            threshold=0.7
        )
        
        with pdfplumber.open(pdf_path, password=PDF_PASSWORD or None) as pdf:
            current_section = ""
            last_context_text = ""
            
            for page_idx, (clean_text, plumber_page) in enumerate(zip(cleaned_pages, pdf.pages), start=1):                
                page_specific_chunks = [] 
                
                clean_text = remove_repeated_lines(clean_text, repeated_headers, repeated_footers)
                
                tables = plumber_page.find_tables()
                
                if not tables and ocr_says_table(fitz_doc, page_idx - 1):
                    tables = plumber_page.find_tables(
                        table_settings={
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "snap_tolerance": 3,
                            "join_tolerance": 3
                        }
                    )
                
                table_blocks = []
                if tables:
                    merged_tables = []
                    current_table = None
                    current_bottom = None
                    
                    for t in tables:
                        normalized = normalize_table(t.extract())
                        if not normalized:
                            continue
                        
                        top, bottom = table_bbox(t)
                        
                        if current_table is None:
                            current_table = normalized
                            current_bottom = bottom
                            continue
                        
                        if not has_text_between(plumber_page, current_bottom, top):
                            if headers_similar(current_table[0], normalized[0]):
                                current_table.extend(normalized[1:])
                            else:
                                current_table.extend(normalized)
                            current_bottom = bottom
                        else:
                            merged_tables.append(current_table)
                            current_table = normalized
                            current_bottom = bottom
                    
                    if current_table:
                        merged_tables.append(current_table)
                    
                    for table in merged_tables:
                        if len(table) > 1 and len(table[0]) > 1:
                            table_md = table_to_markdown(table)
                            table_blocks.append(table_md)
                
                page_lines = []
                for ln in clean_text.splitlines():
                    if is_header(ln):
                        current_section = ln.strip()
                    else:
                        if len(ln.strip().split()) >= 5:
                            last_context_text = ln.strip()
                    
                    page_lines.append(ln)
                
                page_text = "\n".join(page_lines)
                
                has_content = page_text.strip() and len(page_text.strip()) > 10
                
                if not has_content and not table_blocks:
                    continue
                
                if table_blocks:
                    combined_content = f"{page_text}\n\n### TABLES:\n\n" + "\n\n".join(table_blocks)
                    
                    page_specific_chunks.append({
                        "content": combined_content,
                        "metadata": {
                            "source": doc_name,
                            "page": page_idx,
                            "type": "table_with_context",
                            "section": current_section,
                            "table_count": len(table_blocks)
                        }
                    })
                else:
                    if has_content:
                        text_chunks = smart_chunk_text(
                            page_text,
                            metadata={
                                "source": doc_name,
                                "page": page_idx,
                                "type": "semantic_text",
                                "section": current_section,
                                "context": last_context_text
                            }
                        )
                        page_specific_chunks.extend(text_chunks)
                
                if page_specific_chunks:
                    page_chunks_list.append(page_specific_chunks)
        
        fitz_doc.close()
        
        merged_by_page = merge_small_page_chunks(page_chunks_list, MAX_EMBEDDING_TOKENS)
        
        merged_chunks = aggressive_merge_small_chunks(merged_by_page, MIN_CHUNK_TOKENS)
        
        final_chunks = final_filter_small_chunks(merged_chunks, MIN_CHUNK_TOKENS)
        
        for i, chunk in enumerate(final_chunks, 1):
            chunk["content"] = clean_chunk_content(chunk["content"])
                
        token_distribution = {}
        for chunk in final_chunks:
            tokens = estimate_tokens(chunk["content"])
            bucket = (tokens // 50) * 50
            token_distribution[bucket] = token_distribution.get(bucket, 0) + 1
        
        for bucket in sorted(token_distribution.keys()):
            print(f"  {bucket}-{bucket+49} tokens: {token_distribution[bucket]} chunks")
        
        return {"chunks": final_chunks}, None
    
    except Exception as e:
        print(f"\n❌ Error processing {pdf_path}: {str(e)}")
        if fitz_doc:
            fitz_doc.close()
        return None, str(e)

