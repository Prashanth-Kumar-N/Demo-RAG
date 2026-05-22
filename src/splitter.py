# src/splitter.py

from llama_index.core.schema import TextNode
from src.constants import success, error


def is_heading(text, max_heading_length=100):
    """
    Detect if a line is likely a heading.
    Headings are typically short lines, possibly with limited punctuation.
    """
    stripped = text.strip()
    if not stripped:
        return False
    
    # Lines under 100 chars that end with minimal punctuation are likely headings
    is_short = len(stripped) < max_heading_length
    # Check if it looks like a heading (no period, or all caps, or starts with number/symbol)
    looks_like_heading = (
        not stripped.endswith('.') or
        stripped.isupper() or
        stripped.endswith(":") or
        stripped[0].isdigit() or
        stripped[0] in '#-*'
    )
    return is_short and looks_like_heading


def split_text_with_headings(text, chunk_size=1200):
    """
    Split text while preserving headings with their content.
    
    Strategy:
    1. Split by paragraphs (\n\n) and lines (\n)
    2. Group headings with following content
    3. If chunk > 1200 chars, split by sentences, words, then characters
    """
    lines = text.split('\n')
    chunks = []
    current_chunk = {"heading": "", "content": []}
    current_size = 0
    
    i = 0
    while i < len(lines):
        line = lines[i]
        line_size = len(line) + 1  # +1 for newline
        
        # If this is a heading, try to keep it with following content
        if is_heading(line):
            # If current chunk is not empty and adding heading would exceed limit,
            # save current chunk first
            if current_chunk and current_size + line_size > chunk_size:
                chunks.append({'heading': current_chunk["heading"], 'content': "\n".join(current_chunk["content"])})
                current_chunk = {"heading": line, "content": []}
                current_size = line_size
            else:
                current_chunk["content"].append(line)
                current_size += line_size
        else:
            # Regular content line
            if current_size + line_size > chunk_size:
                if current_chunk:
                    chunks.append({'heading': current_chunk["heading"], 'content': "\n".join(current_chunk["content"])})
                current_chunk = {"heading": "", "content": [line]}
                current_size = line_size
            else:
                current_chunk["content"].append(line)
                current_size += line_size
        
        i += 1
    

    # Add remaining chunk
    if current_chunk["content"]:
        chunks.append({'heading': current_chunk["heading"], 'content': "\n".join(current_chunk["content"])})

    
    final_chunks = []
    for chunk in chunks:
        if len(chunk["content"]) > chunk_size:
            final_chunks.extend(_split_large_chunk(chunk, chunk_size))
            final_chunks[-1]["content"] = final_chunks[-1]["content"].rstrip()  # Remove trailing newline if exists
        else:
            final_chunks.append(chunk)
    
    return final_chunks


def _split_large_chunk(chunk, chunk_size=1200):
    """
    Split a large chunk using sentence (". "), word (" "), then character separators.
    """
    text = chunk["content"] if isinstance(chunk["content"], str) else "\n".join(chunk["content"])
    # Try sentence split first
    if ". " in text:
        parts = text.split(". ")
        return [{"heading": chunk["heading"], "content": part} for part in _group_parts_by_size(parts, ". ", chunk_size)]
    
    # Try word split
    if " " in text:
        parts = text.split(" ")
        return [{"heading": chunk["heading"], "content": part} for part in _group_parts_by_size(parts, " ", chunk_size)]
    
    # Character split (fallback)
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append({"heading": chunk["heading"], "content": text[i:i+chunk_size]})
    return chunks


def _group_parts_by_size(parts, separator, chunk_size):
    """
    Group parts back together until they fit within chunk_size.
    """
    if not parts:
        return []
    
    chunks = []
    current = []
    current_size = 0
    
    for part in parts:
        part_size = len(part) + len(separator)
        
        if current_size + part_size > chunk_size:
            if current:
                chunks.append(separator.join(current) + separator)
            current = [part]
            current_size = part_size
        else:
            current.append(part)
            current_size += part_size
    
    if current:
        chunks.append(separator.join(current))
    
    return chunks


def split_documents(docs, chunk_size=500, chunk_overlap=100):
    """
    Split documents while preserving headings with their content.
    
    Args:
        docs: List of documents from PyMuPDFLoader
        chunk_size: Maximum chunk size (default: 1000)
        chunk_overlap: Overlap between chunks (default: 150)
    
    Returns:
        List of split documents with metadata preserved
    """
    base_chunks = []  # tuples (doc, chunk_text)
    out_docs = []
    status = success
    message = 'Documents split successfully'
    try:
        for doc in docs:
            text = doc["page_content"]
            chunks = split_text_with_headings(text, chunk_size=chunk_size)  # from previous implementation
            
            for i, chunk in enumerate(chunks):
                base_chunks.append((doc, chunk, i, len(chunks)))

        # Apply overlap by prefixing each chunk (except first) with suffix of previous chunk
        
        if chunk_overlap and chunk_overlap > 0:
            overlapped = []
            for idx, (doc, chunk, i, total) in enumerate(base_chunks):
                if idx == 0:
                    overlapped.append((doc, chunk, i, total))
                    continue
                _, prev_chunk, _, _ = base_chunks[idx - 1]
                overlap_text = prev_chunk["content"][-chunk_overlap:] if len(prev_chunk["content"]) >= chunk_overlap else prev_chunk["content"]
                #print(f"Applying overlap of {len(overlap_text)} chars to chunk {i} of {total} - {chunk['content']}")
                new_chunk = overlap_text + chunk["content"]
                #print(f"Chunk {i} of {total} has size {len(chunk['content'])} chars, after overlap: {len(new_chunk)} chars")
                overlapped.append((doc, {"heading": chunk["heading"], "content": new_chunk}, i, total))
            base_chunks = overlapped
        
        # Convert into llama TextNodes, preserving original metadata
        out_docs = []
        for doc, chunk, i, total in base_chunks:
            if not chunk["content"].strip():
                continue
            metadata = dict(doc.get("metadata", {}) if isinstance(doc, dict) else (getattr(doc, "metadata", {}) or {}))
            metadata.update({"chunk_index": i, "chunk_total": total, "heading": chunk["heading"]})
            
            if i == 3:
                print(f"chunk[\"metadata\"]: {metadata.get('chunk_index', 'N/A')} ---- page_number {metadata.get('page_number', 'N/A')}, chunk[\"content\"]: {chunk['content']}")
            out_docs.append(TextNode(text=chunk["content"], metadata=metadata))
        
        return {"status": success, "message": message, "chunks": out_docs}     
    except Exception as e:
        print(f"Error splitting documents: {e}")
        return {"status": error, "message": "Error splitting documents", "chunks": None}