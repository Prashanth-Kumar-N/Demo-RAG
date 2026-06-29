"""
Table-aware document chunking with heading preservation.

Strategies:
- Detect headings (h1-h6 style, numbered, caps, etc.)
- Detect tables (lines of pipes |, rows with aligned cols)
- Keep headings with their following content
- Keep tables intact with surrounding context
- 15% overlap between chunks for context continuity
- Chunk size: 750 characters
"""

import re
from typing import List, Dict, Any
from llama_index.core.schema import TextNode
from src.constants import success, error


def is_heading(line: str) -> bool:
    """
    Detect if a line is a section heading.
    Heuristics:
    - Short line (< 100 chars)
    - Markdown style: #, ##, ###, etc.
    - All caps or Title Case with no period
    - Line starting with numbers (1., 2., etc.)
    - Ends with colon
    """
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    
    # Markdown headings
    if stripped.startswith('#'):
        return True
    
    # Numbered headings (1., 1.1., etc.)
    if re.match(r'^\d+(\.\d+)*\s+', stripped):
        return True
    
    # All caps (but not single words)
    if stripped.isupper() and len(stripped.split()) > 1:
        return True
    
    # Title case with no period at end
    if not stripped.endswith('.') and stripped[0].isupper():
        words = stripped.split()
        if len(words) >= 2 and sum(1 for w in words if w[0].isupper()) >= len(words) - 1:
            return True
    
    # Ends with colon
    if stripped.endswith(':'):
        return True
    
    return False


def is_table_line(line: str) -> bool:
    """
    Detect if a line is part of a table.
    Heuristics:
    - Contains pipe characters (|) with content around them
    - Multiple pipes with consistent alignment
    """
    stripped = line.strip()
    if not stripped or '|' not in stripped:
        return False
    
    # Count pipes - tables typically have 2+ pipes
    pipe_count = stripped.count('|')
    if pipe_count < 2:
        return False
    
    # Check if it looks structured (not just random pipes)
    parts = stripped.split('|')
    # At least 2 substantial content areas between pipes
    substantial = sum(1 for p in parts if len(p.strip()) > 0)
    return substantial >= 2


def detect_table_block(lines: List[str], start_idx: int) -> tuple:
    """
    Given a line index that starts a table, find the end of the table block.
    Tables are contiguous lines with pipes and structure.
    Returns (start_idx, end_idx) inclusive.
    """
    end_idx = start_idx
    for i in range(start_idx + 1, len(lines)):
        if not is_table_line(lines[i]):
            break
        end_idx = i
    return (start_idx, end_idx)


def chunk_with_overlap(text: str, chunk_size: int = 750, overlap_percent: float = 0.15) -> List[str]:
    """
    Split text into chunks of size `chunk_size` with overlapping context.
    Overlap is calculated as a percentage of chunk_size.
    """
    if not text or len(text) <= chunk_size:
        return [text]
    
    overlap_size = int(chunk_size * overlap_percent)
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        
        # If we've reached the end, break
        if end >= len(text):
            break
        
        # Move start position: full chunk size minus overlap
        new_start = end - overlap_size
        
        # Ensure progress: if new_start didn't move significantly, jump to end
        if new_start <= start:
            new_start = end
        
        start = new_start
    
    return chunks


def split_documents(docs: List[Dict[str, Any]], chunk_size: int = 750, overlap_percent: float = 0.15) -> Dict[str, Any]:
    """
    Split documents while preserving headings, tables, and metadata.
    
    Args:
        docs: List of documents from fitz loader, each with structure:
              {"page_content": text, "metadata": {"file_name": str, "page_number": int}}
        chunk_size: Target chunk size in characters (default: 750)
        overlap_percent: Overlap as percentage of chunk_size (default: 0.15 = 15%)
    
    Returns:
        {
            "status": "success" | "error",
            "message": str,
            "chunks": List[TextNode] with text and metadata
        }
    """
    try:
        all_chunks = []
        status = success
        message = 'Documents split successfully'
        
        for i, doc in enumerate(docs):
            page_content = doc.get("page_content", "")
            doc_metadata = doc.get("metadata", {})
            
            if not page_content.strip():
                continue
            
            # Parse document structure and apply chunking
            chunks = _smart_split_with_structure(
                page_content,
                chunk_size=chunk_size,
                overlap_percent=overlap_percent
            )
            
            # Convert to TextNodes with enriched metadata
            for chunk_idx, chunk_text in enumerate(chunks):
                if not chunk_text.strip():
                    continue
                
                metadata = dict(doc_metadata)
                chunk_id = f"{metadata.get("file_name", "unknown")}_{metadata.get("page_number", "unknown")}_chunk_{chunk_idx + 1}"
                metadata.update({
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx + 1,
                    "chunk_total": len(chunks),
                    "chunk_size": chunk_size
                })
                
                text_node = TextNode(text=chunk_text, metadata=metadata)
                all_chunks.append(text_node)
        # each chunk has metadata: file_name, page_number, chunk_index, chunk_total, chunk_size, and any original metadata from the document
        # {'file_name': '4_Hydraulic_components_Catalog.pdf', 'page_number': 71, 'chunk_index': 1, 'chunk_total': 2, 'chunk_size': 750}
        return {
            "status": status,
            "message": message,
            "chunks": all_chunks
        }
    
    except Exception as e:
        print(f"Error splitting documents: {e}")
        import traceback
        traceback.print_exc()
        return {
            "status": error,
            "message": f"Error splitting documents: {str(e)}",
            "chunks": None
        }


def _smart_split_with_structure(text: str, chunk_size: int = 750, overlap_percent: float = 0.15) -> List[str]:
    """
    Intelligently split text while preserving structure (headings, tables).
    
    Strategy:
    1. Split by lines and parse structure
    2. Group headings with their content
    3. Keep tables intact
    4. Chunk logically, respecting structure boundaries
    5. Apply overlap for context
    """
    lines = text.split('\n')
    
    # Build structure: identify headings and tables
    structure = []  # List of (type, content, line_indices)
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if is_table_line(line):
            # Found a table - collect all consecutive table lines
            table_start, table_end = detect_table_block(lines, i)
            table_content = '\n'.join(lines[table_start:table_end + 1])
            structure.append(("table", table_content, (table_start, table_end)))
            i = table_end + 1
        elif is_heading(line):
            # Found a heading - collect with following content until next heading/table
            heading_text = line
            heading_start = i
            i += 1
            
            # Collect content lines following this heading
            body_lines = []
            while i < len(lines):
                if is_heading(lines[i]) or is_table_line(lines[i]):
                    break
                body_lines.append(lines[i])
                i += 1
            
            body_text = '\n'.join(body_lines).strip()
            section_content = heading_text + '\n' + body_text if body_text else heading_text
            structure.append(("section", section_content, (heading_start, i - 1)))
        else:
            # Regular content
            content_lines = [line]
            content_start = i
            i += 1
            
            # Collect consecutive non-special lines
            while i < len(lines):
                if is_heading(lines[i]) or is_table_line(lines[i]):
                    break
                content_lines.append(lines[i])
                i += 1
            
            content_text = '\n'.join(content_lines)
            structure.append(("content", content_text, (content_start, i - 1)))
    
    # Now apply chunking with overlap
    chunks = _apply_chunking_with_structure(
        structure, chunk_size=chunk_size, overlap_percent=overlap_percent
    ) 

    return chunks


def _apply_chunking_with_structure(
    structure: List[tuple], chunk_size: int = 750, overlap_percent: float = 0.15
) -> List[str]:
    """
    Apply chunking to structured elements, respecting boundaries.
    
    If a section/table fits in the remaining space, add it.
    If it doesn't, start a new chunk.
    Apply overlap between chunks.
    """
    if not structure:
        return []
    
    overlap_size = int(chunk_size * overlap_percent)
    chunks = []
    current_chunk = ""
    
    for elem_type, elem_content, _ in structure:
        elem_len = len(elem_content)
        
        if not current_chunk:
            # Start a new chunk
            current_chunk = elem_content
        elif len(current_chunk) + 1 + elem_len <= chunk_size:
            # Fits in current chunk
            current_chunk += '\n' + elem_content
        else:
            # Doesn't fit - save current and start new
            if current_chunk.strip():
                chunks.append(current_chunk)
            
            # If element itself is larger than chunk_size, split it
            if elem_len > chunk_size:
                sub_chunks = chunk_with_overlap(elem_content, chunk_size, overlap_percent)
                chunks.extend(sub_chunks)
                # Last sub_chunk becomes new current
                current_chunk = sub_chunks[-1] if sub_chunks else ""
            else:
                current_chunk = elem_content
    
    # Add remaining chunk
    if current_chunk.strip():
        chunks.append(current_chunk)
    
    # Apply overlap: prepend tail of previous chunk to current
    if len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-overlap_size:] if len(chunks[i - 1]) >= overlap_size else chunks[i - 1]
            overlapped_chunk = prev_tail + '\n' + chunks[i]
            overlapped.append(overlapped_chunk)
        chunks = overlapped
    
    return chunks
