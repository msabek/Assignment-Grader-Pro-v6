import PyPDF2
import io
import zipfile
import xml.etree.ElementTree as ET
import re
import streamlit as st
import time

class FileProcessor:
    def __init__(self):
        self._cache = {}
        self._max_retries = 3
        self._chunk_size = 1024 * 1024  # 1MB chunks

    @st.cache_data
    def extract_text(_self, file):
        """Extract text from uploaded file based on its type with caching and progress tracking."""
        try:
            file_hash = hash(file.read())
            file.seek(0)  # Reset file pointer
            
            # Check cache
            if file_hash in _self._cache:
                return _self._cache[file_hash]

            # Get file extension from filename
            file_ext = file.name.lower().split('.')[-1]
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            result = None
            for attempt in range(_self._max_retries):
                try:
                    if file_ext == 'pdf':
                        result = _self._extract_from_pdf(file, progress_bar, status_text)
                    elif file_ext == 'txt':
                        result = _self._extract_from_txt(file, progress_bar, status_text)
                    elif file_ext == 'docx':
                        result = _self._extract_from_docx(file, progress_bar, status_text)
                    else:
                        raise ValueError(f"Unsupported file format: {file_ext}")
                    break
                except Exception as e:
                    if attempt == _self._max_retries - 1:
                        raise
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
            
            # Cache the result
            _self._cache[file_hash] = result
            return result

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            raise
        finally:
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()

    def extract_student_id(self, filename: str) -> str:
        """Extract student ID from filename."""
        try:
            # Common patterns for student IDs in filenames
            patterns = [
                r'^(\d{6,10})[\s_-]',  # Numbers at start followed by separator
                r'(?:id|student|number)[\s_-]?(\d{6,10})',  # ID/student/number followed by numbers
                r'[\s_-](\d{6,10})\.pdf$'  # Numbers before .pdf extension
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename, re.IGNORECASE)
                if match:
                    return match.group(1)
            
            # If no match found, extract first sequence of 6-10 digits
            digits_match = re.search(r'(\d{6,10})', filename)
            if digits_match:
                return digits_match.group(1)
            
            # If no valid student ID found
            return "unknown"
            
        except Exception as e:
            print(f"Error extracting student ID: {str(e)}")
            return "unknown"

    def _extract_from_pdf(self, file, progress_bar, status_text):
        """Extract text from PDF with progress tracking and streaming."""
        text = []
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        
        for i, page in enumerate(pdf_reader.pages):
            text.append(page.extract_text())
            progress = (i + 1) / total_pages
            progress_bar.progress(progress)
            status_text.text(f"Processing page {i + 1}/{total_pages}")
            
        return " ".join(text)

    def _extract_from_txt(self, file, progress_bar, status_text):
        """Extract text from text file with progress tracking."""
        text = file.read().decode('utf-8')
        progress_bar.progress(1)
        status_text.text("Processing text file")
        return text.strip()

    def _extract_from_docx(self, docx_file, progress_bar, status_text):
        """Extract text from Word document using zipfile with progress tracking."""
        try:
            # Read the file into memory
            docx_bytes = io.BytesIO(docx_file.read())
            
            # Open as zip file
            doc = zipfile.ZipFile(docx_bytes)
            
            # Read the main document content
            xml_content = doc.read('word/document.xml')
            
            # Parse XML
            tree = ET.fromstring(xml_content)
            
            # Extract text from all paragraphs (removing XML namespace)
            ns = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
            paragraphs = []
            
            for paragraph in tree.findall(f'.//{ns}p'):
                texts = []
                for node in paragraph.findall(f'.//{ns}t'):
                    if node.text:
                        texts.append(node.text)
                if texts:
                    paragraphs.append(''.join(texts))
            
            progress_bar.progress(1)
            status_text.text("Processing Word document")
            return '\n'.join(paragraphs)
            
        except Exception as e:
            raise Exception(f"Error processing Word document: {str(e)}")

    def process_text(self, text):
        """Process extracted text using basic text processing."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        # Remove common stop words
        stop_words = {'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
                     'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
                     'that', 'the', 'to', 'was', 'were', 'will', 'with'}
        words = text.split()
        processed_words = [word for word in words if word not in stop_words]
        
        return ' '.join(processed_words)
