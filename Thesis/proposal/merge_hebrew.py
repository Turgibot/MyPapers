#!/usr/bin/env python3
"""
Script to replace first 2 pages of thesis_proposal.pdf with hebrew_abstract.pdf
"""

import sys
import os

try:
    import PyPDF2
    PDF_LIB = 'PyPDF2'
except ImportError:
    try:
        import pypdf
        PDF_LIB = 'pypdf'
    except ImportError:
        print("Error: Neither PyPDF2 nor pypdf is installed.")
        print("Please install one: pip install PyPDF2 or pip install pypdf")
        sys.exit(1)

def merge_pdfs():
    hebrew_pdf = 'hebrew_abstract.pdf'
    thesis_pdf = 'thesis_proposal.pdf'
    output_pdf = 'thesis_proposal.pdf'
    backup_pdf = 'thesis_proposal_backup.pdf'
    
    # Check if files exist
    if not os.path.exists(hebrew_pdf):
        print(f"Error: {hebrew_pdf} not found!")
        sys.exit(1)
    
    # Use backup if it exists, otherwise use current thesis_proposal.pdf
    import shutil
    if os.path.exists(backup_pdf):
        source_pdf = backup_pdf
        print(f"Using backup: {backup_pdf}")
    else:
        if not os.path.exists(thesis_pdf):
            print(f"Error: {thesis_pdf} not found!")
            sys.exit(1)
        # Create backup
        shutil.copy2(thesis_pdf, backup_pdf)
        print(f"Created backup: {backup_pdf}")
        source_pdf = thesis_pdf
    
    # Open PDFs
    with open(hebrew_pdf, 'rb') as hebrew_file, open(source_pdf, 'rb') as thesis_file:
        if PDF_LIB == 'PyPDF2':
            hebrew_reader = PyPDF2.PdfReader(hebrew_file)
            thesis_reader = PyPDF2.PdfReader(thesis_file)
            writer = PyPDF2.PdfWriter()
        else:
            hebrew_reader = pypdf.PdfReader(hebrew_file)
            thesis_reader = pypdf.PdfReader(thesis_file)
            writer = pypdf.PdfWriter()
        
        # Add all pages from Hebrew PDF (first 2 pages)
        print(f"Adding {len(hebrew_reader.pages)} pages from Hebrew abstract...")
        for page in hebrew_reader.pages:
            writer.add_page(page)
        
        # Add pages 3 onwards from thesis proposal (skip first 2 pages)
        total_pages = len(thesis_reader.pages)
        print(f"Adding pages 3-{total_pages} from thesis proposal...")
        for i in range(2, total_pages):  # Start from index 2 (page 3)
            writer.add_page(thesis_reader.pages[i])
        
        # Write merged PDF
        with open(output_pdf, 'wb') as output_file:
            writer.write(output_file)
        
        print(f"Successfully merged PDFs. Output: {output_pdf}")
        print(f"Total pages in new PDF: {len(hebrew_reader.pages) + (total_pages - 2)}")

if __name__ == '__main__':
    merge_pdfs()

