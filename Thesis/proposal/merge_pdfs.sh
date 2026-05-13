#!/bin/bash
# Script to merge Hebrew abstract with thesis proposal using pdflatex

cd "$(dirname "$0")"

# Create backup of original thesis proposal first (before we modify it)
# If backup doesn't exist, create it from current thesis_proposal.pdf
# If it exists but thesis_proposal.pdf is larger (might be already merged), restore from backup first
if [ ! -f "thesis_proposal_backup.pdf" ]; then
    cp thesis_proposal.pdf thesis_proposal_backup.pdf
    echo "Created backup: thesis_proposal_backup.pdf"
elif [ -f "thesis_proposal_backup.pdf" ]; then
    # Restore from backup to ensure we're working with original
    cp thesis_proposal_backup.pdf thesis_proposal.pdf
    echo "Restored original from backup for merge"
fi

# Compile the merge LaTeX file (references thesis_proposal_backup.pdf for pages 3-end)
echo "Compiling merge_hebrew.tex..."
pdflatex -interaction=nonstopmode merge_hebrew.tex > merge_hebrew_compile.log 2>&1

if [ -f "merge_hebrew.pdf" ]; then
    # Replace thesis_proposal.pdf with merged version
    mv merge_hebrew.pdf thesis_proposal.pdf
    echo "Successfully merged PDFs!"
    echo "Hebrew abstract pages (1-2) + thesis proposal pages (3-end) = thesis_proposal.pdf"
    
    # Clean up auxiliary files
    rm -f merge_hebrew.aux merge_hebrew.log merge_hebrew.out merge_hebrew_compile.log
else
    echo "Error: merge_hebrew.pdf was not created. Check merge_hebrew_compile.log for errors."
    exit 1
fi

