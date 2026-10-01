# RescuePaws AI

An AI assistant for a volunteer-run animal shelter caring for 500+ rescued animals.

**Problem:** Animal information is scattered across volunteer notes, and matching
adopters with suitable animals takes a lot of manual effort.

**MVP scope:** structured animal profiles, semantic adopter–animal matching,
policy Q&A grounded in shelter documents (RAG), and an agent that picks the right
tool for each request.

**Safety principle:** The system never guesses health, temperament, or
child/cat/dog compatibility. Anything not recorded stays "unknown".

## Setup
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    pytest
