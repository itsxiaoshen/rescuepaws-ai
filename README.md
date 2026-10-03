# RescuePaws AI

An AI assistant for a volunteer-run animal shelter caring for 500+ rescued animals.

**Problem:** Animal information is scattered across volunteer notes, and matching
adopters with suitable animals takes a lot of manual effort.

**MVP scope:** structured animal profiles, semantic adopter–animal matching,
policy Q&A grounded in shelter documents (RAG), and an agent that picks the right
tool for each request.

**Safety principle:** The system never guesses health, temperament, or
child/cat/dog compatibility. Anything not recorded stays "unknown".

## Run locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add your OPENAI_API_KEY
pytest                        # no API key needed: LLM calls are faked in tests
uvicorn app.api:app --reload  # open http://localhost:8000 (chat) or /docs (API)
```

## Run with Docker

```bash
docker build -t rescuepaws-ai .
docker run -p 8000:8000 --env-file .env rescuepaws-ai
```
