# Study Assistant: architecture and setup guide

## Step 1: Architecture

```
Phone browser (Streamlit UI: chat box + mic)
        |
        v
Streamlit app (app.py, hosted free)
  1. Voice  -> OpenAI Whisper / gpt-4o-transcribe (speech-to-text)
  2. Query rewrite (cheap LLM: Tamil/Tanglish -> English search query, or NONE)
  3. Tavily Search API -> 5 numbered sources
  4. Main LLM (Claude or OpenAI) + system prompt + sources -> answer with [1][2] citations
  5. Renders text, Graphviz/Mermaid diagrams, JSON charts, Sources expander
```

| Component | Service | Cost (approx.) |
|---|---|---|
| LLM | Anthropic (default) or OpenAI | pay per use, pennies per question |
| Speech-to-text | OpenAI Whisper API (`whisper-1`) | ~$0.006 / minute |
| Web search | Tavily | free tier ~1,000 searches/month |
| Hosting | Streamlit Community Cloud | free |

Why it cannot automate anything: the app registers no LLM tools, runs no shell, and reads/writes no files.
The only outward call besides the LLM and STT is a read-only search. The system prompt adds a second layer.

Note on the "non-neutral" requirement: the prompt makes it decisive on facts, logic, and argument quality,
and still presents both sides on genuinely value-driven disputes. Forcing a verdict there would make it
confidently wrong, which is the opposite of rigorous.

## Step 2: System prompt

It lives in `app.py` as `SYSTEM_PROMPT`. Edit it there.

## Step 3: Files

- `app.py`: full application
- `requirements.txt`: dependencies

## Step 4: Setup and free hosting

### A. Get API keys
1. Anthropic: https://console.anthropic.com -> API Keys (add a few dollars of credit).
2. OpenAI (for speech-to-text, and optionally as the LLM): https://platform.openai.com -> API keys.
3. Tavily: https://tavily.com -> sign up -> API key.

### B. Run locally first
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...  OPENAI_API_KEY=...  TAVILY_API_KEY=...
streamlit run app.py
```
Open http://localhost:8501. Test in Tamil, Tanglish, English, plus "draw a flowchart of photosynthesis".

### C. Deploy on Streamlit Community Cloud (free)
1. Create a **private or public GitHub repo** with `app.py` and `requirements.txt`. Never commit keys.
2. Go to https://share.streamlit.io -> sign in with GitHub -> **New app** -> pick repo, branch, `app.py`.
3. **Advanced settings -> Secrets**, paste:
```toml
ANTHROPIC_API_KEY = "sk-ant-..."
OPENAI_API_KEY = "sk-..."
TAVILY_API_KEY = "tvly-..."
APP_PASSWORD = "choose-a-long-password"
# optional:
# LLM_PROVIDER = "openai"
# STT_MODEL = "gpt-4o-transcribe"
```
4. Deploy. You get a URL like `https://yourname-study.streamlit.app`.
5. On your phone: open the URL -> browser menu -> **Add to Home Screen**. Allow microphone access when asked.

`APP_PASSWORD` matters: without it anyone with the URL can spend your API credits.

### D. Alternative: Render (free web service)
- New Web Service -> connect repo -> Build: `pip install -r requirements.txt`
- Start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
- Add the same keys as environment variables. The free tier sleeps when idle (slow first load).

### Troubleshooting
- Mic widget missing: upgrade Streamlit (`st.audio_input` needs >= 1.40) and use HTTPS (the hosted URL is).
- Tanglish transcribed poorly: set `STT_MODEL = "gpt-4o-transcribe"`.
- Model name errors: set `MAIN_MODEL` and `FAST_MODEL` to models your account can use.
- Diagram fails to parse: the app falls back to showing the raw code; just ask it to redo the diagram.

### Optional upgrades
Save chats to a file or database, add a "quiz me" mode, upload a PDF for Q&A, or text-to-speech output.
