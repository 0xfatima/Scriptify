## Offline Spell + Grammar + RAG (Desktop)

### Run (Windows)

- **Option A (recommended)**: run using the project venv

```bash
".\.venv\Scripts\python.exe" ".\app.py"
```

- **Option B**: activate venv then run

```bash
.\.venv\Scripts\Activate.ps1
python .\app.py
```

### What’s implemented

- **ChatGPT-style UI**: sidebar + chat bubbles + copy buttons
- **Modes**: `General`, `Email`, `Academic`, `LaTeX`, `Doc Q&A` (Doc Q&A wiring is stubbed in UI for now)
- **Doc upload + RAG (local)**: upload a PDF → background indexing → ask questions in `Doc Q&A` mode (answers are restricted to retrieved context)
- **Dark/Light theme toggle**: matches your palette
- **Chat history**: saved locally in `%USERPROFILE%\.spell_grammar_offline\chat_history\`
- **History limit**: keep only the last N visible prompts per chat (configurable)
- **Smooth UI**: LLM + spell/grammar checks run in background threads
- **Live highlighting**: misspellings/heuristics underline in light red/orange while typing (debounced)

### Offline notes

- The Qwen model is loaded from `.\model` using `local_files_only=True`.
- For spell suggestions, SymSpell is used **only if** you provide a frequency dictionary at:
  - `.\data\frequency_dictionary_en_82_765.txt`
  If you don’t, the app still runs with lightweight heuristic checks.

