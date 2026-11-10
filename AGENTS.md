## Run the App

The backend and frontend servers are already running and available for debugging. If test results report that a server is stopped, restart it with these commands, then continue testing:

```bash
# Start backend
uv run python agent/main.py

# In another terminal, start frontend
cd frontend
npm run dev
```

The host's Windows PowerShell has `uv` installed. Use `uv run` for backend startup and tests so they use the same project environment. If a tool shell cannot resolve `uv`, inspect its command lookup and `PATH` (for example, `Get-Command uv`) and resolve the host's `uv` executable in that shell before running backend commands.

## Tests and validation

- Before ending each turn that changes code, prompts, configuration, or behavior, run the tests or validation script that covers the change. Update or add a regression test when behavior changes.
- Run the backend suite from the repository root with `uv run python -m unittest discover -s agent/tests -p "test_*.py"`.
- For request-routing or prompt-classification changes, also run `uv run python agent/evaluate_query_intents.py` when model credentials are configured.
- Model-backed validation scripts such as `evaluate_query_intents.py` are authorized to send their defined test cases to the configured model service; run them when relevant to the change without requesting separate approval. Keep requests within the test fixtures and never print credentials.
- For frontend changes, follow the applicable checks in `README.md`.
- If a relevant check cannot run, state the specific blocker and the checks that did run; report only verified results.

## Agent skills

### Issue tracker

Issues live in GitHub Issues for `clementineyyy/AgentPortfolio`. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage roles map to the repo's GitHub labels. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context domain docs live in the root `GLOSSARY.md` and `docs/adr/`. See `docs/agents/domain.md`.
