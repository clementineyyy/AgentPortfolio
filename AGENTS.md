## Run the App

The backend and frontend servers are already running and available for debugging. If test results report that a server is stopped, restart it with these commands, then continue testing:

```bash
# Start backend
uv run python agent/main.py

# In another terminal, start frontend
cd frontend
npm run dev
```

## Agent skills

### Issue tracker

Issues live in GitHub Issues for `clementineyyy/AgentPortfolio`. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage roles map to the repo's GitHub labels. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context domain docs live in the root `GLOSSARY.md` and `docs/adr/`. See `docs/agents/domain.md`.
