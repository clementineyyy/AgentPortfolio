# Stock Portfolio Agent

This project shows how to create an AI agent that streams the portfolio analysis workflow in real-time. Built with CrewAI (backend), React/Next.js (frontend), the CopilotKit's AG-UI Protocol enables users to watch the agent fetch stock data, calculate allocations, and generate insights live.

**Tech stack:**
- [React](https://react.dev) + [Next.js](https://nextjs.org) for the frontend UI
- [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) for the backend API
- [CopilotKit](https://github.com/CopilotKit/CopilotKit) + AG UI Protocol for streaming agent events
- [CrewAI](https://github.com/crewAIInc/crewAI) for the agent workflow
- [yfinance](https://github.com/ranaroussi/yfinance) and [pandas](https://pandas.pydata.org) for market data and analysis

## Setup

1. **Install Dependencies**:
   ```bash
   uv sync
   
   # Install frontend dependencies
   cd frontend
   npm install
   cd ..
   ```

2. **Environment Variables**:
   Copy each example file to `.env` in the same directory, then put your key only in those local `.env` files. `.env` files are ignored by Git; commit the `.env.example` templates instead.
   ```powershell
   Copy-Item agent/.env.example agent/.env
   Copy-Item frontend/.env.example frontend/.env
   ```
   - `agent/.env` (backend):
     ```env
     OPENAI_API_KEY=your-key
     OPENAI_BASE_URL=
     OPENAI_MODEL=gpt-4o-mini
     ```
   For SiliconFlow, set `OPENAI_BASE_URL=https://api.siliconflow.cn/v1` and use a model ID enabled for your account, such as `deepseek-ai/DeepSeek-V3`. The agent uses an OpenAI-compatible API; the API key must belong to the provider configured by `OPENAI_BASE_URL`.
   - `frontend/.env` (frontend):
     ```env
     OPENAI_API_KEY=your-openai-key
     NEXT_PUBLIC_CREWAI_URL=http://127.0.0.1:8000/crewai-agent
     ```
   `OPENAI_API_KEY` in the frontend file is read by the Next.js server route. Do not rename it to `NEXT_PUBLIC_OPENAI_API_KEY`: Next.js exposes `NEXT_PUBLIC_*` values to browser code.

3. **Configure Backend URL (optional)**:
   The frontend expects the backend to run locally. If you change host/port, update the frontend API call configuration accordingly.

4. **Run the App**:
   ```bash
   # Start backend
   uv run python agent/main.py

   # In another terminal, start frontend
   cd frontend
   npm run dev
   ```

## Usage

1. **Open the UI**:
   - Visit `http://localhost:3000`.

2. **Run a Stock Analysis**:
   - Use the prompt/input in the UI to ask for a portfolio analysis (e.g., "Analyze AAPL and MSFT with $10k each").
   - Watch live progress: messages, tool calls, and intermediate results stream into the UI via AG UI Protocol events.

3. **Review Results**:
   - View allocation summaries, charts, and insights rendered by the frontend components.

## Development Notes
- If your editor reports missing imports, ensure it points to the same Python environment where you installed dependencies (`uv`, `venv`, `conda`, etc.). Running `uv sync` in the repo root is recommended, but any standard Python environment manager works.
- The FastAPI app is in `agent/main.py`; core workflow logic is in `agent/stock_analysis.py`.
- In `frontend`, run `npm run dev` while developing. Stop the dev server before running `npm run build`; both Next.js commands write to the same `.next` directory and must not run at the same time. If a Turbopack runtime module is missing, stop every frontend dev server, move `frontend/.next` aside, then restart `npm run dev` to regenerate the cache.
- Before submitting frontend changes, run `npm run typecheck`, `npm run lint`, `npm test`, and `npm run build` from `frontend`, one at a time.



## 📬 Stay Updated with Our Newsletter!

**Get a FREE Data Science eBook** 📖 with 150+ essential lessons in Data Science when you subscribe to our newsletter! Stay in the loop with the latest tutorials, insights, and exclusive resources. [Subscribe now!](https://join.dailydoseofds.com)

[![Daily Dose of Data Science Newsletter](https://github.com/patchy631/ai-engineering/blob/main/resources/join_ddods.png)](https://join.dailydoseofds.com)

## Contribution
Contributions are welcome! Please open an issue or submit a PR.
