# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A single-page Streamlit app that downloads historical stock data via `yfinance`, trains a Facebook Prophet time-series model on the fly, and displays both raw data and a forecast using Plotly. The UI is in French. It is deployed on Heroku at `https://prophet-2.herokuapp.com/`.

## Commands

**Run locally:**
```bash
pip install -r requirements.txt
streamlit run main.py
```

**Heroku startup (mirrors the Procfile):**
```bash
sh setup.sh && streamlit run main.py
```
`setup.sh` writes `~/.streamlit/config.toml` using the `$PORT` environment variable — required for Heroku but not needed for local development.

There are no tests and no linter configuration in this project.

## Architecture

Everything lives in `main.py`. The app flow is top-to-bottom Streamlit execution:

1. User picks a stock ticker (AAPL, AMZN, WMT, NFLX, MAR, AAL) and a forecast horizon in days (1–365).
2. `load_data()` fetches OHLCV data from `2015-01-01` to today via `yfinance` and is memoised with `@st.cache` (the legacy decorator — **not** `@st.cache_data`).
3. Raw open/close prices are plotted with Plotly.
4. A Prophet model is fitted on the `Close` price, then `make_future_dataframe` is called with `periods = 48 * n_days` at `freq='30min'` to produce sub-daily forecasts.
5. The forecast tail and a Plotly forecast figure are rendered.

## Key Details

- **Python version:** 3.7.9 (pinned in `runtime.txt`).
- **Cache decorator:** Uses the deprecated `@st.cache` — upgrading to `@st.cache_data` requires Streamlit ≥ 1.18 and a Python version bump.
- **Side effect:** `data.to_csv('out.csv')` is called on every run, writing to the working directory.
- **Forecast granularity:** Predictions are generated at 30-minute intervals for the full forecast window, which can be slow for large `n_days` values.
- **No `.env` / secret management:** The app uses only public Yahoo Finance data; no API keys are required.
