# TA Learning

This repository contains a small collection of scripts for exploring
technical analysis concepts.  The project loads cryptocurrency price data,
computes a variety of indicators and demonstrates simple trading
strategies.  Visualizations of the signals and a very light‑weight back
testing routine are also included.

## Requirements

The project only depends on a few common Python libraries.  They can be
installed with

```bash
pip install -r requirements.txt
```

## Data

Example minute data for Bitcoin and Ethereum is provided in the `data/`
folder.  Each CSV file contains a `date` column and price information used
by the scripts.

## Usage

The main entry point is `main.py`.  By default it reads `config.yaml` which
controls which file to load and the parameters for single or multi time
frame analysis.  The script can be run with

```bash
python main.py            # uses config.yaml
python main.py -c my_config.yaml
```

The output charts the selected strategy and prints logging information via
`rich` and `loguru`.

## Configuration

`config.yaml` contains two main sections:

- `general`: data file, date range and whether multi‑time‑frame analysis is
enabled.
- `single_tf` and `multi_tf`: parameters for the included strategies.

Feel free to edit this file or provide your own when running the script.



## API Server

Run the FastAPI server to expose strategy endpoints:

```bash
uvicorn server.app:app --reload
```

All endpoints require an API key. Create a `.env` file alongside the
application and define your keys:

```bash
API_KEY=your-secret-key
CRYPTOPANIC_API_KEY=your-news-api-key
```

### News endpoint

The `/news` route aggregates recent headlines from CryptoPanic for BTC and
ETH.  Ensure a `CRYPTOPANIC_API_KEY` is available in a `.env` file before
running:

```bash
curl http://localhost:8000/news
```
