name: Index RSI 15m Scan

on:
  schedule:
    - cron: "45 3 * * 1-5"
    - cron: "*/15 4-9 * * 1-5"
    - cron: "0 10 * * 1-5"
  workflow_dispatch:

jobs:
  scan-indices:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install numpy pandas requests yfinance

      - name: Run index RSI scan
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: python screener.py --mode indices
name: F&O Momentum & Value 3:15 PM IST

on:
  schedule:
    - cron: "45 9 * * 1-5"
  workflow_dispatch:

jobs:
  scan-fno:
    runs-on: ubuntu-latest
    timeout-minutes: 25

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install numpy pandas requests yfinance

      - name: Run F&O Momentum and Value Buy scan
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: python screener.py --mode fno
