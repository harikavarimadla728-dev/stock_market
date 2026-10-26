# Stock Market Trend Classification

A Streamlit web app that preprocesses historical market data and trains a Backpropagation Neural Network (MLPClassifier) to classify the next market movement as **UP** or **DOWN_OR_FLAT**.

## Project files

- `stock_market_trend_app.py` — Streamlit application.
- `requirements.txt` — Python dependencies used by Streamlit Community Cloud.
- `Market.csv.zip` — market dataset archive. The app also lets you upload a CSV or ZIP file through the interface.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run stock_market_trend_app.py
   ```

5. In the app, upload `Market.csv.zip` using the file uploader.

## Deploy with GitHub and Streamlit Community Cloud

1. Extract this project ZIP on your computer.
2. Create a new repository on GitHub.
3. Upload the **contents of this extracted folder** to the repository (not just this ZIP file).
4. Open <https://share.streamlit.io/> and sign in with GitHub.
5. Choose **Create app**, select your repository and branch, and set the main file path to `stock_market_trend_app.py`.
6. Deploy the app. Streamlit Community Cloud installs packages from `requirements.txt`.

## Output dataset

The app's processed dataset contains exactly these 10 columns:

`Index`, `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`, `Daily_Return`, `Target`.

## Note

This is an educational machine-learning project, not financial advice. Model results on historical data do not guarantee future performance.
