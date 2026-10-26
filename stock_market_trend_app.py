import io
import zipfile
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(
    page_title="Stock Market Trend Classification",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main {background-color: #f7f9fc;}
      .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
      div[data-testid="stMetric"] {
        background: white; padding: 14px; border-radius: 12px;
        border: 1px solid #e6eaf0;
      }
      .hero {
        padding: 24px; border-radius: 16px; color: white;
        background: linear-gradient(120deg, #172554, #2563eb);
        margin-bottom: 20px;
      }
      .hero h1 {margin: 0; font-size: 2rem;}
      .hero p {margin: 8px 0 0 0; color: #e0e7ff;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>Stock Market Trend Classification</h1>
      <p>Machine Learning Project | Historical market data + Backpropagation Neural Network</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    "This application cleans historical stock/index data, creates technical features, "
    "and trains a neural network to classify whether the next available closing price "
    "will rise or remain the same/fall."
)

with st.sidebar:
    st.header("Project Settings")
    st.caption("Algorithm: Multi-Layer Perceptron (Backpropagation)")
    hidden_layer_size = st.selectbox(
        "Hidden layer architecture",
        options=["32", "64", "64, 32", "128, 64"],
        index=2,
    )
    max_iter = st.slider("Training iterations", min_value=100, max_value=600, value=300, step=50)
    test_fraction = st.slider(
        "Test data percentage", min_value=10, max_value=30, value=20, step=5
    )
    st.caption("The split is chronological, not random, to respect time order.")

uploaded_file = st.file_uploader(
    "Upload Market.csv or Market.csv.zip",
    type=["csv", "zip"],
    help="Expected columns include Index, Date, Open, High, Low, Close, Adj Close, and Volume.",
)


@st.cache_data(show_spinner="Reading market data...")
def load_data(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
            if not csv_names:
                raise ValueError("The ZIP archive does not contain a CSV file.")
            # Prefer a file named Market.csv when available.
            chosen = next((n for n in csv_names if Path(n).name.lower() == "market.csv"), csv_names[0])
            with zf.open(chosen) as f:
                return pd.read_csv(f, low_memory=False)
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)


def normalise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result.columns = [str(c).strip() for c in result.columns]
    # Standardise common variants while preserving Index.
    rename = {}
    for col in result.columns:
        key = col.strip().lower().replace("_", " ")
        if key in {"adj close", "adjusted close", "adjclose"}:
            rename[col] = "Adj Close"
        elif key == "date":
            rename[col] = "Date"
        elif key == "index":
            rename[col] = "Index"
        elif key in {"open", "high", "low", "close", "volume"}:
            rename[col] = key.title()
    return result.rename(columns=rename)


def preprocess_and_engineer(raw: pd.DataFrame):
    """Clean the dataset and keep only 10 user-facing columns.

    Internal helper columns Next_Close and Next_Date are retained temporarily for
    target creation and time-aware splitting, but are not included in the 10-column output.
    """
    data = normalise_column_names(raw)
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if "Index" not in data.columns:
        data["Index"] = "Market"
    if "Adj Close" not in data.columns:
        data["Adj Close"] = data["Close"]

    original_rows = len(data)
    data = data.drop_duplicates().copy()
    duplicates_removed = original_rows - len(data)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce", dayfirst=False)
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["Index"] = data["Index"].astype("string").str.strip()
    data.loc[data["Index"].isna() | data["Index"].eq(""), "Index"] = "Unknown"

    data = data.dropna(subset=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
    data = data[(data[["Open", "High", "Low", "Close", "Adj Close"]] > 0).all(axis=1)]
    data = data[data["Volume"] >= 0]

    outlier_counts = {}
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_counts[col] = int(((data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)).sum())

    high_ok = data["High"] >= data[["Open", "Low", "Close"]].max(axis=1)
    low_ok = data["Low"] <= data[["Open", "High", "Close"]].min(axis=1)
    valid_ohlc = high_ok & low_ok
    inconsistent_rows = int((~valid_ohlc).sum())
    data = data.loc[valid_ohlc].copy()
    if data.empty:
        raise ValueError("No valid rows remain after data cleaning.")

    data = data.sort_values(["Index", "Date"]).reset_index(drop=True)
    grouped = data.groupby("Index", sort=False, group_keys=False)
    data["Daily_Return"] = grouped["Close"].pct_change()
    data["Next_Close"] = grouped["Close"].shift(-1)
    data["Next_Date"] = grouped["Date"].shift(-1)
    data["Target"] = np.where(
        data["Next_Close"].notna(),
        (data["Next_Close"] > data["Close"]).astype(int),
        np.nan,
    )
    data = data.dropna(subset=["Target"]).copy()
    data["Target"] = data["Target"].astype(int)

    metadata = {
        "duplicates_removed": duplicates_removed,
        "inconsistent_rows_removed": inconsistent_rows,
        "rows_after_cleaning_before_features": int(len(data)),
        "outlier_counts": outlier_counts,
    }
    return data, metadata

def train_and_evaluate(data: pd.DataFrame, test_percent: int, hidden_layers: tuple, iterations: int):
    # Only the selected market columns and Daily_Return are used as model features.
    numeric_features = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Daily_Return"]
    categorical_features = ["Index"]
    features = data[numeric_features + categorical_features].replace([np.inf, -np.inf], np.nan)
    target = data["Target"].astype(int)

    # Use a global date boundary; purge training rows whose target date reaches the test period.
    unique_dates = np.sort(data["Date"].dropna().unique())
    if len(unique_dates) < 5:
        raise ValueError("Not enough distinct dates to make a reliable chronological split.")
    split_idx = int(len(unique_dates) * (1 - test_percent / 100))
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates[split_idx])

    train_mask = (data["Date"] < split_date) & (data["Next_Date"] < split_date)
    test_mask = data["Date"] >= split_date
    X_train, y_train = features.loc[train_mask], target.loc[train_mask]
    X_test, y_test = features.loc[test_mask], target.loc[test_mask]

    if len(X_train) < 20 or len(X_test) < 5:
        raise ValueError(
            "Too few rows in the chronological train/test split. Upload a larger dataset "
            "or choose a smaller test percentage."
        )
    if y_train.nunique() < 2:
        raise ValueError("Training data contains only one target class. More data is required.")
    if y_test.nunique() < 2:
        st.warning("The test set contains only one class; ROC-AUC may not be available.")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    solver="adam",
                    max_iter=iterations,
                    early_stopping=True,
                    validation_fraction=0.1,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1] if len(model.classes_) == 2 else None

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": (
            roc_auc_score(y_test, probabilities)
            if probabilities is not None and y_test.nunique() == 2
            else np.nan
        ),
        "report": classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["DOWN_OR_FLAT", "UP"], zero_division=0, output_dict=True
        ),
        "confusion": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "split_date": split_date,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "y_test": y_test,
        "predictions": predictions,
        "probabilities": probabilities,
        "test_index": data.loc[test_mask].index,
        "loss_curve": model.named_steps["classifier"].loss_curve_,
        "iterations_used": model.named_steps["classifier"].n_iter_,
    }
    return model, metrics, numeric_features + categorical_features


if uploaded_file is None:
    st.info("Upload your Market.csv or Market.csv.zip file to start.")
    st.markdown("#### Expected dataset columns")
    st.code("Index, Date, Open, High, Low, Close, Adj Close, Volume")
    st.markdown(
        "**Workflow:** data cleaning → outlier inspection → feature engineering → "
        "chronological split → imputation and scaling → one-hot encoding → "
        "MLP neural network trained with backpropagation."
    )
    st.stop()

try:
    raw_df = load_data(uploaded_file.getvalue(), uploaded_file.name)
    cleaned, cleaning_info = preprocess_and_engineer(raw_df)
except Exception as exc:
    st.error(f"Could not process the uploaded dataset: {exc}")
    st.stop()

if len(cleaned) < 100:
    st.warning("The dataset is quite small. Model performance may not be reliable.")

st.markdown("### Data preprocessing summary")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Uploaded rows", f"{len(raw_df):,}")
m2.metric("Rows after feature preparation", f"{len(cleaned):,}")
m3.metric("Duplicates removed", f"{cleaning_info['duplicates_removed']:,}")
m4.metric("Invalid OHLC rows removed", f"{cleaning_info['inconsistent_rows_removed']:,}")

with st.expander("Outlier inspection (IQR rule)"):
    st.write(
        "Potential outliers are reported, not automatically removed, because unusually large "
        "market movements may be real observations."
    )
    st.dataframe(
        pd.DataFrame(
            [{"Feature": name, "Potential outlier rows": count}
             for name, count in cleaning_info["outlier_counts"].items()]
        ),
        use_container_width=True,
        hide_index=True,
    )

try:
    hidden_layers = tuple(int(x.strip()) for x in hidden_layer_size.split(","))
    with st.spinner("Training neural network with backpropagation..."):
        model, results, feature_names = train_and_evaluate(
            cleaned, test_fraction, hidden_layers, max_iter
        )
except Exception as exc:
    st.error(f"Model training could not be completed: {exc}")
    st.stop()

st.markdown("### Model performance")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Accuracy", f"{results['accuracy']:.3f}")
k2.metric("Balanced accuracy", f"{results['balanced_accuracy']:.3f}")
k3.metric("Precision (UP)", f"{results['precision']:.3f}")
k4.metric("Recall (UP)", f"{results['recall']:.3f}")
k5.metric("F1-score (UP)", f"{results['f1']:.3f}")
st.caption(
    f"Train rows: {results['train_rows']:,} | Test rows: {results['test_rows']:,} | "
    f"Test period starts: {results['split_date'].date()} | "
    f"Training iterations used: {results['iterations_used']}"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Dashboard", "Predictions", "Explore Data", "Methodology"]
)

with tab1:
    left, right = st.columns(2)
    with left:
        trend_counts = cleaned["Target"].map({0: "DOWN_OR_FLAT", 1: "UP"}).value_counts().rename_axis("Trend").reset_index(name="Rows")
        fig = px.bar(
            trend_counts, x="Trend", y="Rows", color="Trend",
            title="Target Class Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)
    with right:
        cm = results["confusion"]
        cm_df = pd.DataFrame(
            cm, index=["Actual DOWN_OR_FLAT", "Actual UP"],
            columns=["Predicted DOWN_OR_FLAT", "Predicted UP"]
        )
        fig = px.imshow(
            cm_df, text_auto=True, aspect="auto",
            title="Confusion Matrix", labels=dict(x="Predicted class", y="Actual class", color="Count")
        )
        st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        cleaned.sort_values("Date"), x="Date", y="Close", color="Index",
        title="Historical Closing Prices",
    )
    st.plotly_chart(fig, use_container_width=True)

    loss_df = pd.DataFrame({
        "Iteration": np.arange(1, len(results["loss_curve"]) + 1),
        "Training loss": results["loss_curve"],
    })
    fig = px.line(loss_df, x="Iteration", y="Training loss", title="Neural Network Training Loss")
    st.plotly_chart(fig, use_container_width=True)

    if results["probabilities"] is not None and results["y_test"].nunique() == 2:
        from sklearn.metrics import precision_recall_curve, roc_curve

        fpr, tpr, _ = roc_curve(results["y_test"], results["probabilities"])
        roc_df = pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr})
        fig = px.line(
            roc_df, x="False positive rate", y="True positive rate",
            title=f"ROC Curve (AUC = {results['roc_auc']:.3f})"
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("#### Test-set predictions")
    test_rows = cleaned.loc[results["test_index"]].copy()
    pred_df = test_rows[["Index", "Date", "Close", "Next_Date", "Next_Close", "Target"]].copy()
    pred_df["Actual Trend"] = pred_df["Target"].map({0: "DOWN_OR_FLAT", 1: "UP"})
    pred_df["Predicted Target"] = results["predictions"]
    pred_df["Predicted Trend"] = pd.Series(results["predictions"], index=pred_df.index).map(
        {0: "DOWN_OR_FLAT", 1: "UP"}
    )
    if results["probabilities"] is not None:
        pred_df["Probability of UP"] = results["probabilities"]
    pred_df["Correct"] = pred_df["Target"].to_numpy() == results["predictions"]
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download test predictions CSV",
        data=pred_df.to_csv(index=False).encode("utf-8"),
        file_name="stock_market_test_predictions.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download cleaned feature dataset CSV",
        data=cleaned[["Index", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Daily_Return", "Target"]].to_csv(index=False).encode("utf-8"),
        file_name="market_cleaned_featured.csv",
        mime="text/csv",
    )
    model_buffer = io.BytesIO()
    joblib.dump(model, model_buffer)
    st.download_button(
        "Download trained model",
        data=model_buffer.getvalue(),
        file_name="stock_market_mlp_pipeline.joblib",
        mime="application/octet-stream",
    )

with tab3:
    st.markdown("#### Inspect processed market records")
    chosen_indices = st.multiselect(
        "Filter market/index",
        options=sorted(cleaned["Index"].astype(str).unique()),
        default=sorted(cleaned["Index"].astype(str).unique()),
    )
    filtered = cleaned[cleaned["Index"].astype(str).isin(chosen_indices)].copy()
    start_date, end_date = st.date_input(
        "Date range",
        value=(filtered["Date"].min().date(), filtered["Date"].max().date()),
        min_value=filtered["Date"].min().date(),
        max_value=filtered["Date"].max().date(),
    )
    filtered = filtered[
        filtered["Date"].dt.date.between(start_date, end_date)
    ]
    st.write(f"Showing **{len(filtered):,}** records")
    st.dataframe(filtered[["Index", "Date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Daily_Return", "Target"]], use_container_width=True, hide_index=True)

with tab4:
    st.markdown("#### Machine-learning workflow")
    st.markdown(
        """
        1. **Load data:** read `Market.csv` or a CSV contained in a ZIP file.
        2. **Remove duplicates:** drop exact duplicate rows.
        3. **Convert data types:** parse dates and convert market attributes to numeric values.
        4. **Handle missing/invalid data:** remove rows missing essential fields and invalid prices or volume.
        5. **Validate OHLC values:** remove rows where High/Low do not agree with Open, Close, and the price range.
        6. **Inspect outliers:** detect IQR outliers and report them; do not automatically remove genuine market spikes.
        7. **Feature engineering:** calculate daily returns, intraday range, lagged prices/returns, moving averages, volatility, volume averages, and calendar features.
        8. **Create target:** `UP` when the next available close for the same Index is higher; otherwise `DOWN_OR_FLAT`.
        9. **Chronological split:** keep the future test period separate and exclude training rows whose next-day label crosses the split boundary.
        10. **Impute and scale:** median-impute missing numeric features and standardise numeric columns using training data only.
        11. **Encode categories:** one-hot encode `Index`, ignoring unseen categories at prediction time.
        12. **Train model:** use `MLPClassifier`, a feed-forward neural network trained with backpropagation.
        13. **Evaluate:** report accuracy, balanced accuracy, precision, recall, F1-score, confusion matrix, training loss, and ROC-AUC when possible.
        """
    )
    report_df = pd.DataFrame(results["report"]).T
    st.markdown("#### Classification report")
    st.dataframe(report_df.round(3), use_container_width=True)
    st.markdown("#### Important limitations")
    st.write(
        "This project demonstrates historical trend classification, not guaranteed future returns. "
        "Stock prices are noisy, results depend on the selected data period, and a good test score "
        "does not guarantee live-market performance. The final record for each Index is excluded "
        "because its next closing price is unavailable."
    )

st.divider()
st.caption(
    "Stock Market Trend Classification | Historical market data | "
    "Algorithm: Backpropagation Neural Network (MLPClassifier)"
)
