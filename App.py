
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "data.csv"
MODEL_PATH = BASE / "house_price_model.joblib"
CONFIG_PATH = BASE / "model_config.json"

st.set_page_config(
    page_title="House Price Prediction Pro",
    page_icon="🏡",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 1.5rem;}
div[data-testid="stMetric"] {
    padding: 15px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,.25);
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def money(value):
    return f"${value:,.0f}"


if not DATA_PATH.exists():
    st.error("data.csv app.py ke same folder mein nahi mili.")
    st.stop()

df = load_data()

if "price" not in df.columns:
    st.error("Dataset mein 'price' column hona zaroori hai.")
    st.stop()

if not MODEL_PATH.exists() or not CONFIG_PATH.exists():
    st.warning("Prediction model abhi train nahi hua hai.")
    st.code("python train_model.py", language="bash")
    st.stop()

model = load_model()
config = load_config()

st.sidebar.title("🏡 House Price Pro")
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Predict Price",
        "Dataset Explorer",
        "Model Performance"
    ]
)
st.sidebar.divider()
st.sidebar.caption("Machine Learning • Real Estate Analytics")
st.sidebar.caption(f"{len(df):,} dataset rows")

# ---------------- DASHBOARD ----------------

if page == "Dashboard":
    st.title("🏡 House Price Prediction Pro")
    st.write(
        "Explore housing data, compare machine-learning models "
        "and estimate a property's price."
    )

    prices = pd.to_numeric(df["price"], errors="coerce").dropna()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Houses", f"{len(df):,}")
    c2.metric("Total Columns", f"{len(df.columns)}")
    c3.metric(
        "Average Price",
        money(prices.mean()) if len(prices) else "N/A"
    )
    c4.metric(
        "Median Price",
        money(prices.median()) if len(prices) else "N/A"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("House Price Distribution")
        chart_df = df[["price"]].copy()
        chart_df["price"] = pd.to_numeric(
            chart_df["price"], errors="coerce"
        )
        chart_df = chart_df.dropna()

        fig = px.histogram(
            chart_df,
            x="price",
            nbins=30,
            title="Distribution of House Prices",
            labels={"price": "House Price"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Average Price by City")
        if "city" in df.columns:
            city_df = df.copy()
            city_df["price"] = pd.to_numeric(
                city_df["price"], errors="coerce"
            )
            city_df = (
                city_df.groupby("city", as_index=False)["price"]
                .mean()
                .dropna()
                .sort_values("price", ascending=False)
                .head(10)
            )

            fig = px.bar(
                city_df,
                x="city",
                y="price",
                title="Top 10 Cities by Average Price",
                labels={
                    "city": "City",
                    "price": "Average Price"
                }
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.caption(
        "Predictions are estimates based on the supplied dataset, "
        "not guaranteed market valuations."
    )

# ---------------- PREDICTION ----------------

elif page == "Predict Price":
    st.title("🔮 Predict House Price")
    st.write("Enter the house details and submit the form.")

    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]
    values = {}

    with st.form("house_prediction_form"):
        st.subheader("Property Details")

        cols = st.columns(3)

        for i, feature in enumerate(numeric_features):
            limits = config["numeric_ranges"].get(feature, {})
            default = config["numeric_defaults"].get(feature, 0.0)
            low = float(limits.get("min", default))
            high = float(limits.get("max", default))

            if low == high:
                low -= 1
                high += 1

            default = min(max(float(default), low), high)
            step = (
                1.0 if feature in config["integer_features"] else 0.1
            )

            with cols[i % 3]:
                values[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=low,
                    max_value=high,
                    value=default,
                    step=step
                )

        if categorical_features:
            st.subheader("Location and Other Details")
            cols = st.columns(2)

            for i, feature in enumerate(categorical_features):
                options = config["category_options"].get(feature, [])
                options = [str(x) for x in options] or ["Unknown"]

                with cols[i % 2]:
                    values[feature] = st.selectbox(
                        feature.replace("_", " ").title(),
                        options
                    )

        submitted = st.form_submit_button(
            "✨ Estimate House Price",
            type="primary",
            use_container_width=True
        )

    if submitted:
        input_data = pd.DataFrame([
            {col: values[col] for col in config["feature_columns"]}
        ])

        try:
            prediction = float(model.predict(input_data)[0])

            if not np.isfinite(prediction):
                st.error("Prediction valid nahi hai. Inputs check karo.")
            else:
                st.success("Prediction generated successfully!")
                st.metric("Estimated House Price", money(prediction))
                st.caption(
                    "This is a machine-learning estimate, not a guaranteed price."
                )
                with st.expander("View submitted house details"):
                    st.dataframe(
                        input_data,
                        use_container_width=True,
                        hide_index=True
                    )
        except Exception as error:
            st.error(f"Prediction error: {error}")
            st.info("Try training the model again using train_model.py.")

# ---------------- DATASET EXPLORER ----------------

elif page == "Dataset Explorer":
    st.title("🧭 Dataset Explorer")
    st.write("Inspect, filter and download the housing dataset.")

    c1, c2 = st.columns([2, 1])

    with c1:
        selected_column = st.selectbox("Choose a column", df.columns)

    with c2:
        missing_only = st.checkbox("Only rows with missing values")

    search = st.text_input(
        "Search across dataset",
        placeholder="For example: Seattle"
    )

    view = df.copy()

    if missing_only:
        view = view[view.isna().any(axis=1)]

    if search.strip():
        mask = view.astype(str).apply(
            lambda col: col.str.contains(
                search.strip(), case=False, na=False
            )
        ).any(axis=1)
        view = view[mask]

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows in View", f"{len(view):,}")
    c2.metric("Missing Cells", f"{int(df.isna().sum().sum()):,}")
    c3.metric("Duplicate Rows", f"{int(df.duplicated().sum()):,}")

    st.subheader(f"Summary: {selected_column}")

    if pd.api.types.is_numeric_dtype(df[selected_column]):
        st.dataframe(
            df[selected_column].describe().to_frame(),
            use_container_width=True
        )
    else:
        summary = (
            df[selected_column].fillna("Missing").astype(str)
            .value_counts().head(20)
            .rename_axis(selected_column)
            .reset_index(name="Count")
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Filtered Data")
    st.dataframe(view, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="house_dataset_view.csv",
        mime="text/csv"
    )

# ---------------- MODEL PERFORMANCE ----------------

elif page == "Model Performance":
    st.title("📈 Model Performance")
    st.write(
        "Models are evaluated on a held-out test dataset. "
        "Lower MAE and RMSE are better; higher R² is better."
    )

    st.metric("Selected Model", config["best_model"])

    metrics = pd.DataFrame(config["metrics"]).T
    metrics.index.name = "Model"
    metrics = metrics.reset_index()

    st.subheader("Model Comparison")
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    fig = px.bar(
        metrics,
        x="Model",
        y="MAE",
        title="Mean Absolute Error (Lower is Better)",
        labels={"MAE": "Mean Absolute Error"}
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(
        metrics,
        x="Model",
        y="R2",
        title="R² Score (Higher is Better)",
        labels={"R2": "R² Score"}
    )
    st.plotly_chart(fig2, use_container_width=True)

    importance = config.get("feature_importance", [])

    if importance:
        st.subheader("Feature Importance")
        importance_df = (
            pd.DataFrame(importance)
            .sort_values("importance", ascending=True)
        )

        fig3 = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Features Used by the Selected Tree Model"
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info(
            "Feature importance is available for tree-based selected models."
        )

    st.caption(
        "Model scores depend on this dataset and train-test split. "
        "They do not guarantee accurate predictions for every property."
    )
