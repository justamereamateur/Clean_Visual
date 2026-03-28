import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import io
from datetime import datetime

# Page config must be the very first Streamlit command in the file.
# layout="wide" uses the full browser width.
st.set_page_config(page_title="Data Wrangler & Visualizer", layout="wide")
st.title("Data Wrangler & Visualizer")

# -----------------------------------------------------------------------------
# SESSION STATE SETUP
# -----------------------------------------------------------------------------
# Streamlit reruns the whole script from top to bottom every time the user
# clicks a button or changes any widget. Session state is a dictionary that
# persists between reruns so the dataframe is not lost on every click.
# The "not in" check means we only set the default value on the very first run.

if "df" not in st.session_state:
    st.session_state.df = None

if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "transform_log" not in st.session_state:
    st.session_state.transform_log = []

if "undo_stack" not in st.session_state:
    st.session_state.undo_stack = []

if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = ""

# Reset button wipes all stored data and starts fresh
if st.button("Reset everything"):
    st.session_state.df = None
    st.session_state.original_df = None
    st.session_state.transform_log = []
    st.session_state.undo_stack = []
    st.session_state.uploaded_filename = ""
    st.rerun()


# -----------------------------------------------------------------------------
# CACHING - loads the file once and reuses the result on reruns
# -----------------------------------------------------------------------------
# Without caching, pandas would re-read the file on every single rerun.
# @st.cache_data stores the result so the file is only parsed once.

@st.cache_data
def load_csv(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data
def load_excel(file_bytes):
    return pd.read_excel(io.BytesIO(file_bytes))

@st.cache_data
def load_json(file_bytes):
    return pd.DataFrame(json.loads(file_bytes))


# =============================================================================
# SECTION 1 - Upload and Overview
# =============================================================================
# The user uploads a file here. We detect the type from the extension and
# use the cached loader function. After loading we show summary metrics
# and four tabs with more detail. st.stop() prevents everything below
# from running if no file has been uploaded yet.

st.header("Section 1 - Upload and Overview")

uploaded_file = st.file_uploader("Upload your dataset (CSV, Excel or JSON)", type=["csv", "xlsx", "json"])

if uploaded_file is not None:
    filename = uploaded_file.name

    # Only reload when the filename changes. Without this guard, every button
    # click reruns this block and resets st.session_state.df back to the
    # original file, wiping all the cleaning the user has done.
    if filename != st.session_state.uploaded_filename:
        file_bytes = uploaded_file.read()
        try:
            if filename.endswith(".csv"):
                st.session_state.df = load_csv(file_bytes)
            elif filename.endswith(".xlsx"):
                st.session_state.df = load_excel(file_bytes)
            elif filename.endswith(".json"):
                st.session_state.df = load_json(file_bytes)

            st.session_state.original_df = st.session_state.df.copy()
            st.session_state.transform_log = []
            st.session_state.undo_stack = []
            st.session_state.uploaded_filename = filename
            st.success("File loaded: " + filename)
        except Exception as e:
            st.error("Could not read the file. Error: " + str(e))

# Google Sheets - optional, loads from a public CSV export URL
with st.expander("Or connect a Google Sheets URL"):
    sheets_url = st.text_input("Paste the public CSV export URL here")
    if st.button("Load from Sheets") and sheets_url:
        try:
            st.session_state.df = pd.read_csv(sheets_url)
            st.session_state.original_df = st.session_state.df.copy()
            st.session_state.transform_log = []
            st.session_state.undo_stack = []
            st.session_state.uploaded_filename = "google_sheets.csv"
            st.success("Loaded from Google Sheets")
        except Exception as e:
            st.error("Could not load from Google Sheets. Error: " + str(e))

# Stop the app here if no file has been uploaded yet
if st.session_state.df is None:
    st.info("Please upload a file to get started.")
    st.stop()

df = st.session_state.df

# Five summary numbers shown at the top of the page
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Rows", f"{len(df):,}")
col2.metric("Columns", len(df.columns))
col3.metric("Duplicate rows", int(df.duplicated().sum()))
col4.metric("Missing values", int(df.isnull().sum().sum()))
col5.metric("Missing %", f"{df.isnull().sum().sum() / df.size * 100:.1f}%")

# Required: always show the column count in a box
st.info("Total number of columns in this dataset: " + str(len(df.columns)))

# Four tabs for the overview section
overview_tab, stats_tab, missing_tab, duplicates_tab = st.tabs([
    "Column Info", "Summary Stats", "Missing Values", "Duplicates"
])

with overview_tab:
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Non-Null Count": df.notnull().sum().values,
        "Unique Values": df.nunique().values,
    })
    st.dataframe(col_info, use_container_width=True)

with stats_tab:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if numeric_columns:
        st.write("Numeric columns - statistics")
        st.dataframe(df[numeric_columns].describe().T.round(3), use_container_width=True)

    if categorical_columns:
        st.write("Categorical columns - summary")
        cat_summary = pd.DataFrame({
            "Column": categorical_columns,
            "Unique Values": [df[col].nunique() for col in categorical_columns],
            "Most Common": [df[col].mode().iloc[0] if not df[col].mode().empty else "n/a" for col in categorical_columns],
        })
        st.dataframe(cat_summary, use_container_width=True)

with missing_tab:
    missing_counts = df.isnull().sum()
    missing_table = pd.DataFrame({
        "Column": df.columns,
        "Missing Count": missing_counts.values,
        "Missing %": (missing_counts / len(df) * 100).round(2).values
    })
    missing_table = missing_table[missing_table["Missing Count"] > 0]

    if missing_table.empty:
        st.success("No missing values found in this dataset.")
    else:
        st.dataframe(missing_table, use_container_width=True)
        fig, ax = plt.subplots(figsize=(7, max(2, len(missing_table) * 0.4)))
        ax.barh(missing_table["Column"], missing_table["Missing %"])
        ax.set_xlabel("Missing %")
        ax.set_title("Missing values per column")
        st.pyplot(fig)
        plt.close()

with duplicates_tab:
    dup_count = int(df.duplicated().sum())
    if dup_count == 0:
        st.success("No duplicate rows found.")
    else:
        st.warning("Found " + str(dup_count) + " duplicate rows")
        if st.checkbox("Show the duplicate rows"):
            st.dataframe(df[df.duplicated(keep=False)].head(100), use_container_width=True)

# Preview of the raw data with a slider to control how many rows to show
st.write("Data preview")
row_count = st.slider("How many rows to show", 5, min(200, len(df)), 10)
st.dataframe(df.head(row_count), use_container_width=True)


# =============================================================================
# SECTION 2 - Cleaning and Preparation
# =============================================================================
# Eight cleaning tabs. Each one follows the same pattern:
#   1. Show current state of the data
#   2. User picks options with dropdowns / sliders / checkboxes
#   3. User clicks Apply
#   4. We save a snapshot to undo_stack, update session state, log the step,
#      then call st.rerun() to refresh the page.
# These four steps are written directly inside each button block so there
# is no function call between the state update and the rerun.

st.header("Section 2 - Cleaning and Preparation")

# Undo button and current shape info shown side by side
undo_col, info_col = st.columns([1, 4])

with undo_col:
    if st.button("Undo last step") and st.session_state.undo_stack:
        st.session_state.df = st.session_state.undo_stack.pop()
        if st.session_state.transform_log:
            st.session_state.transform_log.pop()
        st.rerun()

with info_col:
    st.info(
        "Current shape: "
        + str(st.session_state.df.shape[0])
        + " rows x "
        + str(st.session_state.df.shape[1])
        + " columns  |  Steps done: "
        + str(len(st.session_state.transform_log))
    )

# Refresh df after possible undo before drawing tabs
df = st.session_state.df

# Eight cleaning tabs
(
    missing_tab2,
    duplicates_tab2,
    types_tab,
    cat_tab,
    outliers_tab,
    scaling_tab,
    columns_tab,
    validation_tab
) = st.tabs([
    "Missing Values",
    "Duplicates",
    "Data Types",
    "Categorical",
    "Outliers",
    "Scaling",
    "Column Operations",
    "Validation"
])


# ---- Missing Values ----------------------------------------------------------
with missing_tab2:
    st.subheader("Fix missing values")

    missing_counts = df.isnull().sum()
    missing_table = pd.DataFrame({
        "Column": df.columns,
        "Missing": missing_counts.values,
        "Missing %": (missing_counts / len(df) * 100).round(2).values
    })
    st.dataframe(missing_table[missing_table["Missing"] > 0], use_container_width=True)

    columns_with_missing = missing_table[missing_table["Missing"] > 0]["Column"].tolist()

    if not columns_with_missing:
        st.success("No missing values to fix.")
    else:
        chosen_col = st.selectbox("Which column to fix?", columns_with_missing, key="mv_col")
        chosen_action = st.selectbox("What to do with missing values?", [
            "Drop rows with missing",
            "Fill with mean",
            "Fill with median",
            "Fill with mode",
            "Fill with a constant value",
            "Forward fill",
            "Backward fill"
        ], key="mv_action")

        constant_value = ""
        if chosen_action == "Fill with a constant value":
            constant_value = st.text_input("Enter the constant value", key="mv_constant")

        if st.button("Apply", key="mv_apply"):
            working_df = st.session_state.df.copy()
            rows_before = len(working_df)

            if chosen_action == "Drop rows with missing":
                working_df = working_df.dropna(subset=[chosen_col])
            elif chosen_action == "Fill with mean":
                working_df[chosen_col] = working_df[chosen_col].fillna(working_df[chosen_col].mean())
            elif chosen_action == "Fill with median":
                working_df[chosen_col] = working_df[chosen_col].fillna(working_df[chosen_col].median())
            elif chosen_action == "Fill with mode":
                working_df[chosen_col] = working_df[chosen_col].fillna(working_df[chosen_col].mode().iloc[0])
            elif chosen_action == "Fill with a constant value":
                working_df[chosen_col] = working_df[chosen_col].fillna(constant_value)
            elif chosen_action == "Forward fill":
                working_df[chosen_col] = working_df[chosen_col].ffill()
            elif chosen_action == "Backward fill":
                working_df[chosen_col] = working_df[chosen_col].bfill()

            # save snapshot for undo, update session, log the step, refresh
            st.session_state.undo_stack.append(st.session_state.df.copy())
            st.session_state.df = working_df
            st.session_state.transform_log.append({
                "step": len(st.session_state.transform_log) + 1,
                "operation": chosen_action,
                "columns": [chosen_col],
                "params": {"column": chosen_col, "action": chosen_action},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Done. Rows before: " + str(rows_before) + "  Rows after: " + str(len(working_df)))
            st.rerun()

    st.markdown("---")
    st.write("Drop entire columns that have too many missing values")
    threshold = st.slider("Drop columns with more than this % missing", 1, 100, 50, key="bulk_threshold")

    if st.button("Drop those columns", key="bulk_drop"):
        working_df = st.session_state.df.copy()
        cols_to_drop = [col for col in working_df.columns if working_df[col].isnull().mean() * 100 > threshold]
        if cols_to_drop:
            working_df = working_df.drop(columns=cols_to_drop)
            st.session_state.undo_stack.append(st.session_state.df.copy())
            st.session_state.df = working_df
            st.session_state.transform_log.append({
                "step": len(st.session_state.transform_log) + 1,
                "operation": "Bulk drop high-missing columns",
                "columns": cols_to_drop,
                "params": {"threshold": threshold},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Dropped columns: " + str(cols_to_drop))
            st.rerun()
        else:
            st.info("No columns exceed that threshold.")


# ---- Duplicates --------------------------------------------------------------
with duplicates_tab2:
    st.subheader("Remove duplicate rows")

    st.metric("Full-row duplicates found", int(df.duplicated().sum()))

    subset_cols = st.multiselect(
        "Check duplicates by specific columns only (leave empty to check all columns)",
        df.columns.tolist(),
        key="dup_subset"
    )
    keep_option = st.radio("Which one to keep?", ["first", "last"], horizontal=True, key="dup_keep")

    if st.button("Remove duplicates", key="dup_apply"):
        working_df = st.session_state.df.copy()
        rows_before = len(working_df)
        working_df = working_df.drop_duplicates(
            subset=subset_cols if subset_cols else None,
            keep=keep_option
        )
        removed = rows_before - len(working_df)
        st.session_state.undo_stack.append(st.session_state.df.copy())
        st.session_state.df = working_df
        st.session_state.transform_log.append({
            "step": len(st.session_state.transform_log) + 1,
            "operation": "Remove duplicates",
            "columns": subset_cols,
            "params": {"keep": keep_option},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.success("Removed " + str(removed) + " duplicate rows")
        st.rerun()

    if st.button("Show duplicate rows", key="dup_show"):
        st.dataframe(
            df[df.duplicated(subset=subset_cols if subset_cols else None, keep=False)].head(100),
            use_container_width=True
        )


# ---- Data Types --------------------------------------------------------------
with types_tab:
    st.subheader("Convert column data types")

    chosen_col = st.selectbox("Column to convert", df.columns.tolist(), key="type_col")
    new_type = st.selectbox("Convert to", ["numeric", "categorical", "datetime", "string"], key="type_target")
    datetime_format = st.text_input("Datetime format if needed (example: %Y-%m-%d)", key="type_fmt")
    clean_first = st.checkbox("Remove non-numeric characters first (removes $, commas etc.)", key="type_clean")

    if st.button("Convert", key="type_apply"):
        try:
            working_df = st.session_state.df.copy()

            if clean_first:
                working_df[chosen_col] = working_df[chosen_col].astype(str).str.replace(r"[^\d.\-]", "", regex=True)

            if new_type == "numeric":
                working_df[chosen_col] = pd.to_numeric(working_df[chosen_col], errors="coerce")
            elif new_type == "categorical":
                working_df[chosen_col] = working_df[chosen_col].astype("category")
            elif new_type == "datetime":
                fmt = datetime_format if datetime_format else None
                working_df[chosen_col] = pd.to_datetime(working_df[chosen_col], format=fmt, errors="coerce")
            elif new_type == "string":
                working_df[chosen_col] = working_df[chosen_col].astype(str)

            st.session_state.undo_stack.append(st.session_state.df.copy())
            st.session_state.df = working_df
            st.session_state.transform_log.append({
                "step": len(st.session_state.transform_log) + 1,
                "operation": "Convert to " + new_type,
                "columns": [chosen_col],
                "params": {"type": new_type},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Converted " + chosen_col + " to " + new_type)
            st.rerun()
        except Exception as e:
            st.error("Conversion failed. Error: " + str(e))


# ---- Categorical -------------------------------------------------------------
with cat_tab:
    st.subheader("Fix categorical columns")

    cat_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not cat_columns:
        st.info("No categorical columns found in this dataset.")
    else:
        chosen_col = st.selectbox("Column", cat_columns, key="cat_col")

        st.write("Top 10 values in this column:")
        st.dataframe(df[chosen_col].value_counts().head(10).reset_index(), use_container_width=True)

        # Fix casing and whitespace
        st.write("Fix casing and whitespace")
        case_choice = st.radio(
            "Apply",
            ["lowercase", "UPPERCASE", "Title Case", "Just strip spaces"],
            horizontal=True,
            key="cat_case"
        )

        if st.button("Apply case fix", key="cat_case_apply"):
            working_df = st.session_state.df.copy()
            if case_choice == "lowercase":
                working_df[chosen_col] = working_df[chosen_col].astype(str).str.strip().str.lower()
            elif case_choice == "UPPERCASE":
                working_df[chosen_col] = working_df[chosen_col].astype(str).str.strip().str.upper()
            elif case_choice == "Title Case":
                working_df[chosen_col] = working_df[chosen_col].astype(str).str.strip().str.title()
            else:
                working_df[chosen_col] = working_df[chosen_col].astype(str).str.strip()
            st.session_state.undo_stack.append(st.session_state.df.copy())
            st.session_state.df = working_df
            st.session_state.transform_log.append({
                "step": len(st.session_state.transform_log) + 1,
                "operation": "Fix casing",
                "columns": [chosen_col],
                "params": {"case": case_choice},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Casing fixed.")
            st.rerun()

        # Replace values using a mapping dictionary
        st.write("Replace values using a mapping")
        st.caption('Write as JSON. Example: {"yes": "Yes", "no": "No"}')
        mapping_input = st.text_area("Mapping", "{}", key="cat_mapping")

        if st.button("Apply mapping", key="cat_map_apply"):
            try:
                mapping_dict = json.loads(mapping_input)
                working_df = st.session_state.df.copy()
                working_df[chosen_col] = working_df[chosen_col].replace(mapping_dict)
                st.session_state.undo_stack.append(st.session_state.df.copy())
                st.session_state.df = working_df
                st.session_state.transform_log.append({
                    "step": len(st.session_state.transform_log) + 1,
                    "operation": "Value mapping",
                    "columns": [chosen_col],
                    "params": {"mapping": mapping_input},
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("Mapping applied.")
                st.rerun()
            except Exception as e:
                st.error("Invalid JSON mapping. Error: " + str(e))

        # Group rare categories into Other
        st.write("Group rare categories into a single 'Other' category")
        rare_pct = st.slider("Treat as rare if it appears in less than this % of rows", 1, 20, 5, key="rare_pct")

        if st.button("Group rare values", key="cat_rare_apply"):
            working_df = st.session_state.df.copy()
            value_freqs = working_df[chosen_col].value_counts(normalize=True) * 100
            rare_values = value_freqs[value_freqs < rare_pct].index.tolist()
            working_df[chosen_col] = working_df[chosen_col].apply(lambda x: "Other" if x in rare_values else x)
            st.session_state.undo_stack.append(st.session_state.df.copy())
            st.session_state.df = working_df
            st.session_state.transform_log.append({
                "step": len(st.session_state.transform_log) + 1,
                "operation": "Group rare categories",
                "columns": [chosen_col],
                "params": {"threshold": rare_pct, "grouped": rare_values},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Grouped " + str(len(rare_values)) + " rare categories into Other.")
            st.rerun()

        # One-hot encode
        st.write("One-hot encode this column")
        st.caption("Creates one new binary column per unique value. Recommended before using data in machine learning.")

        if st.button("One-hot encode", key="cat_ohe_apply"):
            working_df = st.session_state.df.copy()
            dummies = pd.get_dummies(working_df[chosen_col], prefix=chosen_col)
            working_df = pd.concat([working_df.drop(columns=[chosen_col]), dummies], axis=1)
            st.session_state.undo_stack.append(st.session_state.df.copy())
            st.session_state.df = working_df
            st.session_state.transform_log.append({
                "step": len(st.session_state.transform_log) + 1,
                "operation": "One-hot encode",
                "columns": [chosen_col],
                "params": {"new_columns": dummies.columns.tolist()},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Created " + str(len(dummies.columns)) + " new columns.")
            st.rerun()


# ---- Outliers ----------------------------------------------------------------
with outliers_tab:
    st.subheader("Detect and handle outliers")

    num_columns = df.select_dtypes(include="number").columns.tolist()

    if not num_columns:
        st.info("No numeric columns found.")
    else:
        chosen_col = st.selectbox("Column to check", num_columns, key="out_col")
        detection_method = st.radio("Detection method", ["IQR", "Z-score"], horizontal=True, key="out_method")
        z_threshold = st.slider("Z-score threshold (only used if Z-score is selected)", 1.5, 5.0, 3.0, 0.1, key="z_threshold")

        col_data = df[chosen_col].dropna()

        if detection_method == "IQR":
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
        else:
            z_scores = (col_data - col_data.mean()) / col_data.std()
            lower_bound = col_data[z_scores.abs() <= z_threshold].min()
            upper_bound = col_data[z_scores.abs() <= z_threshold].max()

        outlier_count = int(((df[chosen_col] < lower_bound) | (df[chosen_col] > upper_bound)).sum())
        st.info(
            "Outliers found: " + str(outlier_count)
            + "   Acceptable range: " + str(round(lower_bound, 4))
            + " to " + str(round(upper_bound, 4))
        )

        fig, ax = plt.subplots(figsize=(7, 2))
        ax.boxplot(col_data, vert=False)
        ax.set_title("Boxplot for " + chosen_col)
        st.pyplot(fig)
        plt.close()

        outlier_action = st.radio(
            "What to do with outliers?",
            ["Do nothing", "Cap values at the boundary", "Remove outlier rows"],
            horizontal=True,
            key="out_action"
        )

        if outlier_action != "Do nothing" and st.button("Apply", key="out_apply"):
            working_df = st.session_state.df.copy()
            rows_before = len(working_df)

            if outlier_action == "Cap values at the boundary":
                working_df[chosen_col] = working_df[chosen_col].clip(lower=lower_bound, upper=upper_bound)
                msg = "Values have been capped."
            else:
                working_df = working_df[~((working_df[chosen_col] < lower_bound) | (working_df[chosen_col] > upper_bound))]
                msg = "Removed " + str(rows_before - len(working_df)) + " outlier rows."

            st.session_state.undo_stack.append(st.session_state.df.copy())
            st.session_state.df = working_df
            st.session_state.transform_log.append({
                "step": len(st.session_state.transform_log) + 1,
                "operation": "Outliers: " + outlier_action,
                "columns": [chosen_col],
                "params": {"method": detection_method, "lower": lower_bound, "upper": upper_bound},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success(msg)
            st.rerun()


# ---- Scaling -----------------------------------------------------------------
with scaling_tab:
    st.subheader("Scale numeric columns")
    st.write("Scaling puts all numeric columns on a similar range so they can be compared fairly.")

    num_columns = df.select_dtypes(include="number").columns.tolist()

    if not num_columns:
        st.info("No numeric columns found.")
    else:
        cols_to_scale = st.multiselect("Choose columns to scale", num_columns, key="scale_cols")
        scale_method = st.radio(
            "Scaling method",
            ["Min-Max (0 to 1)", "Z-score (mean=0, std=1)"],
            horizontal=True,
            key="scale_method"
        )

        if cols_to_scale:
            st.write("Stats before scaling:")
            st.dataframe(df[cols_to_scale].describe().T.round(4), use_container_width=True)

        if cols_to_scale and st.button("Apply scaling", key="scale_apply"):
            working_df = st.session_state.df.copy()

            for col in cols_to_scale:
                if scale_method == "Min-Max (0 to 1)":
                    col_min = working_df[col].min()
                    col_max = working_df[col].max()
                    if col_max != col_min:
                        working_df[col] = (working_df[col] - col_min) / (col_max - col_min)
                    else:
                        working_df[col] = 0
                else:
                    working_df[col] = (working_df[col] - working_df[col].mean()) / working_df[col].std()

            st.write("Stats after scaling:")
            st.dataframe(working_df[cols_to_scale].describe().T.round(4), use_container_width=True)
            st.session_state.undo_stack.append(st.session_state.df.copy())
            st.session_state.df = working_df
            st.session_state.transform_log.append({
                "step": len(st.session_state.transform_log) + 1,
                "operation": "Scaling: " + scale_method,
                "columns": cols_to_scale,
                "params": {"method": scale_method},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Scaling applied.")
            st.rerun()


# ---- Column Operations -------------------------------------------------------
with columns_tab:
    st.subheader("Column operations")

    # Rename
    st.write("Rename a column")
    col_to_rename = st.selectbox("Column to rename", df.columns.tolist(), key="rename_old")
    new_col_name = st.text_input("New name", key="rename_new")

    if st.button("Rename", key="rename_apply") and new_col_name:
        working_df = st.session_state.df.copy()
        working_df = working_df.rename(columns={col_to_rename: new_col_name})
        st.session_state.undo_stack.append(st.session_state.df.copy())
        st.session_state.df = working_df
        st.session_state.transform_log.append({
            "step": len(st.session_state.transform_log) + 1,
            "operation": "Rename column",
            "columns": [col_to_rename],
            "params": {"old": col_to_rename, "new": new_col_name},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.success("Renamed " + col_to_rename + " to " + new_col_name)
        st.rerun()

    st.markdown("---")

    # Drop columns
    st.write("Drop columns")
    cols_to_drop = st.multiselect("Select columns to remove", df.columns.tolist(), key="drop_cols")

    if cols_to_drop and st.button("Drop selected columns", key="drop_apply"):
        working_df = st.session_state.df.copy()
        working_df = working_df.drop(columns=cols_to_drop)
        st.session_state.undo_stack.append(st.session_state.df.copy())
        st.session_state.df = working_df
        st.session_state.transform_log.append({
            "step": len(st.session_state.transform_log) + 1,
            "operation": "Drop columns",
            "columns": cols_to_drop,
            "params": {"dropped": cols_to_drop},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        st.success("Dropped: " + str(cols_to_drop))
        st.rerun()

    st.markdown("---")

    # Create a new column using a formula
    st.write("Create a new column using a formula")
    st.caption("Use column names directly. Examples:  price * quantity    or    np.log(price)")
    formula_col_name = st.text_input("Name for the new column", "new_column", key="formula_name")
    formula = st.text_input("Formula", key="formula_input")

    if formula and st.button("Create column", key="formula_apply"):
        try:
            working_df = st.session_state.df.copy()
            column_refs = {col: working_df[col] for col in working_df.columns}
            column_refs["np"] = np
            working_df[formula_col_name] = eval(formula, {"__builtins__": {}}, column_refs)
            st.session_state.undo_stack.append(st.session_state.df.copy())
            st.session_state.df = working_df
            st.session_state.transform_log.append({
                "step": len(st.session_state.transform_log) + 1,
                "operation": "Create column",
                "columns": [formula_col_name],
                "params": {"formula": formula},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            st.success("Created column: " + formula_col_name)
            st.rerun()
        except Exception as e:
            st.error("Formula error: " + str(e))

    st.markdown("---")

    # Bin a numeric column
    st.write("Bin a numeric column into groups")
    st.caption("Splits a continuous number column into labelled buckets or ranges.")
    num_columns = df.select_dtypes(include="number").columns.tolist()

    if num_columns:
        col_to_bin = st.selectbox("Column to bin", num_columns, key="bin_col")
        number_of_bins = st.slider("Number of bins", 2, 20, 5, key="bin_count")
        bin_strategy = st.radio("Strategy", ["Equal-width", "Quantile-based"], horizontal=True, key="bin_strategy")
        bin_col_name = st.text_input("New column name", col_to_bin + "_group", key="bin_new_name")

        if st.button("Create bins", key="bin_apply"):
            try:
                working_df = st.session_state.df.copy()
                if bin_strategy == "Equal-width":
                    working_df[bin_col_name] = pd.cut(working_df[col_to_bin], bins=number_of_bins, labels=False)
                else:
                    working_df[bin_col_name] = pd.qcut(working_df[col_to_bin], q=number_of_bins, labels=False, duplicates="drop")
                st.session_state.undo_stack.append(st.session_state.df.copy())
                st.session_state.df = working_df
                st.session_state.transform_log.append({
                    "step": len(st.session_state.transform_log) + 1,
                    "operation": "Bin column",
                    "columns": [bin_col_name],
                    "params": {"source": col_to_bin, "bins": number_of_bins, "strategy": bin_strategy},
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("Created column: " + bin_col_name)
                st.rerun()
            except Exception as e:
                st.error("Binning error: " + str(e))


# ---- Validation --------------------------------------------------------------
with validation_tab:
    st.subheader("Check data quality rules")
    st.write("Define rules and the app will find any rows that break them.")

    current_df = st.session_state.df
    num_columns = current_df.select_dtypes(include="number").columns.tolist()
    cat_columns = current_df.select_dtypes(include=["object", "category"]).columns.tolist()
    violations = []

    st.write("Numeric range check")
    if num_columns:
        range_col = st.selectbox("Column", num_columns, key="val_range_col")
        range_min = st.number_input("Minimum allowed value", value=float(current_df[range_col].min()), key="val_min")
        range_max = st.number_input("Maximum allowed value", value=float(current_df[range_col].max()), key="val_max")

    st.write("Allowed categories check")
    if cat_columns:
        cat_check_col = st.selectbox("Column", cat_columns, key="val_cat_col")
        allowed_values = st.text_input(
            "Allowed values (comma separated)",
            ", ".join(current_df[cat_check_col].dropna().unique()[:10].tolist()),
            key="val_allowed"
        )

    st.write("Non-null check")
    notnull_cols = st.multiselect("These columns must not have any empty values", current_df.columns.tolist(), key="val_notnull")

    if st.button("Run validation", key="val_run"):

        if num_columns:
            out_of_range = current_df[
                (current_df[range_col] < range_min) | (current_df[range_col] > range_max)
            ].copy()
            out_of_range["violation"] = range_col + " is out of range"
            violations.append(out_of_range)

        if cat_columns:
            allowed_list = [v.strip() for v in allowed_values.split(",")]
            bad_cats = current_df[
                ~current_df[cat_check_col].isin(allowed_list) & current_df[cat_check_col].notna()
            ].copy()
            bad_cats["violation"] = cat_check_col + " has an unexpected value"
            violations.append(bad_cats)

        for col in notnull_cols:
            nulls = current_df[current_df[col].isnull()].copy()
            nulls["violation"] = col + " is null"
            violations.append(nulls)

        if violations:
            all_violations = pd.concat(violations, ignore_index=True)
            st.warning("Found " + str(len(all_violations)) + " violations")
            st.dataframe(all_violations, use_container_width=True)
            st.download_button(
                "Download violations as CSV",
                all_violations.to_csv(index=False).encode(),
                "violations.csv",
                "text/csv"
            )
        else:
            st.success("All checks passed. No violations found.")


# =============================================================================
# SECTION 3 - Visualization
# =============================================================================
# The user picks a chart type and configures it using dropdowns and sliders.
# The chart only draws when they click the Draw button.
# All charts use matplotlib. The heatmap uses seaborn on top of matplotlib.
# An optional filter lets the user narrow down the data before plotting.

st.header("Section 3 - Visualization")

df = st.session_state.df
numeric_columns = df.select_dtypes(include="number").columns.tolist()
categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
datetime_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

chart_type = st.selectbox("Choose chart type", [
    "Histogram",
    "Box Plot",
    "Scatter Plot",
    "Line Chart",
    "Bar Chart",
    "Heatmap"
])

aggregation = st.selectbox("Aggregation for grouped charts", ["mean", "sum", "count", "median"], key="agg")
top_n_bars = st.slider("Show top N categories (applies to bar chart)", 5, 50, 15, key="top_n")

# Filter section - lets user narrow down the data before plotting
with st.expander("Filter the data before plotting"):
    filter_col = st.selectbox("Filter by column", ["none"] + df.columns.tolist(), key="filter_col")
    filtered_df = df.copy()

    if filter_col != "none":
        if pd.api.types.is_numeric_dtype(df[filter_col]):
            f_min = float(df[filter_col].min())
            f_max = float(df[filter_col].max())
            f_range = st.slider("Value range", f_min, f_max, (f_min, f_max), key="filter_range")
            filtered_df = filtered_df[filtered_df[filter_col].between(f_range[0], f_range[1])]
        else:
            all_options = df[filter_col].dropna().unique().tolist()
            selected = st.multiselect("Include these values", all_options, default=all_options, key="filter_vals")
            filtered_df = filtered_df[filtered_df[filter_col].isin(selected)]

    st.caption("Rows after filter: " + str(len(filtered_df)))


# Histogram
if chart_type == "Histogram":
    x_col = st.selectbox("Column to plot", numeric_columns or df.columns.tolist(), key="hist_x")
    num_bins = st.slider("Number of bins", 5, 100, 30, key="hist_bins")
    color_by = st.selectbox("Color by category (optional)", ["none"] + categorical_columns, key="hist_color")

    if st.button("Draw chart", key="hist_draw"):
        fig, ax = plt.subplots(figsize=(9, 4))
        if color_by != "none":
            for group, group_df in filtered_df.groupby(color_by):
                ax.hist(group_df[x_col].dropna(), bins=num_bins, alpha=0.6, label=str(group))
            ax.legend()
        else:
            ax.hist(filtered_df[x_col].dropna(), bins=num_bins, edgecolor="white")
        ax.set_xlabel(x_col)
        ax.set_title("Distribution of " + x_col)
        st.pyplot(fig)
        plt.close()


# Box Plot
elif chart_type == "Box Plot":
    y_col = st.selectbox("Numeric column", numeric_columns or df.columns.tolist(), key="box_y")
    group_by = st.selectbox("Group by (optional)", ["none"] + categorical_columns, key="box_group")

    if st.button("Draw chart", key="box_draw"):
        fig, ax = plt.subplots(figsize=(9, 4))
        if group_by != "none":
            group_data = [group_df[y_col].dropna().values for _, group_df in filtered_df.groupby(group_by)]
            group_labels = filtered_df[group_by].dropna().unique()[:top_n_bars]
            ax.boxplot(group_data[:top_n_bars], labels=group_labels)
            plt.xticks(rotation=45, ha="right")
        else:
            ax.boxplot(filtered_df[y_col].dropna().values)
        ax.set_title("Box plot of " + y_col)
        st.pyplot(fig)
        plt.close()


# Scatter Plot
elif chart_type == "Scatter Plot":
    x_col = st.selectbox("X axis", numeric_columns or df.columns.tolist(), key="scatter_x")
    y_col = st.selectbox("Y axis", numeric_columns or df.columns.tolist(), key="scatter_y")
    color_by = st.selectbox("Color by (optional)", ["none"] + categorical_columns, key="scatter_color")

    if st.button("Draw chart", key="scatter_draw"):
        fig, ax = plt.subplots(figsize=(8, 5))
        if color_by != "none":
            for group, group_df in filtered_df.groupby(color_by):
                ax.scatter(group_df[x_col], group_df[y_col], alpha=0.5, label=str(group), s=20)
            ax.legend(title=color_by)
        else:
            ax.scatter(filtered_df[x_col], filtered_df[y_col], alpha=0.5, s=20)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(y_col + " vs " + x_col)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# Line Chart
elif chart_type == "Line Chart":
    x_col = st.selectbox("X axis", (datetime_columns + numeric_columns) or df.columns.tolist(), key="line_x")
    y_col = st.selectbox("Y axis", numeric_columns or df.columns.tolist(), key="line_y")
    group_by = st.selectbox("Group by (optional)", ["none"] + categorical_columns, key="line_group")

    if st.button("Draw chart", key="line_draw"):
        fig, ax = plt.subplots(figsize=(10, 4))
        sorted_df = filtered_df.sort_values(x_col)
        if group_by != "none":
            for group, group_df in sorted_df.groupby(group_by):
                aggregated = group_df.groupby(x_col)[y_col].agg(aggregation).reset_index()
                ax.plot(aggregated[x_col], aggregated[y_col], label=str(group))
            ax.legend(title=group_by)
        else:
            aggregated = sorted_df.groupby(x_col)[y_col].agg(aggregation).reset_index()
            ax.plot(aggregated[x_col], aggregated[y_col])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(y_col + " over " + x_col)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# Bar Chart
elif chart_type == "Bar Chart":
    x_col = st.selectbox("Category column (X axis)", categorical_columns or df.columns.tolist(), key="bar_x")
    y_col = st.selectbox("Value column (Y axis)", numeric_columns or df.columns.tolist(), key="bar_y")
    group_by = st.selectbox("Group by (optional)", ["none"] + categorical_columns, key="bar_group")

    if st.button("Draw chart", key="bar_draw"):
        fig, ax = plt.subplots(figsize=(10, 5))
        if group_by != "none":
            pivot = filtered_df.groupby([x_col, group_by])[y_col].agg(aggregation).unstack(fill_value=0)
            pivot.iloc[:top_n_bars].plot(kind="bar", ax=ax)
            ax.legend(title=group_by, bbox_to_anchor=(1.01, 1))
        else:
            aggregated = filtered_df.groupby(x_col)[y_col].agg(aggregation).nlargest(top_n_bars)
            aggregated.plot(kind="bar", ax=ax, edgecolor="white")
        ax.set_xlabel(x_col)
        ax.set_ylabel(aggregation + "(" + y_col + ")")
        ax.set_title(aggregation + " of " + y_col + " by " + x_col)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# Heatmap / Correlation Matrix
elif chart_type == "Heatmap":
    selected_cols = st.multiselect(
        "Select numeric columns to include",
        numeric_columns,
        default=numeric_columns[:min(10, len(numeric_columns))],
        key="heatmap_cols"
    )

    if st.button("Draw chart", key="heatmap_draw"):
        if len(selected_cols) < 2:
            st.warning("Please select at least 2 columns.")
        else:
            correlation_matrix = filtered_df[selected_cols].corr()
            fig, ax = plt.subplots(
                figsize=(max(6, len(selected_cols)), max(5, len(selected_cols) - 1))
            )
            sns.heatmap(
                correlation_matrix,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                ax=ax,
                square=True
            )
            ax.set_title("Correlation Matrix")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# =============================================================================
# SECTION 4 - Export
# =============================================================================
# This section lets the user download:
#   1. The cleaned dataset as CSV or Excel
#   2. A JSON recipe file recording every step taken
#   3. A Markdown report summarising what was done

st.header("Section 4 - Export")

df = st.session_state.df
st.success(
    "Dataset ready to export: "
    + str(df.shape[0])
    + " rows and "
    + str(df.shape[1])
    + " columns"
)

# Show the full log of every step the user applied
st.subheader("Transformation Log")

if not st.session_state.transform_log:
    st.info("No transformations have been applied yet.")
else:
    log_table = pd.DataFrame([{
        "Step": entry["step"],
        "Operation": entry["operation"],
        "Columns affected": ", ".join(map(str, entry["columns"])),
        "Parameters": str(entry["params"]),
        "Time": entry["timestamp"]
    } for entry in st.session_state.transform_log])
    st.dataframe(log_table, use_container_width=True)

st.markdown("---")
st.subheader("Download files")

export_col1, export_col2, export_col3 = st.columns(3)

# Download as CSV
# df.to_csv() converts the dataframe to a comma-separated text string.
# .encode() turns that string into bytes which the download button needs.
# index=False means we do not include the row numbers in the file.
with export_col1:
    st.download_button(
        "Download cleaned CSV",
        data=df.to_csv(index=False).encode(),
        file_name="cleaned_data.csv",
        mime="text/csv",
        use_container_width=True
    )

# Download as Excel
# We use an in-memory buffer (BytesIO) so we never write a real file to disk.
# pd.ExcelWriter writes into that buffer, then we pass the contents out.
with export_col2:
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        "Download cleaned Excel",
        data=excel_buffer.getvalue(),
        file_name="cleaned_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

# Download JSON recipe
# The recipe is a dictionary containing every transformation step so the
# workflow is fully documented and could be reproduced on a new dataset.
with export_col3:
    recipe = {
        "created": datetime.now().isoformat(),
        "source_file": st.session_state.uploaded_filename,
        "steps": st.session_state.transform_log
    }
    st.download_button(
        "Download transformation recipe (JSON)",
        data=json.dumps(recipe, indent=2, default=str).encode(),
        file_name="recipe.json",
        mime="application/json",
        use_container_width=True
    )

# Transformation Report
# Build the report as a list of strings and join them with newlines.
# st.markdown() renders it with proper headings and bullet points.
st.markdown("---")
st.subheader("Transformation Report")

report_lines = [
    "# Transformation Report",
    "Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Source file: " + st.session_state.uploaded_filename,
    "Final shape: " + str(df.shape[0]) + " rows, " + str(df.shape[1]) + " columns",
    "",
    "## Steps applied",
]

for entry in st.session_state.transform_log:
    report_lines.append("### Step " + str(entry["step"]) + ": " + entry["operation"])
    report_lines.append("- Time: " + entry["timestamp"])
    report_lines.append("- Columns: " + ", ".join(map(str, entry["columns"])))
    report_lines.append("- Parameters: " + str(entry["params"]))
    report_lines.append("")

if not st.session_state.transform_log:
    report_lines.append("No steps were applied.")

report_text = "\n".join(report_lines)
st.markdown(report_text)

st.download_button(
    "Download report as Markdown",
    data=report_text.encode(),
    file_name="transformation_report.md",
    mime="text/markdown"
)
