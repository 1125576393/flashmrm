import streamlit as st
import pandas as pd
import time
from FlashMRM import Config, MRMOptimizer
import os

# ===============================
# Page Configuration & Styles
# ===============================
st.set_page_config(
    page_title="FlashMRM",
    page_icon="786a50646609813e89cc2017082525a3.png",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 25px;
        color: #1f77b4;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .section-header {
        font-size: 18px;
        font-weight: bold;
        margin-top: 35px;   /* Increased spacing */
        margin-bottom: 15px;
    }
    .input-container {
        display: flex;
        align-items: center;
        margin-bottom: 15px;
    }
    .input-label {
        width: 150px;
        font-weight: bold;
    }
    .result-container {
        margin-top: 20px;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
    }
    .calculate-button {
        margin-top: 25px;
    }
    .param-section {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 25px;
    }
    .upload-status {
        padding: 8px;
        border-radius: 4px;
        margin-top: 10px;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .error {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# Session State Initialization
# ===============================
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = "Input InChIKey"
if 'inchikey_value' not in st.session_state:
    st.session_state.inchikey_value = "KXRPCFINVWWFHQ-UHFFFAOYSA-N"
if 'batch_file' not in st.session_state:
    st.session_state.batch_file = None
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'upload_status' not in st.session_state:
    st.session_state.upload_status = None
if 'custom_intf_file' not in st.session_state:
    st.session_state.custom_intf_file = None
if 'calculation_in_progress' not in st.session_state:
    st.session_state.calculation_in_progress = False
if 'calculation_complete' not in st.session_state:
    st.session_state.calculation_complete = False
if 'progress_value' not in st.session_state:
    st.session_state.progress_value = 0
if 'result_df' not in st.session_state:
    st.session_state.result_df = pd.DataFrame()
if 'show_help' not in st.session_state:
    st.session_state.show_help = False

# ===============================
# Upload Data Processing
# ===============================
def process_uploaded_data():
    """Process uploaded input data"""
    try:
        if st.session_state.input_mode == "Input InChIKey":
            inchikey = st.session_state.inchikey_value.strip()
            if not inchikey or inchikey.count('-') != 2:
                st.session_state.upload_status = ("error", "Invalid InChIKey format! Example: KXRPCFINVWWFHQ-UHFFFAOYSA-N")
                return False

            st.session_state.uploaded_data = {
                "type": "single_inchikey",
                "data": inchikey,
                "timestamp": time.time()
            }
            st.session_state.upload_status = ("success", f"Successfully uploaded InChIKey: {inchikey}")
            return True

        else:
            batch_file = st.session_state.batch_file
            if batch_file is None:
                st.session_state.upload_status = ("error", "Please upload the file first!")
                return False

            if batch_file.name.endswith('.csv'):
                df = pd.read_csv(batch_file)
                if "InChIKey" not in df.columns:
                    st.session_state.upload_status = ("error", "CSV must contain an 'InChIKey' column!")
                    return False
            elif batch_file.name.endswith('.txt'):
                content = batch_file.getvalue().decode('utf-8')
                inchikeys = [line.strip() for line in content.split('\\n') if line.strip()]
                df = pd.DataFrame({"InChIKey": inchikeys})
            else:
                st.session_state.upload_status = ("error", "Only CSV and TXT files supported!")
                return False

            valid_inchikeys = [ik for ik in df["InChIKey"].dropna().unique() if ik.count('-') == 2]
            if len(valid_inchikeys) == 0:
                st.session_state.upload_status = ("error", "No valid InChIKey found!")
                return False

            st.session_state.uploaded_data = {
                "type": "batch_file",
                "data": pd.DataFrame({"InChIKey": valid_inchikeys}),
                "filename": batch_file.name,
                "timestamp": time.time(),
                "record_count": len(valid_inchikeys)
            }
            st.session_state.upload_status = ("success", f"File uploaded successfully: {batch_file.name} with {len(valid_inchikeys)} valid entries.")
            return True
    except Exception as e:
        st.session_state.upload_status = ("error", f"Upload failed: {str(e)}")
        return False


# ===============================
# Run Calculation
# ===============================
def run_flashmrm_calculation():
    try:
        st.session_state.calculation_in_progress = True
        st.session_state.progress_value = 0
        st.session_state.result_df = pd.DataFrame()

        config = Config()

        # Choose interference data source
        intf_data_selection = st.session_state.get("intf_data", "Default")
        if intf_data_selection == "Default":
            config.INTF_TQDB_PATH = 'INTF-TQDB(from NIST).csv'
            config.USE_NIST_METHOD = True
        elif intf_data_selection == "QE":
            config.INTF_TQDB_PATH = 'INTF-TQDB(from QE).csv'
            config.USE_NIST_METHOD = False
        elif intf_data_selection == "Upload custom":
            uploaded_file = st.session_state.get("custom_intf_file", None)
            if uploaded_file is not None:
                temp_path = f"uploaded_intf_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                config.INTF_TQDB_PATH = temp_path
                # Default to NIST-style method for custom CSV unless the user later adds a toggle
                config.USE_NIST_METHOD = True
                st.info(f"Using custom interference data: {uploaded_file.name}")
            else:
                st.error("Please upload a custom interference data file before running calculation!")
                st.session_state.calculation_in_progress = False
                return

        # Apply tolerance & weight settings
        config.MZ_TOLERANCE = st.session_state.get("mz_tolerance", 0.7)
        config.RT_TOLERANCE = st.session_state.get("rt_tolerance", 2.0)
        config.RT_OFFSET = st.session_state.get("rt_offset", 0.0)
        config.SPECIFICITY_WEIGHT = st.session_state.get("specificity_weight", 0.2)
        config.OUTPUT_PATH = "flashmrm_output.csv"

        # Prepare InChIKeys
        uploaded_data = st.session_state.uploaded_data
        if uploaded_data["type"] == "single_inchikey":
            config.SINGLE_COMPOUND_MODE = True
            config.TARGET_INCHIKEY = uploaded_data["data"]
            target_inchikeys = [uploaded_data["data"]]
        else:
            config.SINGLE_COMPOUND_MODE = False
            target_inchikeys = uploaded_data["data"]["InChIKey"].tolist()
            config.MAX_COMPOUNDS = len(target_inchikeys)

        # Load data once
        optimizer = MRMOptimizer(config)
        optimizer.load_all_data()

        results = []
        total = len(target_inchikeys)
        process_func = optimizer.process_compound_nist if config.USE_NIST_METHOD else optimizer.process_compound_qe

        for idx, inchikey in enumerate(target_inchikeys):
            try:
                if not optimizer.check_inchikey_exists(inchikey):
                    results.append({
                        'chemical': 'not found',
                        'Precursor_mz': 0.0,
                        'InChIKey': inchikey,
                        'RT': 0.0,
                        'coverage_all': 0,
                        'coverage_low': 0,
                        'coverage_medium': 0,
                        'coverage_high': 0,
                        'MSMS1': 0.0,
                        'MSMS2': 0.0,
                        'CE_QQQ1': 0.0,
                        'CE_QQQ2': 0.0,
                        'best5_combinations': "inchikey not found",
                        'max_score': 0.0,
                        'max_sensitivity_score': 0.0,
                        'max_specificity_score': 0.0,
                    })
                else:
                    r = process_func(inchikey)
                    if r:
                        results.append(r)
            except Exception as e:
                results.append({
                    'chemical': 'error',
                    'Precursor_mz': 0.0,
                    'InChIKey': inchikey,
                    'RT': 0.0,
                    'coverage_all': 0,
                    'coverage_low': 0,
                    'coverage_medium': 0,
                    'coverage_high': 0,
                    'MSMS1': 0.0,
                    'MSMS2': 0.0,
                    'CE_QQQ1': 0.0,
                    'CE_QQQ2': 0.0,
                    'best5_combinations': f"error: {str(e)[:50]}...",
                    'max_score': 0.0,
                    'max_sensitivity_score': 0.0,
                    'max_specificity_score': 0.0,
                })

            st.session_state.progress_value = int((idx + 1) / max(total, 1) * 100)
            time.sleep(0.05)

        st.session_state.result_df = pd.DataFrame(results)
        st.session_state.progress_value = 100
        st.session_state.upload_status = ("success", f"Calculation completed for {total} compounds!")
        st.session_state.calculation_complete = True
        st.session_state.calculation_in_progress = False

    except Exception as e:
        st.session_state.upload_status = ("error", f"Error: {str(e)}")
        st.session_state.calculation_in_progress = False
        st.session_state.calculation_complete = True


# ===============================
# Header & Help
# ===============================
col_title, col_help = st.columns([2, 1])
with col_title:
    st.image("786a50646609813e89cc2017082525a3.png", width=200)
with col_help:
    if st.button("Help", key="help_btn", use_container_width=True):
        st.session_state.show_help = not st.session_state.get('show_help', False)

if st.session_state.get('show_help', False):
    st.info("""
**Instruction for Use**
1. **Select Input mode**  
   - *Single InChIKey*: Enter a standard InChIKey (e.g., `KXRPCFINVWWFHQ-UHFFFAOYSA-N`).  
   - *Batch mode*: Upload a CSV (containing column `InChIKey`) or a TXT file (one InChIKey per line).  
2. Click **Upload** to validate and upload the data.  
3. **Parameter setting (optional)**  
   - *M/z tolerance*: mass-to-charge ratio tolerance (default 0.7)  
   - *RT tolerance*: retention time tolerance in minutes (default 2.0)  
   - *RT offset*: retention time offset in minutes (default 0.0)  
   - *Specificity weight*: (0–1), default 0.2  
   - *Select INTF data*: choose interference database  
     - **Default** = NIST-format DB  
     - **QE** = QE-format DB  
     - **Upload custom** = upload a CSV interference data file to be used instead of built-in databases  
4. Click **Calculate** to start; a progress bar will show completion status.  
5. When finished, view the results table and download a CSV.
""")

# ===============================
# Input Mode
# ===============================
st.markdown('<div class="section-header">Select Input mode</div>', unsafe_allow_html=True)
col_a, col_b = st.columns([1, 2])
with col_a:
    selected_mode = st.radio(
        "Input mode:",
        ["Input InChIKey", "Batch mode"],
        index=0 if st.session_state.input_mode == "Input InChIKey" else 1,
        key="mode_selector",
        label_visibility="collapsed"
    )
with col_b:
    if selected_mode == "Input InChIKey":
        inchikey_input = st.text_input(
            "Enter InChIKey:",
            value=st.session_state.inchikey_value,
            placeholder="Example: KXRPCFINVWWFHQ-UHFFFAOYSA-N",
            label_visibility="collapsed",
            key="inchikey_input_active"
        )
        if inchikey_input is not None:
            st.session_state.inchikey_value = inchikey_input
        st.file_uploader(
            "Batch mode (disabled)",
            type=['txt', 'csv'],
            label_visibility="collapsed",
            key="batch_input_disabled",
            disabled=True
        )
    else:
        st.text_input(
            "Input InChIKey (disabled)",
            value="",
            placeholder="Disabled in batch mode",
            label_visibility="collapsed",
            key="inchikey_input_disabled",
            disabled=True
        )
        batch_input = st.file_uploader(
            "Batch mode file:",
            type=['txt', 'csv'],
            label_visibility="collapsed",
            help="Upload CSV (InChIKey column) or TXT (one per line).",
            key="batch_input_active"
        )
        if batch_input is not None:
            st.session_state.batch_file = batch_input

# Upload button (moved up near Input mode)
upload_clicked = st.button(
    "Upload",
    use_container_width=True,
    key="upload_button",
    disabled=st.session_state.calculation_in_progress
)
if upload_clicked:
    process_uploaded_data()

# Upload status
if st.session_state.upload_status:
    t, msg = st.session_state.upload_status
    st.markdown(f'<div class="upload-status {t}">{msg}</div>', unsafe_allow_html=True)

# ===============================
# Parameter Section
# ===============================
st.markdown('<div class="section-header">Parameter setting</div>', unsafe_allow_html=True)
col1, col2 = st.columns([2, 2])
with col1:
    intf_data = st.selectbox(
        "Select INTF data:",
        ["Default", "QE", "Upload custom"],
        index=0,
        key="intf_data",
        help="Default: use NIST; QE: QE-based; Upload custom: use your CSV interference data"
    )
with col2:
    st.write("")

# Custom upload for interference data
if intf_data == "Upload custom":
    st.markdown("#### Upload your interference data (CSV format)")
    custom_file = st.file_uploader(
        "Choose your interference CSV file",
        type=["csv"],
        key="custom_intf_file",
        help="Upload a custom interference data file (CSV)."
    )
    if custom_file is not None:
        st.session_state["custom_intf_file"] = custom_file
        st.success(f"Custom interference file uploaded: {custom_file.name}")

col3, col4 = st.columns([1, 1])
with col3:
    st.number_input("M/z tolerance:", 0.0, 10.0, 0.7, 0.1, key="mz_tolerance")
with col4:
    st.number_input("RT tolerance:", 0.0, 10.0, 2.0, 0.1, key="rt_tolerance")

col5, col6 = st.columns([1, 1])
with col5:
    st.number_input("RT offset:", -10.0, 10.0, 0.0, 0.5, key="rt_offset")
with col6:
    st.number_input("Specificity weight:", 0.0, 1.0, 0.2, 0.05, key="specificity_weight")

# ===============================
# Run Calculation Section
# ===============================
st.markdown('<div class="section-header">Calculate</div>', unsafe_allow_html=True)
col_run, col_prog = st.columns([1, 3])
with col_run:
    calculate_clicked = st.button(
        "Calculate",
        use_container_width=True,
        disabled=(st.session_state.uploaded_data is None) or st.session_state.calculation_in_progress
    )
with col_prog:
    st.progress(st.session_state.progress_value, text=f"{st.session_state.progress_value}% complete")

if calculate_clicked:
    if st.session_state.uploaded_data is None:
        st.error("Please upload data first!")
    else:
        run_flashmrm_calculation()

# ===============================
# Display Results
# ===============================
if st.session_state.calculation_complete:
    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
    df = st.session_state.result_df
    if not df.empty:
        # Show compact table without the long best5 column by default
        display_columns = [c for c in df.columns if c != 'best5_combinations']
        st.dataframe(df[display_columns])
        with st.expander("Show full results (including top-5 ion pairs)"):
            st.dataframe(df)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results CSV", csv, f"FlashMRM_results_{time.strftime('%Y%m%d%H%M%S')}.csv", "text/csv")
        success_count = (df['chemical'].notna() & ~df['chemical'].isin(['not found','calculation failed','error','global error'])).sum()
        st.success(f"Calculation complete ✅ | Successfully processed: {success_count} | Total: {len(df)}")
    else:
        st.warning("No valid results generated.")

# ===============================
# Sidebar
# ===============================
st.sidebar.markdown("---")
st.sidebar.markdown("**FlashMRM** - MRM transition optimization tool")
st.sidebar.markdown(f"Current time: {time.strftime('%Y-%m-%d %H:%M:%S')}")






