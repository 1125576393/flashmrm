import streamlit as st
import pandas as pd
import time
from FlashMRM import Config, MRMOptimizer
import os

# 页面配置
st.set_page_config(
    page_title="FlashMRM",
    page_icon="786a50646609813e89cc2017082525a3.png",
    layout="wide"
)

# 自定义CSS样式
st.markdown("""
<style>
    .main-header {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 20px;
        color: #1f77b4;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .section-header {
        font-size: 18px;
        font-weight: bold;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .input-container {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
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
        margin-top: 20px;
    }
    .param-section {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .upload-status {
        padding: 8px;
        border-radius: 4px;
        margin-top: 5px;
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
     .calculate-container {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-top: 20px;
    }
    .progress-container {
        flex-grow: 1;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
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
if 'calculation_in_progress' not in st.session_state:
    st.session_state.calculation_in_progress = False
if 'calculation_complete' not in st.session_state:
    st.session_state.calculation_complete = False
if 'progress_value' not in st.session_state:
    st.session_state.progress_value = 0
if 'show_help' not in st.session_state:
    st.session_state.show_help = False
if 'result_df' not in st.session_state:
    st.session_state.result_df = pd.DataFrame()


def process_uploaded_data():
    """处理上传的数据"""
    try:
        if st.session_state.input_mode == "Input InChIKey":
            # 处理单个InChIKey
            inchikey = st.session_state.inchikey_value.strip()
            if not inchikey:
                st.session_state.upload_status = ("error", "请输入有效的InChIKey！")
                return False
            
            # InChIKey格式简单验证（标准格式含2个短横线）
            if inchikey.count('-') != 2:
                st.session_state.upload_status = ("error", "InChIKey格式无效！标准格式如：KXRPCFINVWWFHQ-UHFFFAOYSA-N")
                return False
            
            st.session_state.uploaded_data = {
                "type": "single_inchikey",
                "data": inchikey,
                "timestamp": time.time()
            }
            st.session_state.upload_status = ("success", f"成功上传InChIKey: {inchikey}")
            return True
            
        else:  # Batch mode
            # 处理批量文件
            batch_file = st.session_state.batch_file
            if batch_file is None:
                st.session_state.upload_status = ("error", "请上传文件！")
                return False
            
            # 根据文件类型处理
            try:
                if batch_file.name.endswith('.csv'):
                    df = pd.read_csv(batch_file)
                    # 验证CSV是否包含InChIKey列
                    if "InChIKey" not in df.columns:
                        st.session_state.upload_status = ("error", "CSV文件必须包含'InChIKey'列！")
                        return False
                elif batch_file.name.endswith('.txt'):
                    # 假设txt文件每行一个InChIKey
                    content = batch_file.getvalue().decode('utf-8')
                    inchikeys = [line.strip() for line in content.split('\n') if line.strip()]
                    df = pd.DataFrame({"InChIKey": inchikeys})
                else:
                    st.session_state.upload_status = ("error", "不支持的文件格式！仅支持CSV和TXT")
                    return False
            except Exception as e:
                st.session_state.upload_status = ("error", f"文件解析失败: {str(e)}")
                return False
            
            # 过滤无效InChIKey（格式验证）
            valid_inchikeys = [ik for ik in df["InChIKey"].dropna().unique() if ik.count('-') == 2]
            if len(valid_inchikeys) == 0:
                st.session_state.upload_status = ("error", "文件中无有效InChIKey！")
                return False
            
            st.session_state.uploaded_data = {
                "type": "batch_file",
                "data": pd.DataFrame({"InChIKey": valid_inchikeys}),
                "filename": batch_file.name,
                "timestamp": time.time(),
                "record_count": len(valid_inchikeys),
                "original_count": len(df)
            }
            st.session_state.upload_status = (
                "success", 
                f"成功上传文件: {batch_file.name}，原始记录{len(df)}条，有效InChIKey{len(valid_inchikeys)}条"
            )
            return True
            
    except Exception as e:
        st.session_state.upload_status = ("error", f"上传处理失败: {str(e)}")
        return False


def run_flashmrm_calculation():
    """运行 FlashMRM.py 的真实后端计算（支持批量处理）"""
    try:
        st.session_state.calculation_in_progress = True
        st.session_state.calculation_complete = False
        st.session_state.progress_value = 0
        st.session_state.result_df = pd.DataFrame()
        
        # 1. 初始化配置
        config = Config()
        # 从前端获取参数
        config.MZ_TOLERANCE = st.session_state.get("mz_tolerance", 0.7)
        config.RT_TOLERANCE = st.session_state.get("rt_tolerance", 2.0)
        config.RT_OFFSET = st.session_state.get("rt_offset", 0.0)
        config.SPECIFICITY_WEIGHT = st.session_state.get("specificity_weight", 0.2)
        config.OUTPUT_PATH = "flashmrm_output.csv"
        
        # 设置干扰数据库
        intf_data_selection = st.session_state.get("intf_data", "Default")
        if intf_data_selection == "Default":
            config.INTF_TQDB_PATH = 'INTF-TQDB(from NIST).csv'
            config.USE_NIST_METHOD = True
        else:
            config.INTF_TQDB_PATH = 'INTF-TQDB(from QE).csv'
            config.USE_NIST_METHOD = False
        
        # 2. 获取目标InChIKey列表
        uploaded_data = st.session_state.uploaded_data
        if uploaded_data["type"] == "single_inchikey":
            target_inchikeys = [uploaded_data["data"]]
            config.SINGLE_COMPOUND_MODE = True
            config.TARGET_INCHIKEY = target_inchikeys[0]
        else:
            target_inchikeys = uploaded_data["data"]["InChIKey"].tolist()
            config.SINGLE_COMPOUND_MODE = False
            config.MAX_COMPOUNDS = len(target_inchikeys)  # 按有效数量设置最大处理数
        
        # 3. 加载基础数据
        try:
            optimizer = MRMOptimizer(config)
            optimizer.load_all_data()  # 加载demo、Pesudo-TQDB和INTF-TQDB数据
        except ValueError as e:
            if "No matching InChIKeys found" in str(e):
                # 所有化合物均无匹配，生成批量0值结果
                results = []
                for inchikey in target_inchikeys:
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
                        'best5_combinations': "no matching data in database",
                        'max_score': 0.0,
                        'max_sensitivity_score': 0.0,
                        'max_specificity_score': 0.0,
                    })
                st.session_state.result_df = pd.DataFrame(results)
                st.session_state.progress_value = 100
                st.session_state.upload_status = ("error", "所有InChIKey在数据库中无匹配，请检查数据")
                st.session_state.calculation_in_progress = False
                st.session_state.calculation_complete = True
                return
            else:
                raise  # 其他数据加载错误
        
        # 4. 遍历计算所有目标InChIKey
        results = []
        total_compounds = len(target_inchikeys)
        process_func = optimizer.process_compound_nist if config.USE_NIST_METHOD else optimizer.process_compound_qe
        
        for idx, inchikey in enumerate(target_inchikeys):
            try:
                # 检查当前InChIKey是否存在于匹配数据中
                if not optimizer.check_inchikey_exists(inchikey):
                    # 无匹配时生成0值结果
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
                    st.session_state.progress_value = int((idx + 1) / total_compounds * 100)
                    time.sleep(0.1)
                    continue
                
                # 调用后端计算函数
                compound_result = process_func(inchikey)
                if compound_result:
                    results.append(compound_result)
                else:
                    # 计算失败时生成错误标记结果
                    results.append({
                        'chemical': 'calculation failed',
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
                        'best5_combinations': "processing failed",
                        'max_score': 0.0,
                        'max_sensitivity_score': 0.0,
                        'max_specificity_score': 0.0,
                    })
            
            except Exception as e:
                # 单个化合物计算异常，记录错误信息
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
                    'best5_combinations': f"error: {str(e)[:50]}...",  # 截断长错误信息
                    'max_score': 0.0,
                    'max_sensitivity_score': 0.0,
                    'max_specificity_score': 0.0,
                })
            
            # 更新进度条
            st.session_state.progress_value = int((idx + 1) / total_compounds * 100)
            time.sleep(0.1)  # 避免前端进度条卡顿
        
        # 5. 整理最终结果
        st.session_state.result_df = pd.DataFrame(results) if results else pd.DataFrame()
        st.session_state.progress_value = 100
        st.session_state.calculation_complete = True
        st.session_state.calculation_in_progress = False
        st.session_state.upload_status = ("success", f"计算完成！共处理{total_compounds}个化合物")
    
    except Exception as e:
        # 全局异常处理
        st.session_state.calculation_in_progress = False
        st.session_state.calculation_complete = True
        error_msg = f"计算总览错误: {str(e)}"
        st.session_state.upload_status = ("error", error_msg)
        
        # 生成兜底结果（确保前端有数据显示）
        fallback_results = []
        target_inchikeys = []
        if st.session_state.uploaded_data:
            if st.session_state.uploaded_data["type"] == "single_inchikey":
                target_inchikeys = [st.session_state.uploaded_data["data"]]
            else:
                target_inchikeys = st.session_state.uploaded_data["data"]["InChIKey"].tolist()
        
        for inchikey in target_inchikeys[:1]:  # 仅显示第一个化合物的错误兜底
            fallback_results.append({
                'chemical': 'global error',
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
                'best5_combinations': error_msg[:50] + "...",
                'max_score': 0.0,
                'max_sensitivity_score': 0.0,
                'max_specificity_score': 0.0,
            })
        st.session_state.result_df = pd.DataFrame(fallback_results)


# 主标题和Help按钮
col_title, col_help = st.columns([3, 1])
with col_title:
   st.image("786a50646609813e89cc2017082525a3.png", width=250)
with col_help:
    if st.button("Help", width='stretch', key="help_btn"):  
        st.session_state.show_help = not st.session_state.get('show_help', False)

# 显示帮助信息
if st.session_state.get('show_help', False):
    st.info("""
    **使用说明:**
    1. 选择输入模式: 
       - 单个InChIKey：直接输入标准格式的InChIKey（如KXRPCFINVWWFHQ-UHFFFAOYSA-N）
       - 批量模式：上传CSV（含"InChIKey"列）或TXT（每行一个InChIKey）文件
    2. 点击「Upload」按钮验证并上传数据
    3. 参数设置（可选）:
       - M/z tolerance：质荷比容差（默认0.7）
       - RT tolerance：保留时间容差（默认2.0分钟）
       - RT offset：保留时间偏移量（默认0.0）
       - Specificity weight：特异性权重（默认0.2）
       - Select INTF data：选择干扰数据库（Default=NIST，QE=QE格式）
    4. 点击「Calculate」开始计算，进度条显示处理进度
    5. 计算完成后可查看结果表格并下载CSV文件
    """)

# 输入模式选择
st.markdown('<div class="section-header">输入模式</div>', unsafe_allow_html=True)
col_a, col_b = st.columns([1, 2])
with col_a:
    selected_mode = st.radio(
        "选择输入模式:",
        ["Input InChIKey", "Batch mode"],
        index=0 if st.session_state.input_mode == "Input InChIKey" else 1,
        key="mode_selector",
        label_visibility="collapsed"
    )
with col_b:
    if selected_mode == "Input InChIKey":
        # 单个模式输入框
        inchikey_input = st.text_input(
            "Input InChIKey:",
            value=st.session_state.inchikey_value,
            placeholder="例如：KXRPCFINVWWFHQ-UHFFFAOYSA-N",
            label_visibility="collapsed",
            key="inchikey_input_active"
        )
        if inchikey_input:
            st.session_state.inchikey_value = inchikey_input
        
        # 禁用的批量上传框（占位）
        st.file_uploader(
            "Batch mode:",
            type=['txt', 'csv'],
            label_visibility="collapsed",
            key="batch_input_disabled",
            disabled=True,
            help="单个模式下禁用批量上传"
        )
    else:
        # 禁用的单个输入框（占位）
        st.text_input(
            "Input InChIKey:",
            value="",
            placeholder="批量模式下禁用单个输入",
            label_visibility="collapsed",
            key="inchikey_input_disabled",
            disabled=True
        )
        
        # 批量模式文件上传
        batch_input = st.file_uploader(
            "Batch mode:",
            type=['txt', 'csv'],
            help="拖拽文件到此处，支持CSV（含'InChIKey'列）和TXT（每行一个InChIKey），最大200MB",
            label_visibility="collapsed",
            key="batch_input_active"
        )
        if batch_input is not None:
            st.session_state.batch_file = batch_input

# 更新输入模式
if selected_mode != st.session_state.input_mode:
    st.session_state.input_mode = selected_mode
    st.session_state.uploaded_data = None  # 切换模式时清空已上传数据
    st.session_state.upload_status = None
    st.rerun()

# 参数设置部分
st.markdown('<div class="section-header">参数设置</div>', unsafe_allow_html=True)
with st.container():
    # 第一行参数：数据库选择 + 上传按钮
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        intf_data = st.selectbox(
            "Select INTF data:",
            ["Default", "QE"],
            index=0,
            key="intf_data",
            help="Default: 使用NIST格式干扰数据库；QE: 使用QE格式干扰数据库"
        )
    with col2:
        st.write("")  # 占位对齐
    with col3:
        upload_clicked = st.button(
            "Upload", 
            width='stretch',  # 修复use_container_width为width='stretch'
            key="upload_button",
            disabled=st.session_state.calculation_in_progress
        )

    # 第二行参数：M/z容差 + RT偏移
    col4, col5 = st.columns([1, 1])
    with col4:
        mz_tolerance = st.number_input(
            "M/z tolerance:",
            min_value=0.0,
            max_value=10.0,
            value=0.7,
            step=0.1,
            help="质荷比匹配容差，默认0.7",
            key="mz_tolerance"
        )
    with col5:
        rt_offset = st.number_input(
            "RT offset:",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.5,
            help="保留时间偏移量，默认0.0分钟",
            key="rt_offset"
        )

    # 第三行参数：RT容差 + 特异性权重
    col6, col7 = st.columns([1, 1])
    with col6:
        rt_tolerance = st.number_input(
            "RT tolerance:",
            min_value=0.0,
            max_value=10.0,
            value=2.0,
            step=0.1,
            help="保留时间匹配容差，默认2.0分钟",
            key="rt_tolerance"
        )
    with col7:
        specificity_weight = st.number_input(
            "Specificity weight:",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="特异性权重（0-1），默认0.2",
            key="specificity_weight"
        )

# 处理Upload按钮点击
if upload_clicked:
    process_uploaded_data()

# 显示上传状态
if st.session_state.upload_status:
    status_type, message = st.session_state.upload_status
    st.markdown(f'<div class="upload-status {status_type}">{message}</div>', unsafe_allow_html=True)

# 显示已上传的数据信息（展开面板）
if st.session_state.uploaded_data:
    with st.expander("已上传数据信息", expanded=False):
        ud = st.session_state.uploaded_data
        st.write(f"数据类型: {'单个InChIKey' if ud['type'] == 'single_inchikey' else '批量文件'}")
        st.write(f"上传时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ud['timestamp']))}")
        
        if ud["type"] == "single_inchikey":
            st.write(f"InChIKey: {ud['data']}")
        else:
            st.write(f"文件名: {ud['filename']}")
            st.write(f"原始记录数: {ud.get('original_count', 0)}")
            st.write(f"有效InChIKey数: {ud['record_count']}")
            st.write("有效InChIKey预览:")
            st.dataframe(ud['data'].head(10), use_container_width=False)  # 非必要宽度，用默认content
            if len(ud['data']) > 10:
                st.write(f"... 共{len(ud['data'])}条有效记录")

# 计算区域：按钮 + 进度条
st.markdown('<div class="section-header">计算</div>', unsafe_allow_html=True)
col_calc, col_prog = st.columns([1, 3])
with col_calc:
    calculate_clicked = st.button(
        "Calculate", 
        width='stretch',  # 修复use_container_width为width='stretch'
        type="primary", 
        key="calculate_main",
        disabled=st.session_state.calculation_in_progress or st.session_state.uploaded_data is None
    )
with col_prog:
    # 实时更新的进度条
    progress_bar = st.progress(st.session_state.progress_value, text=f"处理进度: {st.session_state.progress_value}%")

# 若进度值变化，更新进度条文本
if st.session_state.progress_value != progress_bar.value:
    progress_bar.progress(st.session_state.progress_value, text=f"处理进度: {st.session_state.progress_value}%")

# 运行计算逻辑
if calculate_clicked:
    if st.session_state.uploaded_data is None:
        st.error("请先使用「Upload」按钮上传并验证数据！")
    else:
        run_flashmrm_calculation()

# 显示计算结果
if st.session_state.calculation_complete:
    st.markdown('<div class="section-header">计算结果</div>', unsafe_allow_html=True)
    result_df = st.session_state.result_df
    
    if not result_df.empty:
        # 显示结果表格（隐藏过长的best5_combinations列，默认不显示）
        display_columns = [col for col in result_df.columns if col != 'best5_combinations']
        st.dataframe(result_df[display_columns], use_container_width=False)  # 非必要宽度，用默认content
        
        # 显示完整结果（展开面板）
        with st.expander("查看完整结果（含最佳5组离子对）", expanded=False):
            st.dataframe(result_df, use_container_width=False)
        
        # 下载结果：修复use_container_width为width='stretch'
        csv_data = result_df.to_csv(index=False, encoding='utf-8').encode('utf-8')
        st.download_button(
            label="📥 下载结果 CSV",
            data=csv_data,
            file_name=f"FlashMRM_results_{time.strftime('%Y%m%d%H%M%S')}.csv",
            mime="text/csv",
            width='stretch',
            key="download_result"
        )
        
        # 计算统计：删除不存在的'other_condition'列，仅基于chemical列有效值判断
        # 成功的条件：chemical不为空且不是错误/未找到标记
        success_conditions = (
            result_df['chemical'].notna() & 
            ~result_df['chemical'].isin(['not found', 'calculation failed', 'error', 'global error'])
        )
        success_count = success_conditions.sum()  # 用sum()统计True的数量，避免len()的歧义
        
        st.success(f"计算完成 ✅ | 成功处理: {success_count}个 | 总处理: {len(result_df)}个")
    else:
        st.warning("未生成任何结果，请检查输入数据或参数配置！")

# 页脚信息
st.sidebar.markdown("---")
st.sidebar.markdown("**FlashMRM** - 质谱MRM参数优化工具")
st.sidebar.markdown(f"当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")









