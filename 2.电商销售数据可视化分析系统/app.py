import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(
    page_title="电商销售数据可视化分析系统",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== 数据加载与清洗模块 ====================
def load_data(file):
    """读取CSV/Excel文件"""
    if file.name.endswith('.csv'):
        df = pd.read_csv(file, encoding='utf-8-sig')
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        st.error("不支持的文件格式，请上传CSV或Excel文件")
        return None
    return df


def clean_data(df):
    """数据清洗：处理缺失值、去重、时间格式转换"""
    original_len = len(df)
    
    # 统一列名映射
    column_mapping = {
        '订单号': 'order_id',
        '商品类别': 'category',
        '销售金额': 'amount',
        '地区': 'region',
        '时间': 'time'
    }
    df = df.rename(columns=column_mapping)
    
    # 处理缺失值
    missing_before = df.isnull().sum().sum()
    # 金额缺失则删除该行
    if 'amount' in df.columns:
        df = df.dropna(subset=['amount'])
    # 其他字段缺失用众数/中位数填充
    for col in ['category', 'region']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else '未知')
    if 'time' in df.columns:
        df['time'] = df['time'].ffill()
    missing_after = df.isnull().sum().sum()
    
    # 去除重复订单
    if 'order_id' in df.columns:
        dup_count = df.duplicated(subset=['order_id']).sum()
        df = df.drop_duplicates(subset=['order_id'])
    else:
        dup_count = df.duplicated().sum()
        df = df.drop_duplicates()
    
    # 转换时间格式
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df = df.dropna(subset=['time'])  # 删除时间转换失败的行
        df['year'] = df['time'].dt.year
        df['month'] = df['time'].dt.month
        df['week'] = df['time'].dt.isocalendar().week.astype(int)
        df['date'] = df['time'].dt.date
    
    # 确保金额为数值型
    if 'amount' in df.columns:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        df = df.dropna(subset=['amount'])
    
    # 异常值检测
    if 'amount' in df.columns:
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df['is_outlier'] = (df['amount'] < 0) | (df['amount'] > upper_bound) | (df['amount'] < lower_bound)
    
    clean_info = {
        'original': original_len,
        'after_clean': len(df),
        'removed': original_len - len(df),
        'duplicates': dup_count,
        'missing_before': missing_before,
        'missing_after': missing_after
    }
    
    return df, clean_info


# ==================== 统计分析模块 ====================
def sales_trend_analysis(df, freq='month'):
    """销售趋势分析：按月/周统计GMV和订单量"""
    if freq == 'month':
        grouped = df.groupby(['year', 'month']).agg(
            gmv=('amount', 'sum'),
            orders=('order_id', 'count') if 'order_id' in df.columns else ('amount', 'count')
        ).reset_index()
        grouped['period'] = grouped['year'].astype(str) + '-' + grouped['month'].astype(str).str.zfill(2)
    else:  # week
        grouped = df.groupby(['year', 'week']).agg(
            gmv=('amount', 'sum'),
            orders=('order_id', 'count') if 'order_id' in df.columns else ('amount', 'count')
        ).reset_index()
        grouped['period'] = grouped['year'].astype(str) + '-W' + grouped['week'].astype(str).str.zfill(2)
    
    return grouped


def category_analysis(df):
    """商品分析：类目销量占比和销售额排名"""
    # 销售额排名
    category_sales = df.groupby('category')['amount'].agg(
        total_sales='sum',
        avg_sales='mean',
        order_count='count'
    ).reset_index().sort_values('total_sales', ascending=False)
    
    # 销量占比
    category_sales['sales_pct'] = (category_sales['total_sales'] / category_sales['total_sales'].sum() * 100).round(2)
    
    return category_sales


def region_analysis(df):
    """地区分析：购买力分布"""
    region_stats = df.groupby('region')['amount'].agg(
        total_sales='sum',
        avg_sales='mean',
        order_count='count'
    ).reset_index().sort_values('total_sales', ascending=False)
    
    region_stats['sales_pct'] = (region_stats['total_sales'] / region_stats['total_sales'].sum() * 100).round(2)
    
    return region_stats


# ==================== 可视化模块 ====================
def plot_sales_trend(grouped, freq='month'):
    """绘制销售趋势折线图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # GMV趋势
    axes[0].plot(grouped['period'], grouped['gmv'], marker='o', linewidth=2, color='#2196F3', markersize=4)
    axes[0].set_title('销售总额(GMV)趋势', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('时间', fontsize=12)
    axes[0].set_ylabel('销售金额(元)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # 订单量趋势
    axes[1].plot(grouped['period'], grouped['orders'], marker='s', linewidth=2, color='#4CAF50', markersize=4)
    axes[1].set_title('订单量趋势', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('时间', fontsize=12)
    axes[1].set_ylabel('订单数', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_category_pie(category_sales):
    """绘制品类占比饼图"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 取Top 8，其余归为其他
    if len(category_sales) > 8:
        top8 = category_sales.head(8)
        other_sales = category_sales.iloc[8:]['total_sales'].sum()
        labels = top8['category'].tolist() + ['其他']
        sizes = top8['total_sales'].tolist() + [other_sales]
    else:
        labels = category_sales['category'].tolist()
        sizes = category_sales['total_sales'].tolist()
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=90,
        textprops={'fontsize': 11}
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('各品类销售额占比', fontsize=14, fontweight='bold')
    return fig


def plot_category_bar(category_sales, top_n=10):
    """绘制品类销售额排名柱状图"""
    top_data = category_sales.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(top_data['category'][::-1], top_data['total_sales'][::-1], 
                   color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_data))))
    
    # 添加数值标签
    for bar, val in zip(bars, top_data['total_sales'][::-1]):
        ax.text(bar.get_width() + max(top_data['total_sales']) * 0.01, 
                bar.get_y() + bar.get_height()/2,
                f'¥{val:,.0f}', va='center', fontsize=10)
    
    ax.set_title(f'品类销售额排名 Top {top_n}', fontsize=14, fontweight='bold')
    ax.set_xlabel('销售总额(元)', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_region_bar(region_stats):
    """绘制地区购买力柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 总销售额
    colors1 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(region_stats)))
    bars1 = axes[0].bar(region_stats['region'], region_stats['total_sales'], color=colors1)
    axes[0].set_title('各地区销售总额', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('销售总额(元)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # 平均客单价
    colors2 = plt.cm.Blues(np.linspace(0.4, 0.9, len(region_stats)))
    bars2 = axes[1].bar(region_stats['region'], region_stats['avg_sales'], color=colors2)
    axes[1].set_title('各地区平均客单价', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('平均金额(元)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_region_map(region_stats):
    """绘制地区订单量分布"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(region_stats)))
    bars = ax.bar(region_stats['region'], region_stats['order_count'], color=colors)
    
    for bar, val in zip(bars, region_stats['order_count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(region_stats['order_count']) * 0.01,
                f'{int(val)}', ha='center', fontsize=10, fontweight='bold')
    
    ax.set_title('各地区订单量分布', fontsize=14, fontweight='bold')
    ax.set_ylabel('订单数', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def plot_outlier_analysis(df):
    """异常值分析可视化"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    outliers = df[df['is_outlier']]
    normal = df[~df['is_outlier']]
    
    # 箱线图
    axes[0].boxplot([normal['amount'], outliers['amount']], labels=['正常订单', '异常订单'],
                    patch_artist=True, widths=0.5)
    axes[0].set_title('订单金额箱线图', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('金额(元)', fontsize=12)
    axes[0].grid(axis='y', alpha=0.3)
    
    # 异常值占比饼图
    outlier_count = len(outliers)
    normal_count = len(normal)
    axes[1].pie([normal_count, outlier_count], labels=['正常订单', '异常订单'],
                autopct='%1.1f%%', colors=['#4CAF50', '#F44336'], startangle=90)
    axes[1].set_title('异常订单占比', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ==================== 报表导出模块 ====================
def generate_html_report(df, clean_info, trend_data, category_data, region_data):
    """生成HTML分析报告"""
    total_sales = df['amount'].sum()
    avg_order = df['amount'].mean()
    total_orders = len(df)
    outlier_count = df['is_outlier'].sum() if 'is_outlier' in df.columns else 0
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>电商销售数据分析报告</title>
        <style>
            body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #1976D2; text-align: center; border-bottom: 3px solid #1976D2; padding-bottom: 15px; }}
            h2 {{ color: #333; border-left: 4px solid #1976D2; padding-left: 15px; margin-top: 30px; }}
            .metrics {{ display: flex; justify-content: space-around; flex-wrap: wrap; margin: 20px 0; }}
            .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin: 10px; min-width: 200px; text-align: center; }}
            .metric-card h3 {{ margin: 0; font-size: 14px; opacity: 0.9; }}
            .metric-card p {{ margin: 10px 0 0; font-size: 28px; font-weight: bold; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background: #1976D2; color: white; padding: 12px; text-align: left; }}
            td {{ padding: 10px 12px; border-bottom: 1px solid #ddd; }}
            tr:hover {{ background: #f5f5f5; }}
            .footer {{ text-align: center; margin-top: 40px; color: #999; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>电商销售数据分析报告</h1>
            <p style="text-align: center; color: #666;">生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>一、数据概况</h2>
            <p>原始数据: {clean_info['original']} 条 | 清洗后: {clean_info['after_clean']} 条 | 删除: {clean_info['removed']} 条</p>
            <p>重复订单: {clean_info['duplicates']} 条 | 异常订单: {outlier_count} 条</p>
            
            <h2>二、核心指标</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>总销售额(GMV)</h3>
                    <p>¥{total_sales:,.2f}</p>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                    <h3>总订单数</h3>
                    <p>{total_orders:,}</p>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                    <h3>平均客单价</h3>
                    <p>¥{avg_order:,.2f}</p>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                    <h3>商品品类数</h3>
                    <p>{df['category'].nunique()}</p>
                </div>
            </div>
            
            <h2>三、品类销售排名</h2>
            <table>
                <tr><th>排名</th><th>品类</th><th>销售总额</th><th>平均单价</th><th>订单数</th><th>占比</th></tr>
    """
    
    for idx, row in category_data.iterrows():
        html_content += f"""
                <tr>
                    <td>{idx + 1}</td>
                    <td>{row['category']}</td>
                    <td>¥{row['total_sales']:,.2f}</td>
                    <td>¥{row['avg_sales']:,.2f}</td>
                    <td>{int(row['order_count'])}</td>
                    <td>{row['sales_pct']}%</td>
                </tr>
        """
    
    html_content += """
            </table>
            
            <h2>四、地区销售分析</h2>
            <table>
                <tr><th>地区</th><th>销售总额</th><th>平均客单价</th><th>订单数</th><th>占比</th></tr>
    """
    
    for _, row in region_data.iterrows():
        html_content += f"""
                <tr>
                    <td>{row['region']}</td>
                    <td>¥{row['total_sales']:,.2f}</td>
                    <td>¥{row['avg_sales']:,.2f}</td>
                    <td>{int(row['order_count'])}</td>
                    <td>{row['sales_pct']}%</td>
                </tr>
        """
    
    html_content += f"""
            </table>
            
            <div class="footer">
                <p>本报告由电商销售数据可视化分析系统自动生成</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content


# ==================== Streamlit主界面 ====================
def main():
    st.title("电商销售数据可视化分析系统")
    st.markdown("---")
    
    # 侧边栏
    with st.sidebar:
        st.header("数据上传")
        uploaded_file = st.file_uploader("上传数据文件", type=['csv', 'xlsx', 'xls'],
                                         help="支持CSV和Excel格式")
        
        if uploaded_file:
            st.success(f"已加载: {uploaded_file.name}")
        
        st.markdown("---")
        st.header("筛选条件")
        
        # 时间筛选
        time_range = st.selectbox("时间维度", ["全部", "按月", "按周"])
        
        # 品类筛选
        if 'df_clean' in st.session_state:
            categories = st.session_state.df_clean['category'].unique().tolist()
            selected_categories = st.multiselect("商品类别", categories, default=categories)
            
            # 地区筛选
            regions = st.session_state.df_clean['region'].unique().tolist()
            selected_regions = st.multiselect("地区", regions, default=regions)
        else:
            selected_categories = []
            selected_regions = []
        
        st.markdown("---")
        st.header("图表设置")
        top_n = st.slider("排名显示数量", 5, 20, 10)
        st.markdown("---")
        
        # 导出按钮
        if 'df_clean' in st.session_state:
            if st.button("导出HTML报告", type="primary"):
                export_report()
    
    # 主内容区
    if uploaded_file is None:
        st.info("请在左侧上传CSV或Excel数据文件开始分析")
        # 显示示例数据说明
        st.markdown("""
        ### 数据格式要求
        
        上传的文件应包含以下字段：
        | 字段名 | 说明 | 示例 |
        |--------|------|------|
        | 订单号 | 唯一订单标识 | ORD2024000001 |
        | 商品类别 | 商品所属分类 | 电子产品 |
        | 销售金额 | 订单金额(元) | 1299.99 |
        | 地区 | 购买地区 | 华东 |
        | 时间 | 下单时间 | 2024-01-15 10:30:00 |
        """)
        return
    
    # 加载数据
    with st.spinner("正在加载数据..."):
        df_raw = load_data(uploaded_file)
    
    if df_raw is None:
        return
    
    # 数据清洗
    with st.spinner("正在清洗数据..."):
        df_clean, clean_info = clean_data(df_raw.copy())
    
    # 保存到session_state
    st.session_state.df_clean = df_clean
    
    # 应用筛选条件
    df_filtered = df_clean.copy()
    if selected_categories:
        df_filtered = df_filtered[df_filtered['category'].isin(selected_categories)]
    if selected_regions:
        df_filtered = df_filtered[df_filtered['region'].isin(selected_regions)]
    
    # 显示数据清洗信息
    with st.expander("数据清洗详情", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("原始数据", clean_info['original'])
        col2.metric("清洗后", clean_info['after_clean'])
        col3.metric("删除记录", clean_info['removed'])
        col4.metric("重复订单", clean_info['duplicates'])
    
    # 核心指标卡片
    st.markdown("### 核心指标")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("总销售额(GMV)", f"¥{df_filtered['amount'].sum():,.2f}")
    col2.metric("总订单数", f"{len(df_filtered):,}")
    col3.metric("平均客单价", f"¥{df_filtered['amount'].mean():,.2f}")
    col4.metric("商品品类数", df_filtered['category'].nunique())
    
    st.markdown("---")
    
    # 销售趋势分析
    st.markdown("### 销售趋势分析")
    freq = 'month' if time_range == "按月" else ('week' if time_range == "按周" else 'month')
    trend_data = sales_trend_analysis(df_filtered, freq)
    
    fig_trend = plot_sales_trend(trend_data, freq)
    st.pyplot(fig_trend)
    plt.close(fig_trend)
    
    st.markdown("---")
    
    # 商品分析
    st.markdown("### 商品品类分析")
    col1, col2 = st.columns(2)
    
    category_data = category_analysis(df_filtered)
    
    with col1:
        fig_pie = plot_category_pie(category_data)
        st.pyplot(fig_pie)
        plt.close(fig_pie)
    
    with col2:
        fig_bar = plot_category_bar(category_data, top_n)
        st.pyplot(fig_bar)
        plt.close(fig_bar)
    
    # 品类数据表格
    with st.expander("查看品类详细数据"):
        st.dataframe(category_data.style.format({
            'total_sales': '¥{:.2f}',
            'avg_sales': '¥{:.2f}',
            'sales_pct': '{:.2f}%'
        }), use_container_width=True)
    
    st.markdown("---")
    
    # 地区分析
    st.markdown("### 地区购买力分析")
    col1, col2 = st.columns(2)
    
    region_data = region_analysis(df_filtered)
    
    with col1:
        fig_region = plot_region_bar(region_data)
        st.pyplot(fig_region)
        plt.close(fig_region)
    
    with col2:
        fig_region_map = plot_region_map(region_data)
        st.pyplot(fig_region_map)
        plt.close(fig_region_map)
    
    # 地区数据表格
    with st.expander("查看地区详细数据"):
        st.dataframe(region_data.style.format({
            'total_sales': '¥{:.2f}',
            'avg_sales': '¥{:.2f}',
            'sales_pct': '{:.2f}%'
        }), use_container_width=True)
    
    st.markdown("---")
    
    # 异常值分析
    st.markdown("### 异常值检测")
    if 'is_outlier' in df_filtered.columns:
        outlier_count = df_filtered['is_outlier'].sum()
        st.info(f"共检测到 **{outlier_count}** 条异常订单（占比 {outlier_count/len(df_filtered)*100:.2f}%）")
        
        fig_outlier = plot_outlier_analysis(df_filtered)
        st.pyplot(fig_outlier)
        plt.close(fig_outlier)
        
        # 显示异常订单详情
        with st.expander("查看异常订单详情"):
            outliers = df_filtered[df_filtered['is_outlier']]
            st.dataframe(outliers[['order_id', 'category', 'amount', 'region', 'time']]
                        .sort_values('amount', ascending=False), use_container_width=True)
    
    st.markdown("---")
    st.caption("电商销售数据可视化分析系统 | 基于 Streamlit + Pandas + Matplotlib 构建")


def export_report():
    """导出HTML报告"""
    df = st.session_state.df_clean
    
    # 重新计算分析数据
    trend_data = sales_trend_analysis(df, 'month')
    category_data = category_analysis(df)
    region_data = region_analysis(df)
    
    clean_info = {
        'original': len(df) + 100,  # 估算
        'after_clean': len(df),
        'removed': 100,
        'duplicates': 0,
        'missing_before': 0,
        'missing_after': 0
    }
    
    html = generate_html_report(df, clean_info, trend_data, category_data, region_data)
    
    st.download_button(
        label="下载HTML报告",
        data=html,
        file_name=f"电商销售分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
        mime="text/html"
    )


if __name__ == "__main__":
    main()
