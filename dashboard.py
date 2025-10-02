import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from PIL import Image
import plotly.express as px

st.set_page_config(page_title="Amazon Sales Dashboard", layout="wide")

# ------------------------- Helpers -------------------------
@st.cache_data
def load_csv(path, nrows=None):
    df = pd.read_csv(path, nrows=nrows, low_memory=False)

    # Try to coerce obvious date columns
    for c in df.columns:
        if 'date' in c.lower():
            try:
                df[c] = pd.to_datetime(df[c], errors='coerce')
            except Exception:
                pass
    return df


def find_column(df, keywords):
    cols = list(df.columns)
    lowmap = {c.lower(): c for c in cols}
    for kw in keywords:
        for c_low, c in lowmap.items():
            if kw in c_low:
                return c
    return None

# ------------------------- App layout -------------------------
st.title("📊 Amazon Sales — Streamlit Dashboard")

# Load the featured dataset only
featured_path = "amazon_sales_featured.csv"
if not os.path.exists(featured_path):
    st.error("`amazon_sales_featured.csv` not found in the current folder. Please place it next to this app.")
    st.stop()

preview_mode = st.sidebar.checkbox("Preview mode (load first 5000 rows)", value=False)
try:
    df = load_csv(featured_path, nrows=5000 if preview_mode else None)
except Exception as e:
    st.error(f"Failed to load {featured_path}: {e}")
    st.stop()

orig_row_count = df.shape[0]

# Detect useful columns
revenue_col = find_column(df, ['revenue', 'sales', 'amount', 'total', 'price', 'order_value'])
qty_col = find_column(df, ['qty', 'quantity', 'units'])
date_col = find_column(df, ['order date', 'order_date', 'date', 'sale date', 'sales date'])
category_col = find_column(df, ['category', 'product category', 'sub-category', 'sub_category'])
city_col = find_column(df, ['city'])
state_col = find_column(df, ['state', 'province'])
channel_col = find_column(df, ['channel', 'sales channel'])
status_col = find_column(df, ['status', 'order_status', 'order status'])
customer_col = find_column(df, ['customer', 'cust_name', 'buyer'])
orderid_col = find_column(df, ['order id', 'order_id', 'orderid', 'id'])

# Convert revenue to numeric
if revenue_col is not None:
    df[revenue_col] = df[revenue_col].astype(str).str.replace('[^0-9.\-]', '', regex=True)
    df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')

# Ensure date column is datetime
if date_col is not None:
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    except Exception:
        pass

# ------------------------- Filters -------------------------
st.sidebar.header("Filters")
filters = {}

if date_col is not None and pd.api.types.is_datetime64_any_dtype(df[date_col]):
    min_date = df[date_col].min().date()
    max_date = df[date_col].max().date()
    date_range = st.sidebar.date_input("Order date range", value=(min_date, max_date))
    if len(date_range) == 2:
        filters['date'] = (pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1]))

if category_col is not None:
    cats = sorted(df[category_col].dropna().unique().astype(str).tolist())
    sel_cats = st.sidebar.multiselect("Category", options=cats, default=cats)
    filters['category'] = sel_cats

if channel_col is not None:
    chs = sorted(df[channel_col].dropna().unique().astype(str).tolist())
    sel_ch = st.sidebar.multiselect("Channel", options=chs, default=chs)
    filters['channel'] = sel_ch

if state_col is not None:
    states = sorted(df[state_col].dropna().unique().astype(str).tolist())
    sel_states = st.sidebar.multiselect("State/Province", options=states, default=states)
    filters['state'] = sel_states

if city_col is not None:
    cities = sorted(df[city_col].dropna().unique().astype(str).tolist())
    sel_cities = st.sidebar.multiselect("City", options=cities, default=None)
    filters['city'] = sel_cities

if status_col is not None:
    statuses = sorted(df[status_col].dropna().unique().astype(str).tolist())
    sel_status = st.sidebar.multiselect("Order Status", options=statuses, default=statuses)
    filters['status'] = sel_status

if revenue_col is not None:
    min_rev = float(np.nanmin(df[revenue_col].fillna(0)))
    max_rev = float(np.nanmax(df[revenue_col].fillna(0)))
    sel_min_rev = st.sidebar.number_input("Minimum revenue (filter)", value=min_rev, min_value=min_rev, max_value=max_rev)
    filters['min_revenue'] = sel_min_rev

search_text = st.sidebar.text_input("Search (product, customer, SKU ...)")

# Apply filters
mask = pd.Series(True, index=df.index)
if 'date' in filters:
    s, e = filters['date']
    mask &= (df[date_col] >= s) & (df[date_col] <= e)
if 'category' in filters:
    mask &= df[category_col].astype(str).isin(filters['category'])
if 'channel' in filters:
    mask &= df[channel_col].astype(str).isin(filters['channel'])
if 'state' in filters:
    mask &= df[state_col].astype(str).isin(filters['state'])
if 'city' in filters and filters['city']:
    mask &= df[city_col].astype(str).isin(filters['city'])
if 'status' in filters:
    mask &= df[status_col].astype(str).isin(filters['status'])
if 'min_revenue' in filters:
    mask &= (df[revenue_col].fillna(0) >= filters['min_revenue'])

if search_text:
    text_mask = pd.Series(False, index=df.index)
    for c in df.select_dtypes(include=['object']).columns:
        text_mask |= df[c].fillna('').str.lower().str.contains(search_text.lower())
    mask &= text_mask

filtered = df[mask].copy()

# ------------------------- Metrics -------------------------
st.subheader("Key metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Rows (filtered)", f"{filtered.shape[0]:,}", delta=f"of {orig_row_count:,}")
with col2:
    if revenue_col:
        st.metric("Total revenue", f"${filtered[revenue_col].sum(min_count=1):,.2f}")
    else:
        st.metric("Total revenue", "N/A")
with col3:
    if orderid_col is not None:
        st.metric("Unique orders", f"{filtered[orderid_col].nunique():,}")
    else:
        st.metric("Unique orders", f"{filtered.shape[0]:,}")
with col4:
    if qty_col:
        st.metric("Total quantity", f"{int(filtered[qty_col].sum(min_count=1))}")
    else:
        st.metric("Total quantity", "N/A")

if revenue_col and orderid_col is not None:
    avg_order_value = filtered.groupby(orderid_col)[revenue_col].sum().mean()
    st.write(f"**Average order value (AOV):** ${avg_order_value:,.2f}")

# ------------------------- Charts -------------------------
st.markdown("---")
st.subheader("Interactive charts")

if date_col is not None and revenue_col is not None:
    ts = filtered.set_index(date_col).resample('D')[revenue_col].sum().reset_index()
    fig_ts = px.line(ts, x=date_col, y=revenue_col, title='Daily revenue')
    st.plotly_chart(fig_ts, use_container_width=True)

if category_col is not None and revenue_col is not None:
    cat_rev = filtered.groupby(category_col)[revenue_col].sum().reset_index().sort_values(revenue_col, ascending=False)
    fig_cat = px.bar(cat_rev, x=category_col, y=revenue_col, title='Revenue by Category')
    st.plotly_chart(fig_cat, use_container_width=True)

cols = st.columns(2)
with cols[0]:
    if state_col is not None and revenue_col is not None:
        state_rev = filtered.groupby(state_col)[revenue_col].sum().reset_index().sort_values(revenue_col, ascending=False).head(10)
        st.plotly_chart(px.bar(state_rev, x=revenue_col, y=state_col, orientation='h', title='Top 10 states by revenue'), use_container_width=True)
with cols[1]:
    if city_col is not None and revenue_col is not None:
        city_rev = filtered.groupby(city_col)[revenue_col].sum().reset_index().sort_values(revenue_col, ascending=False).head(10)
        st.plotly_chart(px.bar(city_rev, x=revenue_col, y=city_col, orientation='h', title='Top 10 cities by revenue'), use_container_width=True)

if revenue_col is not None:
    st.plotly_chart(px.histogram(filtered, x=revenue_col, nbins=50, title='Revenue distribution'), use_container_width=True)
if qty_col is not None:
    st.plotly_chart(px.histogram(filtered, x=qty_col, nbins=50, title='Quantity distribution'), use_container_width=True)

if status_col is not None:
    status_counts = filtered[status_col].value_counts().reset_index()
    status_counts.columns = [status_col, 'count']
    st.plotly_chart(px.pie(status_counts, names=status_col, values='count', title='Order status share'), use_container_width=True)
# ------------------------- Extra Interactive Charts -------------------------

# Monthly Revenue Trend
if date_col is not None and revenue_col is not None:
    monthly_rev = filtered.groupby(filtered[date_col].dt.to_period("M"))[revenue_col].sum().reset_index()
    monthly_rev[date_col] = monthly_rev[date_col].dt.to_timestamp()
    fig_month = px.line(monthly_rev, x=date_col, y=revenue_col, markers=True,
                        title="Monthly Revenue Trend")
    st.plotly_chart(fig_month, use_container_width=True)

# Revenue Heatmap by State & Month
if state_col is not None and date_col is not None and revenue_col is not None:
    heatmap = (
        filtered.groupby([filtered[date_col].dt.to_period("M"), state_col])[revenue_col]
        .sum()
        .reset_index()
    )
    heatmap[date_col] = heatmap[date_col].dt.to_timestamp()

    fig_heatmap = px.density_heatmap(
        heatmap,
        x=date_col,
        y=state_col,
        z=revenue_col,
        color_continuous_scale="Blues",
        title="Revenue Heatmap by State and Month"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

# # ------------------------- Static plots -------------------------
# st.markdown("---")
# st.subheader("Prepared static plots")
# plots_dir = os.path.join(os.getcwd(), 'plots')
# if os.path.exists(plots_dir):
#     plot_files = sorted(glob.glob(os.path.join(plots_dir, '*.png')))
#     if plot_files:
#         cols = st.columns(2)
#         for i, p in enumerate(plot_files):
#             try:
#                 with cols[i % 2]:
#                     st.image(Image.open(p), caption=os.path.basename(p), use_column_width=True)
#             except Exception as e:
#                 st.warning(f"Could not load image {p}: {e}")
#     else:
#         st.info("No PNG images found in the `plots/` folder.")
# else:
#     st.info("No `plots/` folder found in the current directory.")

# ------------------------- Data export -------------------------
st.markdown("---")
st.subheader("Data & export")
if st.checkbox("Show filtered data table", value=False):
    st.dataframe(filtered.head(200))

st.download_button("Download filtered CSV", data=filtered.to_csv(index=False).encode('utf-8'), file_name='filtered_amazon_sales.csv', mime='text/csv')

st.markdown("---")