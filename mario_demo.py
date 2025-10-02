import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter


st.set_page_config(page_title="🕹️ Mario 互動魔法鏡 ", layout="wide")

st.title("🕹️ Mario 互動魔法鏡 — 旅遊評論互動儀表板 ")
st.write("歡迎來到 Mario 互動魔法鏡，這裡以尼泊爾旅遊景點原始評論資料經過數據探勘、視覺化處理後的圖像作為Demo示範，歡迎您體驗。請透過側邊欄選擇景點，即時看到情緒地圖分布與關鍵字。資料來源：Kaggle上的Tourist Review Sentiment Analysis")

# --------------------------
# 載入資料
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("merged_data_place.csv")
    # 確保 review_tokens 為 list
    if df['review_tokens'].dtype == 'O':
        df['review_tokens'] = df['review_tokens'].apply(eval)
    return df

df = load_data()

# --------------------------
# 側邊欄篩選
# --------------------------
places = df['place'].unique().tolist()
selected_place = st.sidebar.multiselect("選擇景點", options=places, default=places[:10])

# 過濾資料
df_filtered = df[df['place'].isin(selected_place)]

# --------------------------
# 計算情緒分布與關鍵字
# --------------------------
# 情緒計數
emotion_counts = df_filtered.groupby(['place','sentiment']).size().unstack(fill_value=0)
emotion_counts['total_reviews'] = emotion_counts.sum(axis=1)
emotion_counts.reset_index(inplace=True)

# 經緯度
place_info = df_filtered[['place','lat','lng']].drop_duplicates(subset=['place'])
emotion_map = pd.merge(emotion_counts, place_info, on='place', how='left')

# 熱門關鍵字
top_keywords = {}
for place, group_place in df_filtered.groupby("place"):
    top_keywords[place] = {}
    for sentiment, group_sent in group_place.groupby("sentiment"):
        tokens = sum(group_sent['review_tokens'], [])
        count = Counter(tokens)
        top = [word for word, freq in count.most_common(5)]
        top_keywords[place][sentiment] = ", ".join(top)

keywords_df = pd.DataFrame([
    {"place": place, "sentiment": sentiment, "keywords": kws}
    for place, s_dict in top_keywords.items()
    for sentiment, kws in s_dict.items()
])

keyword_pivot = keywords_df.pivot(index='place', columns='sentiment', values='keywords').reset_index()
emotion_map = pd.merge(emotion_map, keyword_pivot, on='place', how='left')

# 重新命名欄位
emotion_map = emotion_map.rename(columns={
    'positive_x':'positive',
    'neutral_x':'neutral',
    'negative_x':'negative',
    'positive_y':'positive_keywords',
    'neutral_y':'neutral_keywords',
    'negative_y':'negative_keywords'
})

# 正面比例
emotion_map['positive_ratio'] = emotion_map.get('positive', 0) / emotion_map['total_reviews']

# --------------------------
# 顯示統計摘要
# --------------------------

st.subheader("🔍 統計摘要""：")

positive_reviews = df_filtered[df_filtered['sentiment']=='positive'].shape[0]
st.write(""f"選擇景點數: {df_filtered['place'].nunique()}""、"f"合計評論數: {df_filtered.shape[0]}""、"f"好評比例: {positive_reviews / df_filtered.shape[0]:.2%}")


# --------------------------
# 顯示互動氣泡圖
# --------------------------
st.subheader("🌍 景點情緒氣泡圖")

fig_map = px.scatter_mapbox(
    emotion_map,
    lat="lat",
    lon="lng",
    size="total_reviews",
    color="positive_ratio",
    hover_name="place",
    hover_data=[
        "positive","neutral","negative",
        "positive_keywords","neutral_keywords","negative_keywords"
    ],
    color_continuous_scale=px.colors.diverging.RdYlGn,
    size_max=40,
    zoom=5,
    mapbox_style="carto-positron"
)
st.plotly_chart(fig_map, use_container_width=True)

# --------------------------
# 顯示表格
# --------------------------
st.subheader("📊 景點情緒統計表")
st.dataframe(emotion_map[[
    'place','total_reviews','positive','neutral','negative',
    'positive_keywords','neutral_keywords','negative_keywords'
]].sort_values('total_reviews', ascending=False))



# ============================
# 顯示詞頻圖
# ============================
st.subheader("📖 景點熱門關鍵字")

# 讓使用者選擇景點
places = df_filtered['place'].unique()
selected_place = st.selectbox("選擇景點", places)

# 計算詞頻
tokens = [token for token in df[df['place']==selected_place]['review_tokens'].sum()]
counter = Counter(tokens)
top_words = counter.most_common(10)
words, counts = zip(*top_words)

# 畫互動長條圖
# 假設 words, counts 已經定義好
fig_keywords = px.bar(
    x=counts,
    y=words,
    orientation='h',
    text=counts,
    labels={'x':'counts','y':'熱門關鍵字'},
    title=f"{selected_place} - Top 10 Keywords",
    color=counts,  # ✅ 依數值漸層
    color_continuous_scale=px.colors.diverging.RdYlGn # 或者 Viridis, Plasma, Cividis, Magma
)

# 條排序：最高頻率在上
fig_keywords.update_layout(
    yaxis={'categoryorder':'total ascending'},
    coloraxis_colorbar=dict(title='counts')  # 顯示右側漸層條
)
st.plotly_chart(fig_keywords, use_container_width=True)
