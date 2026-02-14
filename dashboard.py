import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime, timedelta

# 1. 환경 설정 및 보안 로직
load_dotenv()

def get_naver_credentials():
    # 1. 로컬 환경 변수 (.env) 우선 확인
    client_id = os.getenv("NAVER_CLIENT_ID")
    client_secret = os.getenv("NAVER_CLIENT_SECRET")
    
    if client_id and client_secret:
        return client_id, client_secret
        
    # 2. 로컬에 없으면 Streamlit Cloud Secrets 확인
    try:
        if "NAVER_CLIENT_ID" in st.secrets:
            return st.secrets["NAVER_CLIENT_ID"], st.secrets["NAVER_CLIENT_SECRET"]
    except Exception:
        # Secrets 파일이 아예 없는 로컬 환경 등에서의 에러 방지
        pass
        
    return None, None

CLIENT_ID, CLIENT_SECRET = get_naver_credentials()

# 2. Naver API 호출 함수들
def fetch_shopping_trend(keywords):
    url = "https://openapi.naver.com/v1/datalab/shopping/category/keywords"
    headers = {
        "X-Naver-Client-Id": CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET,
        "Content-Type": "application/json"
    }
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    keyword_groups = [{"groupName": kw, "keywords": [kw]} for kw in keywords]
    
    body = {
        "startDate": start_date,
        "endDate": end_date,
        "timeUnit": "month",
        "category": "50000000", # 식품 전체
        "keyword": keyword_groups
    }
    
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        data = response.json()
        results = []
        for group in data['results']:
            group_name = group['title']
            for entry in group['data']:
                results.append({
                    "period": entry['period'],
                    "ratio": entry['ratio'],
                    "keyword": group_name
                })
        return pd.DataFrame(results)
    return pd.DataFrame()

def fetch_search_results(api_type, keyword, display=50):
    url = f"https://openapi.naver.com/v1/search/{api_type}.json"
    headers = {
        "X-Naver-Client-Id": CLIENT_ID,
        "X-Naver-Client-Secret": CLIENT_SECRET
    }
    params = {"query": keyword, "display": display}
    
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return pd.DataFrame(response.json()['items'])
    return pd.DataFrame()

# 3. Streamlit UI 구성
st.set_page_config(page_title="Naver API 데이터 분석 대시보드", layout="wide")
st.title("📊 Naver API 실시간 데이터 분석 대시보드")

# 사이드바
st.sidebar.header("🔍 검색 설정")
keyword_input = st.sidebar.text_input("분석할 키워드를 입력하세요 (쉼표 구분)", "오메가3, 비타민D")
keywords = [k.strip() for k in keyword_input.split(",") if k.strip()]

if not CLIENT_ID or not CLIENT_SECRET:
    st.error("Naver API 키가 설정되지 않았습니다. .env 파일이나 Streamlit Secrets를 확인해주세요.")
    st.stop()

# 데이터 로드
if st.sidebar.button("데이터 분석 시작"):
    with st.spinner("네이버에서 데이터를 가져오고 분석 중입니다..."):
        # 데이터 수집
        trend_df = fetch_shopping_trend(keywords)
        blog_dfs = {kw: fetch_search_results("blog", kw) for kw in keywords}
        shop_dfs = {kw: fetch_search_results("shop", kw) for kw in keywords}
        
        # 탭 구성
        tab_trend, tab_eda, tab_viz, tab_raw = st.tabs(["🚀 트렌드 비교", "📊 기초 EDA", "🎨 상세 시각화", "📄 원본 데이터"])
        
        # --- [탭 1] 쇼핑 트렌드 비교 ---
        with tab_trend:
            st.header("키워드별 쇼핑 클릭 지수 추이")
            if not trend_df.empty:
                # 그래프 1: 트렌드 라인 차트
                fig_trend = px.line(trend_df, x="period", y="ratio", color="keyword", 
                                   title="최근 1년 월별 클릭 지수 추이", markers=True)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # 표 1: 트렌드 월별 평균 지수
                st.subheader("월별 클릭 지수 데이터")
                pivot_trend = trend_df.pivot(index="period", columns="keyword", values="ratio")
                st.dataframe(pivot_trend)
                
                # 표 2: 키워드별 기술 통계
                st.subheader("키워드별 통계 요약")
                st.dataframe(trend_df.groupby("keyword")["ratio"].describe())
            else:
                st.warning("트렌드 데이터를 가져오지 못했습니다.")

        # --- [탭 2] 기초 EDA ---
        with tab_eda:
            st.header("데이터셋 기초 분석")
            col1, col2 = st.columns(2)
            
            for i, kw in enumerate(keywords):
                with (col1 if i % 2 == 0 else col2):
                    st.subheader(f"📍 '{kw}' 검색 요약")
                    shop_df = shop_dfs.get(kw, pd.DataFrame())
                    if not shop_df.empty:
                        shop_df['lprice'] = pd.to_numeric(shop_df['lprice'], errors='coerce')
                        # 표 3: 수치형 데이터 요약
                        st.write("쇼핑 데이터 기술 통계")
                        st.dataframe(shop_df[['lprice']].describe())
                        # 표 4: 상위 판매몰 빈도
                        st.write("주요 판매몰 (Top 10)")
                        st.table(shop_df['mallName'].value_counts().head(10))

        # --- [탭 3] 상세 시각화 ---
        with tab_viz:
            st.header("심층 시각화 분석")
            
            # 그래프 2: 가격 분포 히스토그램
            all_shop_data = []
            for kw, df in shop_dfs.items():
                if not df.empty:
                    tdf = df.copy()
                    tdf['keyword'] = kw
                    tdf['lprice'] = pd.to_numeric(tdf['lprice'], errors='coerce')
                    all_shop_data.append(tdf)
            
            if all_shop_data:
                combined_shop = pd.concat(all_shop_data)
                fig_hist = px.histogram(combined_shop, x="lprice", color="keyword", barmode="overlay",
                                       title="키워드별 상품 가격 분포 비교", nbins=30)
                st.plotly_chart(fig_hist, use_container_width=True)
                
                # 그래프 3: 키워드별 평균 가격 (Bar)
                avg_price_df = combined_shop.groupby("keyword")["lprice"].mean().reset_index()
                fig_avg = px.bar(avg_price_df, x="keyword", y="lprice", color="keyword",
                                title="키워드별 평균 상품 가격")
                st.plotly_chart(fig_avg, use_container_width=True)
                
                # 그래프 4: 블로그 키워드 분석 (TF-IDF)
                st.subheader("📝 블로그 주요 키워드 분석 (TF-IDF)")
                for kw in keywords:
                    b_df = blog_dfs.get(kw, pd.DataFrame())
                    if not b_df.empty:
                        vectorizer = TfidfVectorizer(max_features=20)
                        corpus = b_df['title'].fillna('') + " " + b_df['description'].fillna('')
                        tfidf_matrix = vectorizer.fit_transform(corpus)
                        weights = tfidf_matrix.sum(axis=0).A1
                        words_df = pd.DataFrame({'word': vectorizer.get_feature_names_out(), 'weight': weights}).sort_values('weight', ascending=False)
                        
                        fig_word = px.bar(words_df, x="weight", y="word", orientation='h', title=f"'{kw}' 블로그 핵심 키워드")
                        st.plotly_chart(fig_word, use_container_width=True)
                
                # 그래프 5: 판매몰별 가격대 박스플롯
                top_malls = combined_shop['mallName'].value_counts().head(10).index
                mall_subset = combined_shop[combined_shop['mallName'].isin(top_malls)]
                fig_box = px.box(mall_subset, x="mallName", y="lprice", color="keyword",
                                title="주요 10개 판매몰별 가격대 분포")
                st.plotly_chart(fig_box, use_container_width=True)

        # --- [탭 4] 원본 데이터 ---
        with tab_raw:
            st.header("전체 수집 데이터 조회")
            # 표 5: 가격대별 구간 빈도 (통계 표)
            if all_shop_data:
                combined_shop['price_bin'] = pd.cut(combined_shop['lprice'], bins=5)
                price_summary = combined_shop.groupby(['keyword', 'price_bin'], observed=False).size().unstack(level=0)
                st.subheader("가격 구간별 상품 수")
                st.dataframe(price_summary)

            for kw in keywords:
                with st.expander(f"'{kw}' 상세 데이터 보기"):
                    st.write("🛍️ 쇼핑 데이터")
                    st.dataframe(shop_dfs[kw])
                    st.write("📖 블로그 데이터")
                    st.dataframe(blog_dfs[kw])

else:
    st.info("사이드바에서 키워드를 입력하고 '분석 시작' 버튼을 눌러주세요.")
