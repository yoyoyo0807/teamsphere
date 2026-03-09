import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from collections import Counter

# ページ設定
st.set_page_config(page_title="TeamSphere Pro", layout="wide")
st.title("🤝 TeamSphere Pro: 戦略的チーム最適化エンジン")

# --- アプリの説明セクション (READMEの内容を統合) ---
with st.expander("📖 このアプリについて（プロジェクトの概要と仕組み）"):
    st.markdown("""
    ### 🌟 プロジェクトの概要
    TeamSphere Proは、大規模なエンジニアコミュニティの中から、プロジェクトの目的に応じて最適なチームをAIが選抜・分析するプラットフォームです。
    **Stack Overflow Annual Developer Survey 2022** のデータを活用し、849名のエンジニアをモデル化しています。

    ### 1. コミュニティ検出によるチーム編成
    単なるスキルの有無だけでなく、エンジニア同士の「スキルの近接性」をグラフ理論に基づき解析しています。
    - **Louvain法によるコミュニティ検出**: 専門性が近いエンジニアを自動的にグループ化。
    - **現実的な運用**: 現実の組織では「趣味」「価値観」「過去の協業経験」をデータ化することで、**「気が合うコミュニティ」**を特定し、そこから最強のチームを編成することが可能です。

    ### 2. 機械学習 (Machine Learning) の役割
    - **役割予測**: 保有スキルからその人が果たすべき「実質的な役割（AI Role）」を予測。
    - **戦略的スコアリング**: 選択した重点カテゴリ（Data Science等）に基づき、候補者の適合度を算出。
    
    ### 3. 役職の決定
    - **中心性（ハブ度）解析**: ネットワーク内で最も他者と技術的な共通点が多い人を、調整役の**リーダー**として選定します。
    """)

# データとモデルのパス
nodes_path = "data/processed/nodes_with_communities.csv"
edges_path = "data/processed/edges.csv"
model_path = "models/role_model.pkl"

# スキルカテゴリの定義
SKILL_CATEGORIES = {
    'Web Development': ['HTML/CSS', 'JavaScript', 'TypeScript', 'React.js', 'Node.js', 'PHP', 'Ruby'],
    'Data Science/AI': ['Python', 'R', 'Julia', 'Pandas', 'NumPy', 'PyTorch', 'TensorFlow', 'SQL'],
    'Cloud/Infra': ['AWS', 'Docker', 'Kubernetes', 'Terraform', 'Go', 'Bash/Shell', 'PowerShell'],
    'Mobile Dev': ['Swift', 'Kotlin', 'Dart', 'Objective-C', 'React Native'],
    'Low-level/System': ['C', 'C++', 'Rust', 'Assembly', 'MATLAB']
}

def get_skill_scores(skill_list):
    scores = {}
    for cat, cat_skills in SKILL_CATEGORIES.items():
        match_count = len(set(skill_list) & set(cat_skills))
        scores[cat] = match_count
    return scores

def draw_radar_chart(team_skills_df):
    all_skills = []
    for s in team_skills_df['LanguageHaveWorkedWith'].str.split(';'):
        all_skills.extend(s)
    scores = get_skill_scores(all_skills)
    labels = list(scores.keys())
    values = list(scores.values())
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    return fig

if os.path.exists(nodes_path) and os.path.exists(edges_path):
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    G_full = nx.Graph()
    for _, row in edges_df.iterrows():
        G_full.add_edge(row['source'], row['target'], weight=row['weight'])

    st.subheader("🕸️ 全体コミュニティ構造")
    fig_all, ax_all = plt.subplots(figsize=(12, 5))
    pos = nx.spring_layout(G_full, k=0.15, seed=42)
    node_id_to_color = nodes_df.set_index('ResponseId')['community_id'].to_dict()
    node_colors = [node_id_to_color.get(n, 0) for n in G_full.nodes()]
    nx.draw(G_full, pos, with_labels=False, node_color=node_colors, cmap=plt.cm.rainbow, node_size=20, alpha=0.3, ax=ax_all)
    st.pyplot(fig_all)

    st.divider()

    st.subheader("🎯 チーム選抜・戦略設定")
    col_set1, col_set2 = st.columns(2)
    with col_set1:
        mode = st.radio("🔎 選抜アプローチ", ["コミュニティから選抜", "特定個人(自分)に合わせる"])
    with col_set2:
        st.write("💡 重視するスキルカテゴリを選択してください")
        selected_skills = []
        cols = st.columns(3)
        all_cats = list(SKILL_CATEGORIES.keys())
        for i, cat in enumerate(all_cats):
            if cols[i % 3].checkbox(cat, value=True):
                selected_skills.append(cat)
        if len(selected_skills) == len(all_cats):
            strategy_label = "バランス重視（全方位カバー）"
        elif len(selected_skills) == 0:
            strategy_label = "指定なし（中心性重視）"
        else:
            strategy_label = f"特化型 ({', '.join(selected_skills)})"
        st.info(f"現在の戦略: **{strategy_label}**")

    if mode == "コミュニティから選抜":
        community_counts = nodes_df['community_id'].value_counts()
        valid_communities = community_counts[community_counts >= 5].index.sort_values()
        selected_cid = st.selectbox("コミュニティを選択", valid_communities)
        candidates = nodes_df[nodes_df['community_id'] == selected_cid].copy()
    else:
        target_id = st.number_input("ResponseId を入力", min_value=int(nodes_df['ResponseId'].min()), max_value=int(nodes_df['ResponseId'].max()))
        if target_id in nodes_df['ResponseId'].values:
            cid = nodes_df[nodes_df['ResponseId'] == target_id]['community_id'].values[0]
            candidates = nodes_df[nodes_df['community_id'] == cid].copy()
        else:
            candidates = pd.DataFrame()

    if not candidates.empty and st.button("戦略に基づいてチームを編成"):
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model, mlb = pickle.load(f)
            X = mlb.transform(candidates['LanguageHaveWorkedWith'].str.split(';'))
            candidates['AI_Role'] = model.predict(X)
            def calc_custom_strategy_score(skill_list, selected_cats):
                score = 0
                for cat in selected_cats:
                    score += len(set(skill_list) & set(SKILL_CATEGORIES[cat]))
                return score
            candidates['Strategy_Score'] = candidates['LanguageHaveWorkedWith'].str.split(';').apply(lambda x: calc_custom_strategy_score(x, selected_skills))
            G_sub = G_full.subgraph(candidates['ResponseId'])
            candidates['Centrality'] = candidates['ResponseId'].map(nx.degree_centrality(G_sub))
            if len(selected_skills) == len(all_cats) or len(selected_skills) == 0:
                recommended = candidates.sort_values('Centrality', ascending=False).drop_duplicates(subset=['AI_Role']).head(5)
            else:
                recommended = candidates.sort_values(['Strategy_Score', 'Centrality'], ascending=False).head(5)
            recommended = recommended.sort_values('Centrality', ascending=False)
            recommended['Position'] = ["👑 Leader", "🥈 Tech Lead", "Member", "Member", "Member"]
            col_res1, col_res2 = st.columns([2, 1])
            with col_res1:
                st.success(f"✅ {strategy_label} チームが結成されました")
                st.table(recommended[['Position', 'ResponseId', 'AI_Role', 'LanguageHaveWorkedWith']])
                st.download_button("📤 チームリストを保存", recommended.to_csv(index=False).encode('utf-8'), "team_report.csv", "text/csv")
            with col_res2:
                st.write("📊 **チーム分析**")
                st.pyplot(draw_radar_chart(recommended))
        else:
            st.warning("モデルが見つかりません。")
else:
    st.error("データセットが見つかりません。")
