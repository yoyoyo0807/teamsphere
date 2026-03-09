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

    # --- 全体ネットワークの表示 ---
    st.subheader("🕸️ 全体コミュニティ構造")
    fig_all, ax_all = plt.subplots(figsize=(12, 5))
    pos = nx.spring_layout(G_full, k=0.15, seed=42)
    node_id_to_color = nodes_df.set_index('ResponseId')['community_id'].to_dict()
    node_colors = [node_id_to_color.get(n, 0) for n in G_full.nodes()]
    nx.draw(G_full, pos, with_labels=False, node_color=node_colors, cmap=plt.cm.rainbow, node_size=20, alpha=0.3, ax=ax_all)
    st.pyplot(fig_all)

    st.divider()

    # --- チーム選抜設定 ---
    st.subheader("🎯 チーム選抜・戦略設定")
    col_set1, col_set2 = st.columns(2)
    
    with col_set1:
        mode = st.radio("🔎 選抜アプローチ", ["コミュニティから選抜", "特定個人(自分)に合わせる"])
    
    with col_set2:
        st.write("💡 重視するスキルカテゴリを選択してください")
        selected_skills = []
        # 5つの要素を選択制にする
        cols = st.columns(3)
        all_cats = list(SKILL_CATEGORIES.keys())
        for i, cat in enumerate(all_cats):
            if cols[i % 3].checkbox(cat, value=True):
                selected_skills.append(cat)
        
        # 戦略の自動判定
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
        with open(model_path, 'rb') as f:
            model, mlb = pickle.load(f)
        
        X = mlb.transform(candidates['LanguageHaveWorkedWith'].str.split(';'))
        candidates['AI_Role'] = model.predict(X)
        
        # 選択されたカテゴリに基づいたスコアリング
        def calc_custom_strategy_score(skill_list, selected_cats):
            score = 0
            for cat in selected_cats:
                score += len(set(skill_list) & set(SKILL_CATEGORIES[cat]))
            return score

        candidates['Strategy_Score'] = candidates['LanguageHaveWorkedWith'].str.split(';').apply(lambda x: calc_custom_strategy_score(x, selected_skills))
        
        G_sub = G_full.subgraph(candidates['ResponseId'])
        candidates['Centrality'] = candidates['ResponseId'].map(nx.degree_centrality(G_sub))

        # 選抜ロジック
        if len(selected_skills) == len(all_cats) or len(selected_skills) == 0:
            # バランス重視：役割の多様性と中心性を優先
            recommended = candidates.sort_values('Centrality', ascending=False).drop_duplicates(subset=['AI_Role']).head(5)
        else:
            # 特化型：戦略スコアが高い順に選ぶ
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
    st.error("データセットが見つかりません。")
