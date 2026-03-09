import pandas as pd
import itertools
import os

def process_data(input_path, output_dir, n_rows=1000):
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Reading {n_rows} rows from {input_path}...")
    # 必要な列だけを読み込み、欠損値を削除する
    df = pd.read_csv(input_path, nrows=n_rows)
    df = df[['ResponseId', 'LanguageHaveWorkedWith', 'DevType']].dropna()
    
    # スキルをリスト化（文字列であることを保証）
    df['skill_list'] = df['LanguageHaveWorkedWith'].astype(str).str.split(';')

    print(f"Generating edges for {len(df)} users...")
    edges = []
    users = df.to_dict('records')
    
    for u1, u2 in itertools.combinations(users, 2):
        # set()に変換して共通スキルを抽出
        common = set(u1['skill_list']) & set(u2['skill_list'])
        if len(common) >= 3: 
            edges.append({
                'source': u1['ResponseId'], 
                'target': u2['ResponseId'], 
                'weight': len(common)
            })

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(edges).to_csv(os.path.join(output_dir, 'edges.csv'), index=False)
    df.to_csv(os.path.join(output_dir, 'nodes.csv'), index=False)
    print(f"Success! Nodes: {len(df)}, Edges: {len(edges)}")

if __name__ == "__main__":
    process_data("data/raw/survey_results_public.csv", "data/processed")
