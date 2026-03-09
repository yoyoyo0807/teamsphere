import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import os

def train_role_model(nodes_path, model_output_path):
    if not os.path.exists(nodes_path):
        print("Error: nodes.csv not found.")
        return

    df = pd.read_csv(nodes_path)
    # スキルを分割
    df['skill_list'] = df['LanguageHaveWorkedWith'].fillna('').str.split(';')
    
    # スキルのベクトル化
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df['skill_list'])
    
    # 目的変数（職種の1番目を代表とする）
    y = df['DevType'].fillna('Unknown').apply(lambda x: x.split(';')[0])
    
    # 学習
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # 保存
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, 'wb') as f:
        pickle.dump((model, mlb), f)
    print(f"ML Model trained and saved to {model_output_path}")

if __name__ == "__main__":
    train_role_model("data/processed/nodes.csv", "models/role_model.pkl")
