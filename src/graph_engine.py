import pandas as pd
import networkx as nx
import community as community_louvain
import os

def detect_communities(nodes_path, edges_path, output_dir):
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        G.add_node(row['ResponseId'], dev_type=row['DevType'])
    for _, row in edges_df.iterrows():
        G.add_edge(row['source'], row['target'], weight=row['weight'])
    partition = community_louvain.best_partition(G, weight='weight')
    nodes_df['community_id'] = nodes_df['ResponseId'].map(partition)
    output_path = os.path.join(output_dir, 'nodes_with_communities.csv')
    nodes_df.to_csv(output_path, index=False)
    print(f"Community detection complete: Result saved to {output_path}")

if __name__ == "__main__":
    detect_communities("data/processed/nodes.csv", "data/processed/edges.csv", "data/processed")
