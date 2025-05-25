import xgboost as xgb
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

def extract_feature_interactions(model):
    booster = model.get_booster()
    dump = booster.get_dump(with_stats=False)
    
    interactions = set()
    for tree in dump:
        lines = tree.split('\n')
        features_in_tree = set()
        for line in lines:
            if '[' in line:
                feature = line.split('[')[1].split('<')[0].strip()
                features_in_tree.add(feature)
        for f1 in features_in_tree:
            for f2 in features_in_tree:
                if f1 != f2:
                    interactions.add(tuple(sorted((f1, f2))))
    return interactions

def plot_feature_interaction_graph(model, allowed_constraints=None):
    interactions = extract_feature_interactions(model)
    
    G = nx.Graph()
    G.add_edges_from(interactions)
    
    pos = nx.spring_layout(G)
    colors = []
    
    for edge in G.edges():
        if allowed_constraints is None:
            colors.append("gray")
        else:
            allowed = False
            for constraint_group in allowed_constraints:
                if edge[0] in constraint_group and edge[1] in constraint_group:
                    allowed = True
                    break
            colors.append("green" if allowed else "red")

    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, edge_color=colors, node_color='lightblue', node_size=2000, font_size=12)
    plt.title("Feature Interaction Graph (Green: Allowed, Red: Violated)")
    plt.show()
