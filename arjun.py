import xgboost as xgb
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict, Counter


def extract_feature_interactions(model, feature_names=None):
    booster = model.get_booster()
    dump = booster.get_dump(with_stats=False)

    interaction_counter = Counter()
    
    for tree in dump:
        lines = tree.split('\n')
        used_features = set()
        for line in lines:
            if '[' in line:
                feature = line.split('[')[1].split('<')[0].strip()
                if feature_names and feature.startswith('f'):
                    # Map "f0" -> actual feature name
                    index = int(feature[1:])
                    if index < len(feature_names):
                        feature = feature_names[index]
                used_features.add(feature)
        for f1 in used_features:
            for f2 in used_features:
                if f1 != f2:
                    key = tuple(sorted((f1, f2)))
                    interaction_counter[key] += 1
    return interaction_counter


def plot_feature_interaction_graph(model, interaction_constraints=None, feature_names=None):
    interactions = extract_feature_interactions(model, feature_names)

    G = nx.Graph()
    for (f1, f2), weight in interactions.items():
        G.add_edge(f1, f2, weight=weight)

    pos = nx.spring_layout(G, seed=42)
    edge_colors = []
    edge_weights = []

    for u, v, data in G.edges(data=True):
        weight = data['weight']
        edge_weights.append(weight / max(1, max(interactions.values())))  # normalize
        if interaction_constraints:
            allowed = False
            for group in interaction_constraints:
                if u in group and v in group:
                    allowed = True
                    break
            edge_colors.append("green" if allowed else "red")
        else:
            edge_colors.append("gray")

    plt.figure(figsize=(14, 10))
    nx.draw(
        G, pos,
        with_labels=True,
        edge_color=edge_colors,
        width=[w * 3 for w in edge_weights],
        node_color='lightblue',
        node_size=2000,
        font_size=12
    )
    plt.title("Feature Interaction Graph (Edge thickness = frequency)")
    plt.show()
