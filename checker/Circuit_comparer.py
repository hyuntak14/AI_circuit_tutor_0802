# Circuit_comparer.py

import networkx as nx
from collections import Counter

class CircuitComparer:
    def __init__(self, graph1: nx.Graph, graph2: nx.Graph):
        self.G1 = graph1
        self.G2 = graph2

    def node_match_score(self):
        cls1 = [
            data.get('comp_class') or data.get('type') or 'Unknown'
            for _, data in self.G1.nodes(data=True)
        ]
        cls2 = [
            data.get('comp_class') or data.get('type') or 'Unknown'
            for _, data in self.G2.nodes(data=True)
        ]
        c1, c2 = Counter(cls1), Counter(cls2)
        inter = sum(min(c1[k], c2.get(k, 0)) for k in c1)
        union = sum(max(c1.get(k,0), c2.get(k,0)) for k in set(c1)|set(c2))
        return inter/union if union else 1.0

    def edge_match_score(self):
        def get_cls(u, G):
            return G.nodes[u].get('comp_class') or G.nodes[u].get('type') or 'Unknown'

        pairs1 = []
        for u, v in self.G1.edges():
            cls_u = get_cls(u, self.G1)
            cls_v = get_cls(v, self.G1)
            # 클래스 이름 기준으로 정렬된 튜플
            pair = tuple(sorted((cls_u, cls_v)))
            pairs1.append(pair)

        pairs2 = []
        for u, v in self.G2.edges():
            cls_u = get_cls(u, self.G2)
            cls_v = get_cls(v, self.G2)
            pair = tuple(sorted((cls_u, cls_v)))
            pairs2.append(pair)

        c1, c2 = Counter(pairs1), Counter(pairs2)
        inter = sum(min(c1[p], c2.get(p, 0)) for p in c1)
        union = sum(max(c1.get(p,0), c2.get(p,0)) for p in set(c1)|set(c2))
        return inter/union if union else 1.0

    def compute_similarity(self, alpha=0.5):
        ns = self.node_match_score()
        es = self.edge_match_score()
        return alpha * ns + (1 - alpha) * es
