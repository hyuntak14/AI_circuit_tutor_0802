import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter, deque

class CircuitComparer:
    def __init__(self, generated_graph, reference_graph, debug=False):
        """
        :param generated_graph: networkx.Graph containing edges with 'component' dicts
        :param reference_graph: networkx.Graph to compare against
        :param debug: bool, if True prints internal matching information
        """
        self.debug = debug
        # Extract and normalize components, excluding junctions
        self.components = self._extract_components(generated_graph)
        self.other_components = self._extract_components(reference_graph)
        # Build internal graphs without junctions
        self.graph = self._build_graph(self.components)
        self.other_graph = self._build_graph(self.other_components)

    def _extract_components(self, graph):
        comps = []
        for u, v, data in graph.edges(data=True):
            comp = data.get('component')
            if not isinstance(comp, dict):
                continue
            raw = comp.get('comp_class') or comp.get('type')
            norm = self.normalize_component_class(raw)
            if norm == 'Junction':
                continue
            comp_copy = comp.copy()
            comp_copy['norm_class'] = norm
            comp_copy['nodes'] = (u, v)
            comps.append(comp_copy)
        return comps

    def normalize_component_class(self, cls):
        if cls in ('V+', 'V-', 'VoltageSource'):
            return 'VoltageSource'
        if 'Resistor' in cls:
            return 'Resistor'
        if cls in ('Wire', 'net', 'junction', 'Junction'):
            return 'Junction'
        return cls

    def _build_graph(self, components):
        G = nx.Graph()
        for comp in components:
            n1, n2 = comp['nodes']
            G.add_edge(n1, n2, component=comp)
        return G

    def node_match_score(self):
        cnt1 = Counter(c['norm_class'] for c in self.components)
        cnt2 = Counter(c['norm_class'] for c in self.other_components)
        inter = sum((cnt1 & cnt2).values())
        union = sum((cnt1 | cnt2).values())
        score = inter / union if union else 1.0
        if self.debug:
            print(f"Node match: inter={inter}, union={union}, score={score}")
        return score

    def edge_match_score(self):
        def edge_types(comps):
            types = []
            for comp in comps:
                cls = comp['norm_class']
                types.append((cls, cls))
            return types

        et1 = Counter(edge_types(self.components))
        et2 = Counter(edge_types(self.other_components))
        inter = sum((et1 & et2).values())
        union = sum((et1 | et2).values())
        score = inter / union if union else 1.0
        if self.debug:
            print(f"Edge match: inter={inter}, union={union}, score={score}")
        return score

    def bfs_traversal_sequence(self, graph=None, components=None, start_class='VoltageSource'):
        if graph is None or components is None:
            graph = self.graph
            components = self.components
        seq = []
        start_nodes = [n for comp in components if comp['norm_class'] == start_class for n in comp['nodes']]
        if not start_nodes:
            return seq
        visited = set()
        queue = deque([start_nodes[0]])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            for nbr in graph.neighbors(node):
                comp = graph.edges[node, nbr]['component']
                cls = comp['norm_class']
                seq.append(cls)
                queue.append(nbr)
        if self.debug:
            print(f"BFS sequence: {seq}")
        return seq

    def sequence_similarity(self, seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m):
            for j in range(n):
                if seq1[i] == seq2[j]:
                    dp[i + 1][j + 1] = dp[i][j] + 1
                else:
                    dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
        lcs = dp[m][n]
        score = lcs / max(m, n) if max(m, n) else 1.0
        if self.debug:
            print(f"LCS length={lcs}, len1={m}, len2={n}, score={score}")
        return score

    def compute_similarity(self):
        ns = self.node_match_score()
        es = self.edge_match_score()
        seq1 = self.bfs_traversal_sequence()
        seq2 = self.bfs_traversal_sequence(self.other_graph, self.other_components)
        ts = self.sequence_similarity(seq1, seq2)
        sim = 0.3 * ns + 0.3 * es + 0.4 * ts
        if self.debug:
            print(f"NS={ns}, ES={es}, TS={ts}, SIM={sim}")
        return sim


    def visualize_comparison(self, save_path=None, show=True):
        """두 그래프를 나란히 시각화하여 비교"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        color_map = {
            'VoltageSource': '#FF6B6B', 'Resistor': '#4ECDC4', 'Capacitor': '#45B7D1',
            'Diode': '#96CEB4', 'LED': '#FFEAA7', 'IC': '#DDA0DD', 'Wire': '#95A5A6', 'Unknown': '#BDC3C7'
        }
        self._draw_single_graph(self.G1, ax1, "Generated Circuit", color_map)
        self._draw_single_graph(self.G2, ax2, "Reference Circuit", color_map)
        sim = self.compute_similarity()
        fig.suptitle(f'Circuit Comparison (Similarity: {sim:.3f})', fontsize=16, fontweight='bold')
        # 범례
        used = set()
        for G in [self.G1, self.G2]:
            for _, d in G.nodes(data=True):
                used.add(self.normalize_component_class(d.get('comp_class') or d.get('type') or 'Unknown'))
        legend = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=color_map[t], markersize=10, label=t)
                  for t in sorted(used)]
        fig.legend(handles=legend, loc='center', bbox_to_anchor=(0.5,0.02), ncol=len(legend))
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    def _draw_single_graph(self, G, ax, title, color_map):
        """단일 그래프를 그리는 헬퍼 함수"""
        pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50) if G.nodes() else {}
        node_colors, labels = [], {}
        for n, d in G.nodes(data=True):
            ct = self.normalize_component_class(d.get('comp_class') or d.get('type') or 'Unknown')
            node_colors.append(color_map.get(ct,'#BDC3C7'))
            lbl = str(n)
            v = d.get('value',0)
            if isinstance(v,(int,float)) and v!=0:
                suffix = {'Resistor':'Ω','VoltageSource':'V','Capacitor':'F'}.get(ct,'')
                lbl += f"\n{v}{suffix}"
            labels[n] = lbl
        nx.draw_networkx_nodes(G,pos,ax=ax,node_color=node_colors,node_size=1500,alpha=0.9,edgecolors='black')
        nx.draw_networkx_edges(G,pos,ax=ax,edge_color='#7F8C8D',width=2,alpha=0.7)
        nx.draw_networkx_labels(G,pos,labels,ax=ax,font_size=9,font_weight='bold')
        edge_lbl = {(u,v):data['nets'] for u,v,data in G.edges(data=True) if 'nets' in data}
        if edge_lbl:
            nx.draw_networkx_edge_labels(G,pos,edge_lbl,ax=ax,font_size=8,font_color='red')
        ax.set_title(f"{title}\nNodes: {len(G.nodes())}, Edges: {len(G.edges())}",fontsize=14,fontweight='bold')
        ax.axis('off')
        stats = self._get_graph_stats_text(G)
        ax.text(0.02,0.98,stats,transform=ax.transAxes,fontsize=10,verticalalignment='top',
                bbox=dict(boxstyle='round',facecolor='wheat',alpha=0.8))

    def _get_graph_stats_text(self, G):
        """그래프 통계 정보 텍스트 생성"""
        counts = Counter()
        for _, d in G.nodes(data=True):
            counts[self.normalize_component_class(d.get('comp_class') or d.get('type') or 'Unknown')] += 1
        lines = ["Components:"] + [f"  {t}: {counts[t]}" for t in sorted(counts)]
        return "\n".join(lines)
