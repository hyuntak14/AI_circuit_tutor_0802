import networkx as nx
from collections import Counter, deque
import matplotlib.pyplot as plt

class CircuitComparer:
    def __init__(self, graph1: nx.Graph, graph2: nx.Graph, debug=False):
        self.G1 = graph1
        self.G2 = graph2
        self.debug = debug

    def normalize_component_class(self, cls):
        """컴포넌트 클래스 이름을 정규화"""
        if not cls:
            return 'Unknown'
        # VoltageSource 관련 정규화
        if cls in ['VoltageSource', 'V+', 'V-']:
            return 'VoltageSource'
        cls_map = {
            'Unknown': 'Unknown',
            'Resistor': 'Resistor',
            'Capacitor': 'Capacitor',
            'Diode': 'Diode',
            'LED': 'LED',
            'IC': 'IC',
            'Line_area': 'Wire',
            'Wire': 'Wire'
        }
        return cls_map.get(cls, cls)

    def node_match_score(self):
        """노드(컴포넌트) 매칭 점수 계산"""
        cls1 = [
            self.normalize_component_class(
                data.get('comp_class') or data.get('type') or 'Unknown'
            )
            for _, data in self.G1.nodes(data=True)
        ]
        cls2 = [
            self.normalize_component_class(
                data.get('comp_class') or data.get('type') or 'Unknown'
            )
            for _, data in self.G2.nodes(data=True)
        ]
        c1, c2 = Counter(cls1), Counter(cls2)
        if self.debug:
            print("\n=== Node Matching Debug ===")
            print(f"Graph1 components: {dict(c1)}")
            print(f"Graph2 components: {dict(c2)}")
            for k in sorted(set(c1) | set(c2)):
                match = min(c1.get(k,0), c2.get(k,0))
                total = max(c1.get(k,0), c2.get(k,0))
                ratio = match/total if total>0 else 0
                print(f"  {k}: {match}/{total} = {ratio:.3f}")
        all_classes = set(c1.keys()) | set(c2.keys())
        intersection = sum(min(c1.get(k,0), c2.get(k,0)) for k in all_classes)
        union = sum(max(c1.get(k,0), c2.get(k,0)) for k in all_classes)
        return intersection/union if union>0 else 1.0

    def edge_match_score(self):
        """엣지(연결) 매칭 점수 계산"""
        def norm_cls(node, G):
            cls = G.nodes[node].get('comp_class') or G.nodes[node].get('type') or 'Unknown'
            return self.normalize_component_class(cls)
        pairs1 = [tuple(sorted((norm_cls(u,self.G1), norm_cls(v,self.G1)))) for u,v in self.G1.edges()]
        pairs2 = [tuple(sorted((norm_cls(u,self.G2), norm_cls(v,self.G2)))) for u,v in self.G2.edges()]
        c1, c2 = Counter(pairs1), Counter(pairs2)
        if self.debug:
            print("\n=== Edge Matching Debug ===")
            print(f"Graph1 connections: {dict(c1)}")
            print(f"Graph2 connections: {dict(c2)}")
            for p in sorted(set(c1)|set(c2)):
                match = min(c1.get(p,0), c2.get(p,0))
                total = max(c1.get(p,0), c2.get(p,0))
                ratio = match/total if total>0 else 0
                print(f"  {p}: {match}/{total} = {ratio:.3f}")
        all_pairs = set(c1.keys()) | set(c2.keys())
        intersection = sum(min(c1.get(p,0), c2.get(p,0)) for p in all_pairs)
        union = sum(max(c1.get(p,0), c2.get(p,0)) for p in all_pairs)
        return intersection/union if union>0 else 1.0

    def bfs_traversal_sequence(self, G):
        """Voltage source부터 BFS 순회하며 부품 클래스 순서를 반환"""
        # 시작 노드: VoltageSource
        start_nodes = [n for n,d in G.nodes(data=True)
                       if self.normalize_component_class(d.get('comp_class') or d.get('type') or '')=='VoltageSource']
        if not start_nodes:
            start_nodes = [next(iter(G.nodes()), None)]
        visited = set()
        seq = []
        for start in start_nodes:
            if start is None: continue
            queue = deque([start])
            visited.add(start)
            while queue:
                node = queue.popleft()
                comp_type = self.normalize_component_class(
                    G.nodes[node].get('comp_class') or G.nodes[node].get('type') or '')
                seq.append(comp_type)
                for nbr in sorted(G.neighbors(node)):
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
        if self.debug:
            print(f"BFS sequence: {seq}")
        return seq

    def sequence_similarity(self, seq1, seq2):
        """LCS 기반 시퀀스 유사도 계산"""
        m, n = len(seq1), len(seq2)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                if seq1[i]==seq2[j]:
                    dp[i+1][j+1] = dp[i][j]+1
                else:
                    dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
        lcs = dp[m][n]
        return lcs/max(m,n) if max(m,n)>0 else 1.0

    def traversal_match_score(self):
        """BFS 순회 시퀀스 간 매칭 점수"""
        seq1 = self.bfs_traversal_sequence(self.G1)
        seq2 = self.bfs_traversal_sequence(self.G2)
        score = self.sequence_similarity(seq1, seq2)
        if self.debug:
            print(f"Sequence similarity: {score:.3f}")
        return score

    def compute_similarity(self, alpha=0.3, beta=0.3, gamma=0.4):
        """노드, 엣지, BFS 시퀀스 가중치 결합 최종 유사도"""
        ns = self.node_match_score()
        es = self.edge_match_score()
        ts = self.traversal_match_score()
        sim = alpha*ns + beta*es + gamma*ts
        if self.debug:
            print("\n=== Final Similarity Debug ===")
            print(f"Node score: {ns:.3f}, Edge score: {es:.3f}, Sequence score: {ts:.3f}")
            print(f"Weights: alpha={alpha}, beta={beta}, gamma={gamma}")
            print(f"Overall similarity: {sim:.3f}")
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
