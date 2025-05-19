# Circuit_comparer.py

import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

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
        
        # 기타 정규화 (필요시 추가)
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
        # 각 그래프에서 컴포넌트 클래스 추출 및 정규화
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
        
        # 디버그 정보 출력
        if self.debug:
            print("\n=== Node Matching Debug ===")
            print(f"Graph1 components: {dict(c1)}")
            print(f"Graph2 components: {dict(c2)}")
        
        # 교집합과 합집합 계산
        all_classes = set(c1.keys()) | set(c2.keys())
        intersection = sum(min(c1.get(k, 0), c2.get(k, 0)) for k in all_classes)
        union = sum(max(c1.get(k, 0), c2.get(k, 0)) for k in all_classes)
        
        score = intersection / union if union > 0 else 1.0
        
        if self.debug:
            print(f"Intersection: {intersection}")
            print(f"Union: {union}")
            print(f"Node match score: {score:.3f}")
            
            # 각 클래스별 매칭 세부사항
            for cls in all_classes:
                match = min(c1.get(cls, 0), c2.get(cls, 0))
                total = max(c1.get(cls, 0), c2.get(cls, 0))
                print(f"  {cls}: {match}/{total} = {match/total:.3f}" if total > 0 else f"  {cls}: 0/0")
        
        return score

    def edge_match_score(self):
        """엣지(연결) 매칭 점수 계산"""
        def get_normalized_class(node, G):
            cls = G.nodes[node].get('comp_class') or G.nodes[node].get('type') or 'Unknown'
            return self.normalize_component_class(cls)

        # 각 그래프의 엣지를 정규화된 클래스 쌍으로 변환
        pairs1 = []
        for u, v in self.G1.edges():
            cls_u = get_normalized_class(u, self.G1)
            cls_v = get_normalized_class(v, self.G1)
            # 클래스 이름 기준으로 정렬된 튜플 (순서 무관하게 만들기 위해)
            pair = tuple(sorted((cls_u, cls_v)))
            pairs1.append(pair)

        pairs2 = []
        for u, v in self.G2.edges():
            cls_u = get_normalized_class(u, self.G2)
            cls_v = get_normalized_class(v, self.G2)
            pair = tuple(sorted((cls_u, cls_v)))
            pairs2.append(pair)

        c1, c2 = Counter(pairs1), Counter(pairs2)
        
        # 디버그 정보 출력
        if self.debug:
            print("\n=== Edge Matching Debug ===")
            print(f"Graph1 connections: {dict(c1)}")
            print(f"Graph2 connections: {dict(c2)}")
        
        # 교집합과 합집합 계산
        all_pairs = set(c1.keys()) | set(c2.keys())
        intersection = sum(min(c1.get(p, 0), c2.get(p, 0)) for p in all_pairs)
        union = sum(max(c1.get(p, 0), c2.get(p, 0)) for p in all_pairs)
        
        score = intersection / union if union > 0 else 1.0
        
        if self.debug:
            print(f"Intersection: {intersection}")
            print(f"Union: {union}")
            print(f"Edge match score: {score:.3f}")
            
            # 각 연결 쌍별 매칭 세부사항
            for pair in all_pairs:
                match = min(c1.get(pair, 0), c2.get(pair, 0))
                total = max(c1.get(pair, 0), c2.get(pair, 0))
                print(f"  {pair}: {match}/{total} = {match/total:.3f}" if total > 0 else f"  {pair}: 0/0")
        
        return score

    def detailed_comparison(self):
        """상세한 비교 정보 반환"""
        node_score = self.node_match_score()
        edge_score = self.edge_match_score()
        
        return {
            'node_score': node_score,
            'edge_score': edge_score,
            'graph1_nodes': len(self.G1.nodes()),
            'graph2_nodes': len(self.G2.nodes()),
            'graph1_edges': len(self.G1.edges()),
            'graph2_edges': len(self.G2.edges())
        }

    def compute_similarity(self, alpha=0.5):
        """최종 유사도 계산"""
        node_score = self.node_match_score()
        edge_score = self.edge_match_score()
        similarity = alpha * node_score + (1 - alpha) * edge_score
        
        if self.debug:
            print(f"\n=== Final Similarity ===")
            print(f"Node score: {node_score:.3f}")
            print(f"Edge score: {edge_score:.3f}")
            print(f"Alpha (node weight): {alpha}")
            print(f"Beta (edge weight): {1-alpha}")
            print(f"Final similarity: {similarity:.3f}")
        
        return similarity

    def visualize_comparison(self, save_path=None, show=True):
        """두 그래프를 나란히 시각화하여 비교"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 컴포넌트 타입별 색상 정의
        color_map = {
            'VoltageSource': '#FF6B6B',  # 빨간색
            'Resistor': '#4ECDC4',       # 청록색
            'Capacitor': '#45B7D1',      # 파란색
            'Diode': '#96CEB4',          # 초록색
            'LED': '#FFEAA7',            # 노란색
            'IC': '#DDA0DD',             # 보라색
            'Wire': '#95A5A6',           # 회색
            'Unknown': '#BDC3C7'         # 연회색
        }
        
        # Graph 1 시각화
        self._draw_single_graph(self.G1, ax1, "Generated Circuit", color_map)
        
        # Graph 2 시각화  
        self._draw_single_graph(self.G2, ax2, "Reference Circuit", color_map)
        
        # 전체 제목
        similarity = self.compute_similarity()
        fig.suptitle(f'Circuit Comparison (Similarity: {similarity:.3f})', fontsize=16, fontweight='bold')
        
        # 범례 추가
        legend_elements = []
        used_types = set()
        for G in [self.G1, self.G2]:
            for _, data in G.nodes(data=True):
                comp_type = self.normalize_component_class(
                    data.get('comp_class') or data.get('type') or 'Unknown'
                )
                used_types.add(comp_type)
        
        for comp_type in sorted(used_types):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color_map.get(comp_type, '#BDC3C7'),
                                            markersize=10, label=comp_type))
        
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=len(used_types))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"그래프 비교 이미지가 저장되었습니다: {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _draw_single_graph(self, G, ax, title, color_map):
        """단일 그래프를 그리는 헬퍼 함수"""
        # 레이아웃 계산
        if len(G.nodes()) > 0:
            pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
        else:
            pos = {}
        
        # 노드 색상 및 라벨 설정
        node_colors = []
        node_labels = {}
        
        for node, data in G.nodes(data=True):
            comp_type = self.normalize_component_class(
                data.get('comp_class') or data.get('type') or 'Unknown'
            )
            node_colors.append(color_map.get(comp_type, '#BDC3C7'))
            
            # 라벨에 타입과 값 정보 포함
            label = str(node)
            if 'value' in data and data['value'] != 0:
                if comp_type == 'Resistor':
                    label += f"\n{data['value']}Ω"
                elif comp_type == 'VoltageSource':
                    label += f"\n{data['value']}V"
                elif comp_type == 'Capacitor':
                    label += f"\n{data['value']}F"
            node_labels[node] = label
        
        # 노드 그리기
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                              node_size=1500, alpha=0.9, edgecolors='black', linewidths=1.5)
        
        # 엣지 그리기
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#7F8C8D', 
                              width=2, alpha=0.7)
        
        # 라벨 그리기
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, 
                               font_size=9, font_weight='bold')
        
        # 엣지 라벨 (nets 정보가 있으면)
        edge_labels = {}
        for u, v, data in G.edges(data=True):
            if 'nets' in data:
                edge_labels[(u, v)] = str(data['nets'])
        
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, 
                                        font_size=8, font_color='red')
        
        # 제목 및 정보
        ax.set_title(f"{title}\nNodes: {len(G.nodes())}, Edges: {len(G.edges())}", 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # 통계 정보 텍스트
        stats_text = self._get_graph_stats_text(G)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def _get_graph_stats_text(self, G):
        """그래프 통계 정보 텍스트 생성"""
        # 컴포넌트 타입별 개수
        type_counts = Counter()
        for _, data in G.nodes(data=True):
            comp_type = self.normalize_component_class(
                data.get('comp_class') or data.get('type') or 'Unknown'
            )
            type_counts[comp_type] += 1
        
        stats_lines = ["Components:"]
        for comp_type, count in sorted(type_counts.items()):
            stats_lines.append(f"  {comp_type}: {count}")
        
        return "\n".join(stats_lines)

# 예제 사용법 및 테스트 코드
if __name__ == "__main__":
    # 테스트용 그래프 생성
    G1 = nx.Graph()
    G1.add_node('R1', comp_class='Resistor', value=100)
    G1.add_node('V1', comp_class='VoltageSource', value=5)
    G1.add_node('C1', comp_class='Capacitor', value=0.001)
    G1.add_edge('R1', 'V1', nets='1')
    G1.add_edge('V1', 'C1', nets='2')
    
    G2 = nx.Graph()
    G2.add_node('R1', comp_class='Resistor', value=100)
    G2.add_node('V+', type='V+', value=5)  # 다른 타입명 사용
    G2.add_node('C1', comp_class='Capacitor', value=0.001)
    G2.add_edge('R1', 'V+', nets='1')
    G2.add_edge('V+', 'C1', nets='2')
    
    # 비교 수행
    comparer = CircuitComparer(G1, G2, debug=True)
    
    # 그래프 정보 출력
    comparer.print_graph_info()
    
    # 상세 비교
    details = comparer.detailed_comparison()
    print(f"\n=== Detailed Comparison ===")
    for key, value in details.items():
        print(f"{key}: {value}")
    
    # 최종 유사도
    similarity = comparer.compute_similarity()
    print(f"\n=== Final Result ===")
    print(f"Overall Similarity: {similarity:.3f}")