# circuit_generator.py (전체 수정)
import pandas as pd
import numpy as np
from detector.hole_detector import HoleDetector
from diagram import drawDiagram
from writeSPICE import toSPICE
from calcVoltageAndCurrent import calcCurrentAndVoltage
import networkx as nx
from networkx.readwrite import write_graphml
import matplotlib.pyplot as plt

def generate_circuit(
    all_comps: list,
    holes: list,
    wires: list,
    voltage: float,
    output_spice: str,
    output_img: str,
    hole_to_net: dict,
    power_plus: tuple[int, float] = None,
    power_minus: tuple[int, float] = None
):
    # 1) hole → (row, net) 매핑
    

    # 2) wires 기반 넷 병합 (Union-Find)
    parent = { net: net for net in set(hole_to_net.values()) }
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            parent[pv] = pu
    print("=== Wire Connections (Net Unions) ===")
    for net1, net2 in wires:
        print(f"Union: {net1} <--> {net2}")
        union(net1, net2)

    # 3) 컴포넌트 필터링 및 매핑 (Line_area 제외)
    comps = [c for c in all_comps if c['class'] != 'Line_area']
    mapped = []
    for idx, comp in enumerate(comps, start=1):
        pin_a, pin_b = comp['pins']
        def nearest_net(pt):
            x, y = pt
            closest = min(hole_to_net.keys(), key=lambda h: (h[0]-x)**2 + (h[1]-y)**2)
            return find(hole_to_net[closest])
        node1 = nearest_net(pin_a)
        node2 = nearest_net(pin_b)
        prefix = {'Resistor':'R','Diode':'D','LED':'L','Capacitor':'C','IC':'U'}.get(comp['class'], 'X')
        name = f"{prefix}{idx}"
        mapped.append({
            'name': name,
            'class': comp['class'],
            'value': comp['value'],
            'nodes': (node1, node2)
        })

    # 3.5) 연결 정보 출력
    print("=== Component to Net Mapping ===")
    for comp in mapped:
        print(f"{comp['name']} ({comp['class']}): Net1={comp['nodes'][0]}, Net2={comp['nodes'][1]}")

    # 4) DataFrame 생성: 두 핀 모두 컬럼에 저장
    df = pd.DataFrame([{  
        'name': m['name'],
        'class': m['class'],
        'value': m['value'],
        'node1_n': m['nodes'][0],
        'node2_n': m['nodes'][1],
    } for m in mapped])

    # ❶ 그래프 생성
    G = build_circuit_graph(mapped)
    save_circuit_graph(G, output_img.replace('.jpg', '.graphml'))
    # ❷ (선택) 파일로 저장
    write_graphml(G, output_img.replace('.jpg','.graphml'))
    # 또는 네트워크 직렬화: nx.write_gpickle(G, 'circuit.pkl')

    # 5) SPICE 넷리스트 생성
    toSPICE(df, voltage, output_spice)

    # 6) 회로도 이미지 생성 (flat list 사용)
    drawing = drawDiagram(voltage, mapped,wires,power_plus=power_plus,power_minus=power_minus)
    drawing.draw()          # 도면을 렌더링합니다 (schemdraw 내부적으로 필요 시 생략 가능)
    drawing.save(output_img)  # 파일로 바로 저장

    # 7) 선택적 해석: 레벨별 리스트 구성 후 호출
    circuit_levels = []
    for lvl, grp in df.groupby('node1_n', sort=False):
        level_comps = []
        for _, row in grp.iterrows():
            level_comps.append({
                'name': row['name'],
                'value': int(row['value']),
                'class': row['class']
            })
        circuit_levels.append(level_comps)
    R_th, I_tot, node_currents = calcCurrentAndVoltage(voltage, circuit_levels)
    print(f"[Circuit] 등가저항: {R_th}, 전체전류: {I_tot}")
    print("=== Node Voltages/ Currents per Level ===")
    for lvl_idx, currents in enumerate(node_currents):
        print(f"Level {lvl_idx+1}: currents = {currents}")

def build_circuit_graph(mapped_comps):
    G = nx.Graph()
    # 1) 노드 추가 (nets 튜플 → 문자열)
    for comp in mapped_comps:
        nets_str = ','.join(map(str, comp['nodes']))
        G.add_node(comp['name'],
                   comp_class=comp['class'],
                   value=comp['value'],
                   nets=nets_str)   # tuple → "1,2" 식 문자열

    # 2) net → 컴포넌트 역색인
    net_to_comps = {}
    for comp in mapped_comps:
        for net in comp['nodes']:
            net_to_comps.setdefault(net, []).append(comp['name'])

    # 3) 같은 net에 묶인 컴포넌트들끼리 엣지 추가 (nets set → 문자열)
    for net, clist in net_to_comps.items():
        for i in range(len(clist)):
            for j in range(i+1, len(clist)):
                u, v = clist[i], clist[j]
                if G.has_edge(u, v):
                    # 이미 있으면 기존 문자열 뒤에 추가
                    prev = G[u][v]['nets']
                    G[u][v]['nets'] = f"{prev},{net}"
                else:
                    G.add_edge(u, v, nets=str(net))

    return G

def save_circuit_graph(G, path_graphml):
    # GraphML로 저장
    write_graphml(G, path_graphml)

def visualize_circuit_graph(G, out_path='circuit_graph.png'):
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6,6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10)
    edge_labels = {(u,v): ','.join(map(str,data['nets'])) for u,v,data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_color='red', font_size=8)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Circuit graph saved to {out_path}")


# 예시 사용법:
if __name__ == "__main__":
    # mapped 리스트는 generate_circuit 내부에서 만든 것과 동일한 형태
    mapped = [
        {"name":"R1","class":"Resistor","value":100,"nodes":(1,2)},
        {"name":"C1","class":"Capacitor","value":0.001,"nodes":(2,3)},
        {"name":"LED1","class":"LED","value":0,"nodes":(3,0)}
    ]
    G = build_circuit_graph(mapped)
    visualize_circuit_graph(G)