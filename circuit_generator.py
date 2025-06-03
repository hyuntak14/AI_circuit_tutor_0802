# circuit_generator.py (전체 수정)
import os
import pandas as pd
import numpy as np
from detector.hole_detector import HoleDetector
from diagram import drawDiagram
from writeSPICE import toSPICE
from calcVoltageAndCurrent import calcCurrentAndVoltage
import networkx as nx
from networkx.readwrite import write_graphml
import matplotlib.pyplot as plt
from diagram import draw_connectivity_graph
import glob
from checker.Circuit_comparer import CircuitComparer
import matplotlib
import tkinter as tk
from tkinter import messagebox
matplotlib.use('TkAgg')  # 또는 'Qt5Agg', 'WxAgg' 등 다른 대화형 백엔드
# 이후 schemdraw 코드 실행
import cv2
import os, glob, re
from diagram import validate_circuit_connectivity,generate_circuit_from_spice
from new_diagram import draw_new_diagram

# 실습 주제 맵
topic_map = {
    0: "test용 회로", 1: "병렬회로", 2: "직렬회로", 3: "키르히호프 1법칙", 4: "키르히호프 2법칙",
    5: "중첩의 원리", 6: "오실로스코프 실습1", 7: "오실로스코프 실습2",
    8: "반파정류회로", 9: "반파정류회로2", 10: "비반전 증폭기"
}

def compare_and_notify(G, output_img, checker_dir="checker"):
    # 1) 파일 수집
    files = glob.glob(os.path.join(checker_dir, "*.graphml"))
    if not files:
        print("[비교] 기준 .graphml 파일이 없습니다.")
        return

    # 2) 유사도 계산
    sims = []
    for path in files:
        try:
            G_ref = nx.read_graphml(path)
            sim = CircuitComparer(G, G_ref).compute_similarity()
            sims.append((os.path.basename(path), sim))
        except Exception as e:
            print(f"[비교 실패] {path}: {e}")

    # 3) 결과 출력 (Top3)
    sims.sort(key=lambda x: x[1], reverse=True)
    print("\n=== 유사도 TOP 3 ===")
    for i, (fn, sc) in enumerate(sims[:3], 1):
        print(f"{i}. {fn}: {sc:.3f}")

    # 4) 최우수 항목 팝업 알림
    best_fn, _ = sims[0]
    m = re.search(r"(\d+)", best_fn)
    topic = topic_map.get(int(m.group(1))) if m else None
    msg = f"본 회로는 {topic} 실습 주제입니다." if topic else "실습 주제를 알 수 없습니다."

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("회로 비교 결과", msg)
    root.destroy()

# circuit_generator.py의 generate_circuit 함수 부분 수정
def generate_circuit(
    all_comps: list,
    holes: list,
    wires: list,
    voltage: float,  # 대표 전압 (첫 번째 전원)
    output_spice: str,
    output_img: str,
    hole_to_net: dict,
    power_pairs: list[tuple[int, float, int, float]] = None  # [(net_p, x_p, net_m, x_m), ...]
):
    # 1) wires 기반 넷 병합 (Union-Find)
    parent = {net: net for net in set(hole_to_net.values())}

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

    # 2) 전원 net 매핑
    mapped_powers = []
    for raw_np, x_np, raw_nm, x_nm in power_pairs or []:
        mapped_powers.append((find(raw_np), x_np, find(raw_nm), x_nm))
    power_pairs = mapped_powers

    # 3) 컴포넌트 필터링 및 매핑
    comps = [c for c in all_comps if c['class'] != 'Line_area']
    mapped = []
    for idx, comp in enumerate(comps, start=1):
        # ① 핀 정보가 정확히 2개인지 체크
        pins = comp.get('pins', [])
        if len(pins) != 2:
            # 잘못된 핀 개수는 건너뛰거나, 로그를 남기고 다음 컴포넌트로
            print(f"[경고] 컴포넌트 #{idx}({comp['class']}) 핀 개수 오류: {pins}")
            continue
        pin_a, pin_b = pins

        def nearest_net(pt):
            x, y = pt
            closest = min(hole_to_net.keys(), key=lambda h: (h[0] - x) ** 2 + (h[1] - y) ** 2)
            return find(hole_to_net[closest])

        node1 = nearest_net(pin_a)
        node2 = nearest_net(pin_b)
        prefix = {'Resistor': 'R', 'Diode': 'D', 'LED': 'L', 'Capacitor': 'C', 'IC': 'U','VoltageSource': 'V', 'V+': 'V','V-': 'V'}.get(comp['class'], 'X')
        name = f"{prefix}{idx}"
        mapped.append({
            'name': name,
            'class': comp['class'],
            'value': comp['value'],
            'nodes': (node1, node2)
        })

    print("=== Component to Net Mapping ===")
    for comp in mapped:
        print(f"{comp['name']} ({comp['class']}): Net1={comp['nodes'][0]}, Net2={comp['nodes'][1]}")

    # 🔧 4) 다중 전원 소스 추가 (수정된 부분)
    print("=== Adding Multiple Power Sources ===")
    for i, (net_p, x_p, net_m, x_m) in enumerate(power_pairs, start=1):
        vs_name = f"V{i}"
        
        # 각 전원마다 개별 전압 설정 가능하도록 확장
        # 현재는 대표 전압(voltage)을 모든 전원에 적용
        # 필요시 power_pairs에 전압 정보도 포함하도록 확장 가능
        vs_comp = {
            'name': vs_name,
            'class': 'VoltageSource',
            'value': voltage,  # 향후 개별 전압으로 확장 가능
            'nodes': (net_p, net_m)
        }
        mapped.append(vs_comp)
        print(f"{vs_name} (VoltageSource): Net1={net_p}, Net2={net_m}, Value={voltage}V")

    # 5) DataFrame 구성
    df = pd.DataFrame([{
        'name': m['name'],
        'class': m['class'],
        'value': m['value'],
        'node1_n': m['nodes'][0],
        'node2_n': m['nodes'][1],
    } for m in mapped])

    # 6) 그래프 저장
    G = build_circuit_graph(mapped)
    save_circuit_graph(G, output_img.replace('.jpg', '.graphml'))
    write_graphml(G, output_img.replace('.jpg', '.graphml'))

    # 7) SPICE 저장 (다중 전원 지원)
    toSPICE_multi_power(df, power_pairs, voltage, output_spice)

    # 8) 각 전원별 회로도 및 연결 그래프 시각화 (다중 전원 지원)
    for i, (net_p, x_p, net_m, x_m) in enumerate(power_pairs, 1):
        if i == 1:
            path = output_img
        else:
            path = output_img.replace('.jpg', f'_pwr{i}.jpg')
        
        # ✅ 연결성 검증 추가
        connectivity_report = validate_circuit_connectivity(G)
        
        if not connectivity_report['is_connected']:
            print(f"\n🚨 전원 {i} 회로 연결성 문제:")
            for issue in connectivity_report['issues']:
                print(f"  - {issue}")
            
            # 연결되지 않은 경우 상세 정보 출력
            print("연결된 그룹:")
            for j, group in enumerate(connectivity_report['groups']):
                comp_names = [comp['name'] for comp in group]
                print(f"  그룹 {j+1}: {comp_names}")
        
        # ✅ 연결 그래프 생성
        try:
            from diagram import draw_connectivity_graph_from_nx
            draw_connectivity_graph_from_nx(G, output_path=path.replace('.jpg', '_graph.png'))
        except Exception as e:
            print(f"연결성 그래프 생성 실패 (전원 {i}): {e}")
        
        # ✅ 회로도 생성 (다중 전원 지원)
        try:
            from diagram import drawDiagramFromGraph_with_connectivity_check
            print(f"전원 {i} 회로도 생성 중...")
            
            # power_pairs 리스트를 G.graph에 저장
            G.graph['power_pairs'] = power_pairs
            G.graph['current_power_index'] = i - 1  # 현재 그리는 전원의 인덱스
            
            d = drawDiagramFromGraph_with_connectivity_check(G, voltage)
            
            if d:
                try:
                    d.draw()
                    d.save(path)
                    
                    # 연결성 문제가 있으면 파일명에 표시
                    if not connectivity_report['is_connected']:
                        disconnected_path = path.replace('.jpg', '_DISCONNECTED.jpg')
                        d.save(disconnected_path)
                        print(f"⚠️  연결 끊어진 회로도 저장: {disconnected_path}")
                    else:
                        print(f"✅ 전원 {i} 회로도 저장: {path}")
                    
                except Exception as save_error:
                    print(f"전원 {i} 회로도 저장 실패: {save_error}")
            else:
                print(f"❌ 전원 {i} 회로도 생성 실패")
                
        except Exception as diagram_error:
            print(f"전원 {i} 회로도 생성 오류: {diagram_error}")
    
    # 9) 비교 (다중 전원 고려)
    try:
        compare_and_notify(G, output_img, checker_dir="checker")
    except Exception as e:
        print(f"[오류] 회로 비교 실패: {e}")
    
    print(f"\n✅ 다중 전원 회로 생성 완료!")
    print(f"   - 총 전원 개수: {len(power_pairs)}")
    print(f"   - 컴포넌트 개수: {len([m for m in mapped if m['class'] != 'VoltageSource'])}")

    return mapped, hole_to_net


def toSPICE_multi_power(df, power_pairs, default_voltage, output_file):
    """
    다중 전원을 지원하는 SPICE 넷리스트 생성
    """
    with open(output_file, 'w') as f:
        f.write("* Multi-Power Circuit Netlist\n")
        f.write(f"* Generated with {len(power_pairs)} power sources\n")
        f.write("* \n")
        
        # 전압원들 먼저 출력
        for i, (net_p, _, net_m, _) in enumerate(power_pairs, 1):
            f.write(f"V{i} {net_p} {net_m} {default_voltage}\n")
        
        # 다른 컴포넌트들 출력
        for _, row in df.iterrows():
            if row['class'] == 'VoltageSource':
                continue  # 이미 위에서 처리함
            elif row['class'] == 'Resistor':
                f.write(f"{row['name']} {row['node1_n']} {row['node2_n']} {row['value']}\n")
            elif row['class'] == 'Capacitor':
                f.write(f"{row['name']} {row['node1_n']} {row['node2_n']} {row['value']}F\n")
            elif row['class'] == 'Diode':
                f.write(f"{row['name']} {row['node1_n']} {row['node2_n']} DMOD\n")
            elif row['class'] == 'LED':
                f.write(f"{row['name']} {row['node1_n']} {row['node2_n']} LEDMOD\n")
        
        f.write("* \n")
        f.write(".MODEL DMOD D\n")
        f.write(".MODEL LEDMOD D(IS=1E-12 N=2)\n")
        f.write(".END\n")
    
    print(f"✅ 다중 전원 SPICE 파일 저장: {output_file}")


# 기존 toSPICE 함수를 위한 래퍼 (호환성 유지)
def toSPICE(df, voltage, output_file):
    """기존 인터페이스 호환성을 위한 래퍼"""
    # 단일 전원으로 가정하고 power_pairs 생성
    power_pairs = [(1, 0, 0, 0)]  # 기본값
    toSPICE_multi_power(df, power_pairs, voltage, output_file)

def draw_connectivity_graph_from_nx_with_issues(G, connectivity_report, output_path=None):
    """
    연결성 문제를 시각적으로 표시하는 연결 그래프
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    # 기본 그래프 그리기
    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
    
    # 연결된 그룹별로 색상 지정
    group_colors = plt.cm.Set3(np.linspace(0, 1, len(connectivity_report['groups'])))
    
    node_colors = []
    node_to_group = {}
    
    # 각 노드를 해당 그룹 색상으로 매핑
    for group_idx, group in enumerate(connectivity_report['groups']):
        for comp in group:
            node_to_group[comp['name']] = group_idx
    
    # 전압원 처리
    voltage_nodes = [node for node, data in G.nodes(data=True) 
                    if data.get('comp_class') == 'VoltageSource']
    
    for node in G.nodes():
        if node in voltage_nodes:
            node_colors.append('gray')  # 전압원 색깔은 빨간색(회색으로 설정함)
        elif node in node_to_group:
            group_idx = node_to_group[node]
            node_colors.append(group_colors[group_idx])
        else:
            node_colors.append('gray')  # 미분류는 회색
    
    plt.figure(figsize=(12, 8))
    
    # 연결되지 않은 경우 제목에 경고 표시
    if not connectivity_report['is_connected']:
        title = f"🚨 연결 끊어진 회로 - {connectivity_report['num_groups']}개 그룹"
        title_color = 'red'
    else:
        title = "✅ 연결된 회로"
        title_color = 'green'
    
    plt.suptitle(title, fontsize=16, color=title_color, weight='bold')
    
    # 노드 그리기 (연결 상태에 따라 테두리 스타일 변경)
    if connectivity_report['is_connected']:
        edge_style = 'solid'
        linewidth = 1.5
    else:
        edge_style = 'dashed'
        linewidth = 2.0
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1500, alpha=0.9, 
                          edgecolors='black', linewidths=linewidth)
    
    # 엣지 그리기 (끊어진 회로는 점선)
    edge_style = '--' if not connectivity_report['is_connected'] else '-'
    nx.draw_networkx_edges(G, pos, edge_color='#7F8C8D', 
                          width=2, alpha=0.7, style=edge_style)
    
    # 라벨 그리기
    node_labels = {}
    for node, data in G.nodes(data=True):
        label = str(node)
        if 'value' in data and data['value'] != 0:
            comp_type = data.get('comp_class', '')
            if comp_type == 'Resistor':
                label += f"\n{data['value']}Ω"
            elif comp_type == 'VoltageSource':
                label += f"\n{data['value']}V"
        node_labels[node] = label
    
    nx.draw_networkx_labels(G, pos, labels=node_labels, 
                           font_size=9, font_weight='bold')
    
    # 범례 추가 (그룹별)
    legend_elements = []
    for i, group in enumerate(connectivity_report['groups']):
        group_size = len(group)
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=group_colors[i],
                      markersize=10, 
                      label=f"그룹 {i+1} ({group_size}개 컴포넌트)")
        )
    
    if voltage_nodes:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor='black',
                      markersize=10, label="전압원")
        )
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    # 하단에 문제점 표시
    if connectivity_report['issues']:
        issues_text = "문제점:\n" + "\n".join(f"• {issue}" for issue in connectivity_report['issues'])
        plt.figtext(0.02, 0.02, issues_text, fontsize=10, color='red', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    plt.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"연결성 그래프 저장: {output_path}")
    
    #plt.show()
    return plt.gcf()


def build_circuit_graph(mapped_comps):

    G = nx.Graph()

    
    # 1) 노드 추가 (nets 튜플 → 문자열)
    for comp in mapped_comps:
        n1, n2 = comp['nodes']
        nets_str = f"{n1},{n2}"

        # V+/V- 이름 변환 로직
        cls = comp['class']
        if cls == 'VoltageSource':
            if comp.get('value', 0) > 0:
                cls = 'V+'
            elif comp.get('value', 0) == 0:
                cls = 'V-'


        #nets_str = ','.join(map(str, comp['nodes']))
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