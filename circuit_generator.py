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
matplotlib.use('TkAgg')  # 또는 'Qt5Agg', 'WxAgg' 등 다른 대화형 백엔드
# 이후 schemdraw 코드 실행
import cv2

def generate_circuit(
    all_comps: list,
    holes: list,
    wires: list,
    voltage: float,
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

     # 🔧 4) 전원 소스 추가 (이 부분이 누락되어 있었음!)
    print("=== Adding Power Sources ===")
    for i, (net_p, x_p, net_m, x_m) in enumerate(power_pairs, start=1):
        vs_name = f"V{i}"
        vs_comp = {
            'name': vs_name,
            'class': 'VoltageSource',
            'value': voltage,
            'nodes': (net_p, net_m)
        }
        mapped.append(vs_comp)
        print(f"{vs_name} (VoltageSource): Net1={net_p}, Net2={net_m}, Value={voltage}V")



    # 4) DataFrame 구성
    df = pd.DataFrame([{
        'name': m['name'],
        'class': m['class'],
        'value': m['value'],
        'node1_n': m['nodes'][0],
        'node2_n': m['nodes'][1],
    } for m in mapped])

    # 5) 그래프 저장
    G = build_circuit_graph(mapped)
    save_circuit_graph(G, output_img.replace('.jpg', '.graphml'))
    write_graphml(G, output_img.replace('.jpg', '.graphml'))

    # 6) 비교
    # 상세 점수 출력
    # 6) 비교 - 개선된 버전
    try:
        import glob
        graphml_dir = "checker"
        files = glob.glob(os.path.join(graphml_dir, "*.graphml"))
        if files:
            print(f"\n[회로 비교] {len(files)}개의 기준 회로와 비교 중...")
            
            sims = []
            best_match_details = None
            best_match_score = -1
            
            for f in files:
                try:
                    G2 = nx.read_graphml(f)
                    
                    # 디버그 모드로 상세 비교 수행
                    comparer = CircuitComparer(G, G2, debug=False)  # 전체적으로는 debug=False
                    similarity = comparer.compute_similarity()
                    details = comparer.detailed_comparison()
                    
                    sims.append((os.path.basename(f), similarity, details))
                    
                    # 가장 높은 점수의 회로에 대해서는 상세 정보 저장
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_details = (os.path.basename(f), comparer, details)
                    
                except Exception as e:
                    print(f"[비교 실패] {f}: {e}")
            
            # 유사도 순으로 정렬
            sims.sort(key=lambda x: -x[1])
            
            print("\n=== 유사도 TOP 3 회로 ===")
            for i, (filename, score, details) in enumerate(sims[:3]):
                print(f"{i+1}. {filename}")
                print(f"   💯 전체 유사도: {score:.3f}")
                print(f"   🔧 컴포넌트 매칭: {details['node_score']:.3f}")
                print(f"   🔗 연결 매칭: {details['edge_score']:.3f}")
                print(f"   📊 노드 수: {details['graph1_nodes']} vs {details['graph2_nodes']}")
                print(f"   📊 엣지 수: {details['graph1_edges']} vs {details['graph2_edges']}")
                print()
            
            # 가장 유사한 회로에 대한 상세 분석
            if best_match_details and best_match_score > 0.7:
                filename, comparer, details = best_match_details
                print(f"\n=== 최고 유사도 회로 상세 분석: {filename} ===")
                print(f"🎯 유사도: {best_match_score:.3f}")
                
                # 디버그 모드로 다시 비교하여 상세 정보 출력
                comparer_debug = CircuitComparer(G, comparer.G2, debug=True)
                comparer_debug.compute_similarity()
                
                # 🎨 그래프 시각화 추가
                try:
                    vis_path = output_img.replace('.jpg', '_comparison.png')
                    comparer_debug.visualize_comparison(save_path=vis_path, show=False)
                    print(f"📊 회로 비교 시각화 저장됨: {vis_path}")
                except Exception as viz_e:
                    print(f"[시각화 오류] {viz_e}")
                    
            elif best_match_score > 0:
                filename, comparer, details = best_match_details
                print(f"\n💡 가장 유사한 회로: {filename} (유사도: {best_match_score:.3f})")
                print("   - 유사도가 낮습니다. 새로운 형태의 회로일 가능성이 높습니다.")
                
                # 🎨 낮은 유사도라도 시각화 제공
                try:
                    vis_path = output_img.replace('.jpg', '_comparison.png')
                    comparer_debug = CircuitComparer(G, comparer.G2, debug=False)
                    comparer_debug.visualize_comparison(save_path=vis_path, show=False)
                    print(f"📊 회로 비교 시각화 저장됨: {vis_path}")
                except Exception as viz_e:
                    print(f"[시각화 오류] {viz_e}")
            
            # 전체 통계
            if sims:
                avg_similarity = sum(sim[1] for sim in sims) / len(sims)
                print(f"\n📈 전체 통계")
                print(f"   평균 유사도: {avg_similarity:.3f}")
                print(f"   최고 유사도: {max(sim[1] for sim in sims):.3f}")
                print(f"   최저 유사도: {min(sim[1] for sim in sims):.3f}")
                
                # 유사한 회로 개수
                similar_count = sum(1 for sim in sims if sim[1] > 0.8)
                print(f"   매우 유사한 회로 (0.8+): {similar_count}개")
                
        else:
            print("[비교] 비교 대상 .graphml 파일이 checker 폴더에 없습니다.")
            print("       CircuitSaver로 기준 회로를 먼저 생성해주세요.")
            
    except ImportError:
        print("[오류] networkx 또는 glob 모듈을 가져올 수 없습니다.")
    except Exception as e:
        print(f"[오류] 회로 비교 실패: {e}")
        import traceback
        traceback.print_exc()


    #기존 비교 (최종 점수만 출력)
    '''try:
        import glob
        graphml_dir = "checker"
        files = glob.glob(os.path.join(graphml_dir, "*.graphml"))
        if files:
            
            sims = []
            for f in files:
                try:
                    G2 = nx.read_graphml(f)
                    sim = CircuitComparer(G, G2).compute_similarity()
                    sims.append((os.path.basename(f), sim))
                except Exception as e:
                    print(f"[비교 실패] {f}: {e}")
            sims.sort(key=lambda x: -x[1])
            print("\n[유사도 TOP 3 회로]")
            for i, (f, score) in enumerate(sims[:3]):
                print(f"{i+1}. {f} → 유사도: {score:.3f}")
        else:
            print("[비교] 비교 대상 .graphml 없음")
    except Exception as e:
        print(f"[오류] 회로 비교 실패: {e}")'''

    # 7) SPICE 저장
    toSPICE(df, voltage, output_spice)

    # 8) 전원별 회로도 및 연결 그래프 시각화
    for i, (net_p, x_p, net_m, x_m) in enumerate(power_pairs, 1):
        #path = output_img.replace('.jpg', f'_pwr{i}.jpg')
        if i == 1:
            path = output_img
        else:
            path = output_img.replace('.jpg', f'_pwr{i}.jpg')
        
        # ✅ 연결성 검증 추가
        from diagram import validate_circuit_connectivity
        
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
        
        # ✅ 연결 그래프: 연결성 정보 포함 (import 오류 해결)
        try:
            # import 오류를 피하기 위해 조건부 import 사용
            try:
                from diagram import draw_connectivity_graph_from_nx_with_issues
                draw_connectivity_graph_from_nx_with_issues(G, connectivity_report, 
                                                           output_path=path.replace('.jpg', '_graph.png'))
            except ImportError:
                # 함수가 없으면 기본 연결 그래프 함수 사용
                from diagram import draw_connectivity_graph_from_nx
                draw_connectivity_graph_from_nx(G, output_path=path.replace('.jpg', '_graph.png'))
                print(f"✅ 기본 연결성 그래프 저장: {path.replace('.jpg', '_graph.png')}")
        except Exception as e:
            print(f"Failed to generate connectivity graph: {e}")
        
        # ✅ 회로도: 연결성 확인하여 생성 (GUI 오류 해결)
        try:
            # GUI 관련 오류를 피하기 위해 try-except로 감싸기
            try:
                from diagram import drawDiagramFromGraph_with_connectivity_check
                d = drawDiagramFromGraph_with_connectivity_check(G, voltage)
            except Exception as gui_error:
                # GUI 오류가 발생하면 기본 다이어그램 생성 시도
                print(f"GUI 오류로 기본 다이어그램 생성 시도: {gui_error}")
                from diagram import drawDiagramFromGraph
                d = drawDiagramFromGraph(G, voltage)
            
            if d:
                try:
                    # 다이어그램 그리기 및 저장 (GUI 오류 방지)
                    import matplotlib
                    matplotlib.use('TkAgg')  # GUI 백엔드 사용 안함
                    d.draw()
                    d.save(path)
                    
                    # 연결성 문제가 있으면 파일명에 표시
                    if not connectivity_report['is_connected']:
                        disconnected_path = path.replace('.jpg', '_DISCONNECTED.jpg')
                        d.save(disconnected_path)
                        print(f"⚠️  연결 끊어진 회로도 저장: {disconnected_path}")
                    else:
                        print(f"✅ 정상 회로도 저장: {path}")
                    
                    # OpenCV 버전 저장 (메인 스레드 오류 방지)
                    try:
                        from diagram import render_drawing_to_cv2
                        img_cv = render_drawing_to_cv2(d)
                        import cv2
                        cv2.imwrite(path.replace('.jpg', '_cv.jpg'), img_cv)
                    except Exception as cv_error:
                        print(f"Warning: OpenCV 버전 저장 실패: {cv_error}")
                        
                except Exception as save_error:
                    print(f"다이어그램 저장 실패: {save_error}")
            else:
                print(f"❌ 회로도 생성 실패 (전원 {i})")
                
        except Exception as diagram_error:
            print(f"Error generating diagram: {diagram_error}")
            # 연결성 문제 리포트를 텍스트 파일로 저장
            report_path = path.replace('.jpg', '_connectivity_report.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("회로 연결성 분석 보고서\n")
                f.write("=" * 30 + "\n\n")
                f.write(f"연결 상태: {'연결됨' if connectivity_report['is_connected'] else '끊어짐'}\n")
                f.write(f"그룹 수: {connectivity_report['num_groups']}\n\n")
                
                if connectivity_report['issues']:
                    f.write("문제점:\n")
                    for issue in connectivity_report['issues']:
                        f.write(f"- {issue}\n")
                    f.write("\n")
                
                f.write("그룹별 컴포넌트:\n")
                for j, group in enumerate(connectivity_report['groups']):
                    comp_names = [comp['name'] for comp in group]
                    f.write(f"그룹 {j+1}: {comp_names}\n")
            
            print(f"📋 연결성 보고서 저장: {report_path}")
    # 9) 전류·전압 해석
    circuit_levels = []
    for lvl, grp in df.groupby('node1_n', sort=False):
        comps = []
        for _, row in grp.iterrows():
            comps.append({
                'name': row['name'],
                'value': int(row['value']),
                'class': row['class']
            })
        circuit_levels.append(comps)

    #R_th, I_tot, node_currents = calcCurrentAndVoltage(voltage, circuit_levels)
    #print(f"[Circuit] 등가저항: {R_th}, 전체전류: {I_tot}")
    #print("=== Node Voltages/ Currents per Level ===")
    #for i, c in enumerate(node_currents):
    #    print(f"Level {i+1}: currents = {c}")

    return mapped, hole_to_net

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
            node_colors.append('red')  # 전압원은 빨간색
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
                      markerfacecolor='red',
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
    
    plt.show()
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