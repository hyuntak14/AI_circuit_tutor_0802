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
        path = output_img.replace('.jpg', f'_pwr{i}.jpg')
        
        # ✅ 연결 그래프: 이미 생성된 Graph G를 직접 시각화
        try:
            from diagram import draw_connectivity_graph_from_nx
            draw_connectivity_graph_from_nx(G, output_path=path.replace('.jpg', '_graph.png'))
        except Exception as e:
            print(f"Failed to generate connectivity graph: {e}")
        
        # ✅ 회로도: networkx Graph로부터 깔끔한 schemdraw 회로도 생성
        try:
            from diagram import drawDiagramFromGraph_fixed
            d = drawDiagramFromGraph_fixed(G, voltage)
            
            if d:
                # schemdraw 자체 저장 기능 사용
                d.draw()
                d.save(path)
                print(f"Circuit diagram saved: {path}")
                
                # OpenCV 이미지로도 저장 (선택적)
                try:
                    from diagram import render_drawing_to_cv2
                    img_cv = render_drawing_to_cv2(d)
                    cv2.imwrite(path.replace('.jpg', '_cv.jpg'), img_cv)
                    print(f"OpenCV version saved: {path.replace('.jpg', '_cv.jpg')}")
                except Exception as cv_error:
                    print(f"Warning: Failed to save OpenCV version: {cv_error}")
            else:
                print(f"Failed to generate circuit diagram for power pair {i}")
                # 기존 방식으로 fallback
                d_fallback = drawDiagram(voltage, mapped, wires, power_plus=(net_p, x_p), power_minus=(net_m, x_m))
                d_fallback.draw()
                d_fallback.save(path)
                
        except Exception as diagram_error:
            print(f"Error generating diagram: {diagram_error}")
            # 최종 fallback - 기존 방식 사용
            try:
                d_fallback = drawDiagram(voltage, mapped, wires, power_plus=(net_p, x_p), power_minus=(net_m, x_m))
                d_fallback.draw()
                d_fallback.save(path)
                print(f"Fallback diagram saved: {path}")
            except Exception as fallback_error:
                print(f"All diagram generation methods failed: {fallback_error}")

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

    R_th, I_tot, node_currents = calcCurrentAndVoltage(voltage, circuit_levels)
    print(f"[Circuit] 등가저항: {R_th}, 전체전류: {I_tot}")
    print("=== Node Voltages/ Currents per Level ===")
    for i, c in enumerate(node_currents):
        print(f"Level {i+1}: currents = {c}")

    return mapped, hole_to_net


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