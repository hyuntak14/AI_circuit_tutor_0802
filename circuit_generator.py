# circuit_generator.py (전체 수정)
import pandas as pd
import numpy as np
from detector.hole_detector import HoleDetector
from diagram import drawDiagram
from writeSPICE import toSPICE
from calcVoltageAndCurrent import calcCurrentAndVoltage


def generate_circuit(
    img_path: str,
    all_comps: list,
    holes: list,
    wires: list,
    voltage: float,
    output_spice: str,
    output_img: str
):
    # 1) hole → (row, net) 매핑
    hd = HoleDetector()
    row_nets = hd.get_row_nets(holes)
    hole_to_net = {}
    for row_idx, clusters in row_nets:
        for net_idx, pts in enumerate(clusters):
            for x, y in pts:
                hole_to_net[(int(round(x)), int(round(y)))] = (row_idx, net_idx)

    # 2) wires 기반 넷 병합 (Union-Find)
    parent = {(r, n): (r, n) for r, clusters in row_nets for n in range(len(clusters))}
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
        'node1_r': m['nodes'][0][0],
        'node1_n': m['nodes'][0][1],
        'node2_r': m['nodes'][1][0],
        'node2_n': m['nodes'][1][1]
    } for m in mapped])

    # 5) SPICE 넷리스트 생성
    toSPICE(df, voltage, output_spice)

    # 6) 회로도 이미지 생성 (flat list 사용)
    img_data = drawDiagram(voltage, mapped)
    with open(output_img, 'wb') as f:
        f.write(img_data)

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