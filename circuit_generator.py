# circuit_generator.py
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

    # 2) 소자별 핀 → 네트 매핑
    mapped = []
    for idx, comp in enumerate(all_comps, start=1):
        pin_a, pin_b = comp['pins']
        def nearest_net(pt):
            x, y = pt
            if x is None or y is None:
                raise ValueError(f"Invalid pin coordinates: {pt}")
            valid = [h for h in hole_to_net if None not in h]
            if not valid:
                raise RuntimeError("No valid hole coordinates")
            closest = min(valid, key=lambda h: (h[0]-x)**2 + (h[1]-y)**2)
            return hole_to_net[closest]

        net1 = nearest_net(pin_a)
        net2 = nearest_net(pin_b)

        # SPICE 이름 접두어
        prefix = {
            'Resistor':  'R',
            'Diode':     'D',
            'LED':       'L',
            'Capacitor': 'C',
            'IC':        'U'
        }.get(comp['class'], 'X')
        name = f"{prefix}{idx}"

        mapped.append({
            'name':  name,
            'class': comp['class'],
            'value': comp['value'],
            'nodes': (net1, net2)
        })

    # 3) DataFrame 생성
    df = pd.DataFrame([{
        'name':  m['name'],
        'class': m['class'],
        'value': m['value'],
        'layer': m['nodes'][0][1]
    } for m in mapped])

    # 4) SPICE 넷리스트
    toSPICE(df, voltage, output_spice)
    print(f"[Circuit] SPICE 파일: {output_spice}")

    # 5) 회로도 이미지 생성
    circuit = []
    for layer, grp in df.groupby('layer', sort=False):
        circuit.append([
            {
                'name':  row['name'],
                'value': row['value'],
                'class': row['class']   # ← 이 줄을 추가
            }
            for _, row in grp.iterrows()
        ])
    img_data = drawDiagram(voltage, circuit)
    with open(output_img, 'wb') as f:
        f.write(img_data)
    print(f"[Circuit] 회로도 이미지: {output_img}")

    # 6) 해석 (선택)
    R_th, I_tot, node_currents = calcCurrentAndVoltage(voltage, circuit)
    print(f"[Circuit] 등가저항: {R_th}, 전체전류: {I_tot}")
    print(f"[Circuit] 노드별 전류: {node_currents}")
