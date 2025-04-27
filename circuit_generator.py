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
    """
    all_comps: [
      {'class': str, 'box':(x1,y1,x2,y2), 'pins':((x1,y1),(x2,y2)), 'value':float},
      ...
    ]
    holes:     [(x,y), ...]
    wires:     [ (net1, net2), ... ]  # 와이어 엔드포인트 매핑된 넷 아이디
    """

    # 1) 브레드보드 토폴로지로부터 hole→(row,net) 매핑
    hd = HoleDetector()
    row_nets = hd.get_row_nets(holes)  # [(row_idx, [ [pts...] , ...]), ...]
    hole_to_net = {}
    for row_idx, clusters in row_nets:
        for net_idx, pts in enumerate(clusters):
            for x, y in pts:
                hole_to_net[(int(round(x)), int(round(y)))] = (row_idx, net_idx)

    # 2) 소자별 핀 좌표 → 넷 매핑
    mapped = []
    for idx, comp in enumerate(all_comps, start=1):
        pin_a, pin_b = comp['pins']
        # 가장 가까운 홀 좌표 찾기
        def nearest_net(pt):
            pt_x, pt_y = pt
            # 1) 핀 좌표 유효성 검사
            if pt_x is None or pt_y is None:
                raise ValueError(f"Invalid pin coordinates: {pt}")

            # 2) hole_to_net 에서 None 이 포함된 key는 제외
            valid_holes = [h for h in hole_to_net.keys() if h[0] is not None and h[1] is not None]
            if not valid_holes:
                raise RuntimeError("No valid hole coordinates available to match against")

            # 3) 유효한 hole 중 가장 가까운 것을 선택
            nearest = min(
                valid_holes,
                key=lambda h: (h[0] - pt_x)**2 + (h[1] - pt_y)**2
            )
            return hole_to_net[nearest]
        net1 = nearest_net(pin_a)
        net2 = nearest_net(pin_b)
        name = f"{comp['class'][0].upper()}{idx}"
        mapped.append({
            'name':  name,
            'class': comp['class'],
            'value': comp['value'],
            'nodes': (net1, net2)
        })

    # 3) DataFrame 형식으로 변환 (SPICE용)
    # toSPICE가 기대하는 컬럼: name, layer, value, class
    # 여기서는 layer에 net1 개념을, node2를 implicit로 사용하도록 약식 처리
    df = pd.DataFrame([{
        'name':    m['name'],
        'layer':   m['nodes'][0][1],  # net_idx of pin A
        'value':   m['value'],
        'class':   m['class']
    } for m in mapped])

    # 4) SPICE 넷리스트 생성
    toSPICE(df, voltage, output_spice)
    print(f"[Circuit] SPICE 파일: {output_spice}")

    # 5) 회로도 이미지 생성
    # diagram.drawDiagram은 [[{'name','value'},...],...] 꼴의 리스트 기대
    circuit = []
    for layer, grp in df.groupby('layer', sort=False):
        circuit.append([
            {'name':row['name'], 'value':row['value']}
            for _, row in grp.iterrows()
        ])
    img_data = drawDiagram(voltage, circuit)
    with open(output_img, 'wb') as f:
        f.write(img_data)
    print(f"[Circuit] 회로도 이미지: {output_img}")

    # 6) 해석 결과 출력 (선택)
    R_th, I_tot, node_currents = calcCurrentAndVoltage(voltage, circuit)
    print(f"[Circuit] 등가저항: {R_th}, 전체전류: {I_tot}")
    print(f"[Circuit] 노드별 전류: {node_currents}")
