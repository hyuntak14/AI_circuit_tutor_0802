import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import schemdraw
import schemdraw.elements as e


def drawDiagram(V, comps: list):
    """
    V: 공급 전압 (V)
    comps: flat list of dicts, each with 'name','class','value','nodes' (tuple of two node IDs)
    Optimal schematic: horizontal bus lines for each net, components between buses
    """
    d = schemdraw.Drawing()
    # 심볼 매핑
    sym_map = {
        'Resistor': e.Resistor,
        'Capacitor': e.Capacitor,
        'Diode': e.Diode,
        'LED': e.LED,
        'IC': lambda: e.RBox(width=2, height=1)
    }
    # 1) 노드 정리 및 Y 위치 지정
    nets = sorted({n for comp in comps for n in comp['nodes']})
    y_positions = {n: -i * 1.5 for i, n in enumerate(nets)}
    # 2) 버스 라인 그리기
    width = len(comps) * 2 + 2
    for n, y in y_positions.items():
        d += e.Line().at((0, y)).to((width, y))
    # 3) 컴포넌트 그리기
    for idx, comp in enumerate(comps):
        x = 1 + idx * 2
        n1, n2 = comp['nodes']
        y1, y2 = y_positions[n1], y_positions[n2]
        # 심볼 생성
        sym_cls = sym_map.get(comp['class'], lambda: e.Dot)
        symbol = sym_cls()
        # 레이블 설정
        if comp['class'] == 'Resistor':
            label = [comp['name'], f"{comp['value']}Ω"]
        else:
            label = [comp['name']]
        # 심볼 배치: y1에서 y2 연결
        elem = symbol.at((x, y1)).to((x, y2)).label(label)
        d += elem
        # 터미널에서 버스로 연결 (spur)
        d += e.Line().at(elem.start).to((elem.start[0], y1))
        d += e.Line().at(elem.end).to((elem.end[0], y2))
    # 4) 전원 연결 (net 0 기준)
    if 0 in y_positions:
        y0 = y_positions[0]
    else:
        y0 = y_positions[nets[0]]
    d += e.SourceV().at((width, y0)).label(f"{V}V").reverse()
    # 5) 이미지 반환
    imgdata = d.get_imagedata('jpg')
    plt.clf(); plt.close('all')
    return imgdata

if __name__ == "__main__":
    # 테스트: R1(1-2), C1(2-89), LED1(90-0)
    V = 5.0
    comps = [
        {"name": "R1",  "class": "Resistor",  "value": 100,   "nodes": (1, 2)},
        {"name": "C1",  "class": "Capacitor", "value": 0.001, "nodes": (2, 89)},
        {"name": "LED1","class": "LED",       "value": 0,     "nodes": (90, 0)}
    ]
    img = drawDiagram(V, comps)
    out_path = "test_diagram_natural.jpg"
    with open(out_path, "wb") as f:
        f.write(img)
    print(f"Natural schematic saved to {out_path}")
