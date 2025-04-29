# diagram.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import schemdraw
import schemdraw.elements as e

def drawDiagram(V, comps: list):
    d = schemdraw.Drawing()
    # 노드 ID별 Y 위치 지정
    nets = sorted({n for comp in comps for n in comp['nodes']}, key=lambda x: (x[0], x[1]))
    ys = {n: -i*1.5 for i, n in enumerate(nets)}
    x = 0
    for comp in comps:
        n1, n2 = comp['nodes']
        y1, y2 = ys[n1], ys[n2]
        sym = {
            'Resistor': e.Resistor,
            'Capacitor': e.Capacitor,
            'Diode': e.Diode,
            'LED': e.LED,
            'IC': lambda: e.RBox(width=2, height=1)
        }.get(comp['class'], e.Dot)()
        elem = sym.at((x, y1)).to((x, y2)).label([comp['name'], f"{comp['value']}Ω"] if comp['class']=='Resistor' else [comp['name']])
        d += elem
        x += 2
    # 전원 연결
    d += e.SourceV().at((x, 0)).label(f"{V}V").reverse()
    imgdata = d.get_imagedata('jpg')
    plt.clf(); plt.close('all')
    return imgdata

if __name__ == "__main__":
    # 간단 테스트용 예시
    V = 5
    circuit = [
        [
            {"name":"R1","value":10,"class":"Resistor"},
            {"name":"C1","value":0.001,"class":"Capacitor"},
            {"name":"D1","value":0,"class":"Diode"},
        ],
        [
            {"name":"L1","value":0,"class":"LED"},
            {"name":"U1","value":0,"class":"IC"},
        ]
    ]
    img = drawDiagram(V, circuit)
    with open("test_diagram.jpg","wb") as f:
        f.write(img)
