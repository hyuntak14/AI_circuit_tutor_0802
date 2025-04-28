# diagram.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import schemdraw
import schemdraw.elements as e

def drawDiagram(V, circuit: list):
    """
    circuit: [
      [ {'name': str, 'value': float, 'class': str}, … ],  # 레벨별 소자
      …
    ]
    """
    d = schemdraw.Drawing()
    d.push()

    # 레벨별로 소자 심볼을 순서대로 배치
    for level in circuit:
        for comp in level:
            nm  = comp['name']
            val = comp['value']
            cls = comp['class']

            if cls == "Resistor":
                d += e.Resistor().label([nm, f"\n{val}Ω"])
            elif cls == "Capacitor":
                d += e.Capacitor().label([nm, f"\n{val}F"])
            elif cls == "Diode":
                d += e.Diode().reverse().label([nm])
            elif cls == "LED":
                d += e.LED().label([nm])
            elif cls == "IC":
                # schemdraw에 IC 심볼이 없다면 사각형으로 대체
                chip = e.RBox(width=2, height=1).label([nm])
                d += chip
            else:
                d += e.Dot().label([nm])

    # 노드 연결부 (V 소스 포함)
    d += (n1 := e.Dot())
    d += e.Line().down().at(n1.end)
    d += (n2 := e.Dot())
    d.pop()
    d += (n3 := e.Dot())
    d += e.SourceV().down().label(f"{V}V").at(n3.end).reverse()
    d += (n4 := e.Dot())
    d += e.Line().right().endpoints(n4.end, n2.end)

    imgdata = d.get_imagedata('jpg')
    plt.clf()
    plt.close('all')
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
