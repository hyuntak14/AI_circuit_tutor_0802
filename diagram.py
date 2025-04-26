import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import schemdraw
import schemdraw.elements as e

def drawDiagram(V, circuit: list):
    # circuit: [[{'name':…, 'value':…}, …], …]
    R_name  = [[el["name"]  for el in level] for level in circuit]
    R_value = [[el["value"] for el in level] for level in circuit]

    d = schemdraw.Drawing()
    d.push()

    # 각 레벨별 컴포넌트 (병렬/직렬) 생성
    components = []
    for lvl in range(len(R_name)):
        names  = R_name[lvl]
        values = R_value[lvl]

        # 병렬: 2개
        if len(names) == 2:
            with schemdraw.Drawing(show=False) as parallelR:
                parallelR += e.Line().right(parallelR.unit/4)
                parallelR.push()

                # 위쪽
                parallelR += e.Line().up(parallelR.unit/2)
                parallelR += (R1 := e.Resistor().right()
                              .label([names[0], f'\n{values[0]}Ω']))
                parallelR += e.Line().down(parallelR.unit/2)
                parallelR.pop()

                # 아래쪽
                parallelR += e.Line().down(parallelR.unit/2)
                parallelR += (R2 := e.Resistor().right()
                              .label([names[1], f'\n{values[1]}Ω']))
                parallelR += e.Line().up(parallelR.unit/2)

                parallelR += e.Line().right(parallelR.unit/4)

            components.append(parallelR)

        # 병렬: 3개
        elif len(names) == 3:
            with schemdraw.Drawing(show=False) as parallelR:
                parallelR += e.Line().right(parallelR.unit/4)
                parallelR.push()

                # 첫째
                parallelR += e.Line().up(parallelR.unit/2)
                parallelR += (R1 := e.Resistor().right()
                              .label([names[0], f'\n{values[0]}Ω']))
                parallelR += e.Line().down(parallelR.unit/2)
                parallelR.pop()

                # 둘째
                parallelR.push()
                parallelR += e.Line().down(parallelR.unit/2)
                parallelR += (R2 := e.Resistor().right()
                              .label([names[2], f'\n{values[2]}Ω']))
                parallelR += e.Line().up(parallelR.unit/2)
                parallelR.pop()

                # 셋째 (중간)
                parallelR += (R3 := e.Resistor()
                              .label([names[1], f'\n{values[1]}Ω']))

                parallelR += e.Line().right(parallelR.unit/4)

            components.append(parallelR)

        # 단일 저항
        elif len(names) == 1:
            with schemdraw.Drawing(show=False) as single:
                single += e.Resistor().label([names[0], f'\n{values[0]}Ω'])
            components.append(single)

    # Drawing 에 모두 추가
    for comp in components:
        d += e.ElementDrawing(comp)

    # 노드 연결 그리기
    d += (n1 := e.Dot())
    d += e.Line().down().at(n1.end)
    d += (n2 := e.Dot())
    d.pop()
    d += (n3 := e.Dot())
    d += e.SourceV().down().label(f"{V}V").at(n3.end).reverse()
    d += (n4 := e.Dot())
    d += e.Line().right().endpoints(n4.end, n2.end)

    # 이미지 데이터로 리턴
    imgdata = d.get_imagedata('jpg')
    plt.clf()
    plt.close('all')
    return imgdata


if __name__ == "__main__":
    V = 5
    circuit = [
        [	      
            {"name": "R10", "value": 3},
            {"name": "R11", "value": 3},
            {"name": "R11", "value": 3},
        ],
        [     
            {"name": "R21", "value": 2},
            {"name": "R21", "value": 2},
        ], 
        [
            {"name": "R30", "value": 6},
            {"name": "R31", "value": 6},
            {"name": "R31", "value": 6},
        ],
    ]

    drawDiagram(V, circuit)
