import matplotlib
# Use Qt5Agg for interactive GUI
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import cv2
import schemdraw
import schemdraw.elements as e
import networkx as nx
import argparse

def get_n_clicks(img, window_name, prompts):
    """
    다중 클릭으로 사용자 입력 좌표를 수집합니다.
    """
    pts = []
    clone = img.copy()
    # 첫 번째 프롬프트를 이미지에 표시
    if prompts:
        cv2.putText(clone, prompts[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    def click_event(event, x, y, flags, param):
        nonlocal clone, pts
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < len(prompts):
            pts.append((x, y))
            # 원본 이미지를 복원하고 클릭한 점 모두 그리기
            clone = img.copy()
            for px, py in pts:
                cv2.circle(clone, (px, py), 5, (0, 0, 255), -1)
            # 다음 프롬프트를 이미지에 표시
            if len(pts) < len(prompts):
                next_msg = prompts[len(pts)]
                cv2.putText(clone, next_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow(window_name, clone)
            # 모든 프롬프트 처리 시 잠시 대기 후 창 닫기
            if len(pts) >= len(prompts):
                cv2.waitKey(500)
                cv2.destroyWindow(window_name)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    cv2.imshow(window_name, clone)
    # 콘솔에도 프롬프트 출력
    for msg in prompts:
        print(msg)
    cv2.waitKey(0)
    return pts

import matplotlib
# Use Qt5Agg for interactive GUI
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import cv2
import schemdraw
import schemdraw.elements as e
import networkx as nx


def get_n_clicks(img, window_name, prompts):
    """
    다중 클릭으로 사용자 입력 좌표를 수집합니다.
    """
    pts = []
    clone = img.copy()
    # 첫 번째 프롬프트를 이미지에 표시
    if prompts:
        cv2.putText(clone, prompts[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    def click_event(event, x, y, flags, param):
        nonlocal clone, pts
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < len(prompts):
            pts.append((x, y))
            clone = img.copy()
            for px, py in pts:
                cv2.circle(clone, (px, py), 5, (0, 0, 255), -1)
            if len(pts) < len(prompts):
                next_msg = prompts[len(pts)]
                cv2.putText(clone, next_msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow(window_name, clone)
            if len(pts) >= len(prompts):
                cv2.waitKey(500)
                cv2.destroyWindow(window_name)

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, click_event)
    cv2.imshow(window_name, clone)
    for msg in prompts:
        print(msg)
    cv2.waitKey(0)
    return pts


def draw_circuit_from_connectivity(comps, power_plus=None, power_minus=None, output_path=None):
    """
    connectivity graph을 기반으로 간단한 회로도 레이아웃을 그립니다.

    - networkx spring layout을 사용해 구성요소 위치를 자동 배치
    - edge label로 net 번호 표시
    - node 색상으로 컴포넌트 종류 구분
    """
    # connectivity graph 생성
    G = draw_connectivity_graph(comps, power_plus, power_minus)
    # spring layout 좌표 재사용
    pos = nx.get_node_attributes(G, 'pos') if 'pos' in G.graph else nx.spring_layout(G, seed=0)
    # 실제 회로도처럼 배치: x,y 좌표를 schemdraw로 변환
    d = schemdraw.Drawing()
    # 축 스케일링
    scale = 4.0
    for node, (x, y) in pos.items():
        # 노드 위치에 간단한 점 표시 (wire junction)
        d += e.Dot().at((x*scale, y*scale))
        # 이름 표시
        d += e.Label().label(node).at((x*scale, y*scale + 0.2)).fontsize(8)
    for u, v, data in G.edges(data=True):
        x1, y1 = pos[u]; x2, y2 = pos[v]
        d += e.Line().at((x1*scale, y1*scale)).to((x2*scale, y2*scale))
        # 엣지 중간에 net 번호 표시
        mx, my = (x1 + x2)/2 * scale, (y1 + y2)/2 * scale
        d += e.Label().label(data['nets']).at((mx, my)).fontsize(6)
    if output_path:
        d.save(output_path)
    return d


def drawDiagram(voltage, comps, wires=None, power_plus=None, power_minus=None):
    """
    회로도를 그리는 함수

    Args:
        voltage: 공급 전압 (V)
        comps: 컴포넌트 리스트, 각 항목 {'name','class','value','nodes'}
        wires: 와이어 연결 정보 리스트 [(net1, net2), ...]
        power_plus: (net_id, x_pos) for + terminal
        power_minus: (net_id, x_pos) for - terminal

    Returns:
        schemdraw.Drawing 객체
    """
    if wires is None:
        wires = []

    # 전원 소자 추가
    vs_names = []
    if power_plus is not None and power_minus is not None:
        net_p, xp = power_plus
        net_m, xm = power_minus
        comps = list(comps)
        vs = {'name': 'V1', 'class': 'VoltageSource', 'value': voltage, 'nodes': (net_p, net_m)}
        comps.append(vs)
        vs_names.append(vs['name'])

    # 넷 집합 구성
    net_set = {n for comp in comps for n in comp['nodes']}
    for u, v in wires:
        net_set.update((u, v))
    nets = sorted(net_set)

    # y 위치 계산
    y_positions = {n: -i * 1.5 for i, n in enumerate(nets)}

    # 컴포넌트 위치 계산
    comp_positions = {}
    x_start, spacing = 3.0, 5.0
    for comp in comps:
        if comp['class'] == 'VoltageSource' and power_plus is not None and power_minus is not None:
            comp_positions[comp['name']] = (power_plus[1] + power_minus[1]) / 2.0
        else:
            n1, n2 = comp['nodes']
            idx1, idx2 = nets.index(n1), nets.index(n2)
            comp_positions[comp['name']] = x_start + spacing * ((idx1 + idx2) / 2.0)

    # Dynamic Y-offset for voltage sources based on net span
    y_vals = list(y_positions.values())
    span = max(y_vals) - min(y_vals) if len(y_vals) > 1 else spacing
    y_offset = span * 0.3  # 20% of vertical span

    # Drawing 시작
    d = schemdraw.Drawing()
    sym_map = {
        'Resistor': e.Resistor,
        'Capacitor': e.Capacitor,
        'Diode': e.Diode,
        'LED': e.LED,
        'IC': lambda: e.RBox(width=2, height=1),
        'VoltageSource': e.SourceV
    }

    # 버스 라인 그리기 (컴포넌트 및 전원 단자 포함)
    for n in nets:
        xs = [comp_positions[c['name']] for c in comps if n in c['nodes']]
        if power_plus is not None and n == power_plus[0]: xs.append(power_plus[1])
        if power_minus is not None and n == power_minus[0]: xs.append(power_minus[1])
        if len(xs) >= 2:
            d += e.Line().at((min(xs), y_positions[n])).to((max(xs), y_positions[n]))

    # 전원 연결 리드 그리기
    for comp in comps:
        if comp['name'] in vs_names:
            x = comp_positions[comp['name']]
            n1, n2 = comp['nodes']
            yb1, yb2 = y_positions[n1], y_positions[n2]
            yvs1, yvs2 = yb1 - y_offset, yb2 - y_offset
            d += e.Line().at((x, yb1)).to((x, yvs1))
            d += e.Line().at((x, yb2)).to((x, yvs2))

    # 컴포넌트 심볼 추가
    for comp in comps:
        x = comp_positions[comp['name']]
        n1, n2 = comp['nodes']
        if comp['class'] == 'VoltageSource' and comp['name'] in vs_names:
            y1, y2 = y_positions[n1] - y_offset, y_positions[n2] - y_offset
        else:
            y1, y2 = y_positions[n1], y_positions[n2]
        elem = sym_map.get(comp['class'], lambda: e.Dot)().at((x, y1)).to((x, y2))
        if comp['class'] == 'Resistor':
            elem.label(f"{comp['name']} {comp['value']}Ω")
        else:
            elem.label(comp['name'])
        d += elem

    return d



def draw_connectivity_graph(comps, power_plus=None, power_minus=None, output_path=None):
    """전원 및 컴포넌트를 노드로, 공통 Net을 엣지로 그리는 그래프"""
    G = nx.Graph()
    for comp in comps:
        G.add_node(comp['name'], type=comp['class'])
    for i, c1 in enumerate(comps):
        for c2 in comps[i+1:]:
            shared = set(c1['nodes']) & set(c2['nodes'])
            if shared:
                G.add_edge(c1['name'], c2['name'], nets=','.join(map(str, shared)))
    if power_plus:
        net_p, _ = power_plus
        G.add_node('V+', type='Power+')
        for comp in comps:
            if net_p in comp['nodes']:
                G.add_edge('V+', comp['name'], nets=str(net_p))
    if power_minus:
        net_m, _ = power_minus
        G.add_node('V-', type='Power-')
        for comp in comps:
            if net_m in comp['nodes']:
                G.add_edge('V-', comp['name'], nets=str(net_m))
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6,6))
    colors = ['lightgreen' if G.nodes[n]['type'].startswith('Power') else 'lightblue' for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=800)
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edge_labels(G, pos, nx.get_edge_attributes(G, 'nets'), font_color='red')
    plt.axis('off')
    plt.tight_layout()
    if output_path: plt.savefig(output_path, dpi=200)
    plt.show()
    return G


def load_circuit_from_spice(path: str) -> list[dict]:
    """SPICE 파일에서 컴포넌트 리스트를 읽어옵니다"""
    comps = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('*') or s.startswith('.'): continue
            tokens = s.split()
            name = tokens[0]
            try:
                n1, n2 = int(tokens[1]), int(tokens[2])
            except ValueError:
                continue
            if name.upper().startswith('V'):
                cls = 'VoltageSource'
                val = float(tokens[4]) if tokens[3].upper()=='DC' else float(tokens[3])
            else:
                val = float(tokens[3])
                cls = ('Resistor' if name.startswith('R') else
                       'Capacitor' if name.startswith('C') else
                       'Diode' if name.startswith('D') else
                       'LED' if name.startswith('L') else
                       'IC' if name.startswith('X') else 'Unknown')
            comps.append({'name':name,'class':cls,'value':val,'nodes':(n1,n2)})
    return comps


def draw_from_spice(spice_path, wires=None, power_plus=None, power_minus=None, output_path=None):
    """
    SPICE 파일을 불러와서 회로도를 그립니다
    """
    comps = load_circuit_from_spice(spice_path)
    d = drawDiagram(None, comps, wires, power_plus, power_minus)
    if output_path:
        d.draw()
        d.save(output_path)
    return d


def create_example_circuit(circuit_type='voltage_divider') -> list[dict]:
    """예시 회로를 생성합니다."""
    if circuit_type=='voltage_divider':
        return [
            {'name':'R1','class':'Resistor','value':10000,'nodes':(1,2)},
            {'name':'R2','class':'Resistor','value':5000,'nodes':(2,0)}
        ]
    elif circuit_type=='rc_filter':
        return [
            {'name':'R1','class':'Resistor','value':4700,'nodes':(1,2)},
            {'name':'C1','class':'Capacitor','value':1e-6,'nodes':(2,0)}
        ]
    elif circuit_type=='led_circuit':
        return [
            {'name':'D1','class':'LED','value':2.0,'nodes':(1,2)},
            {'name':'R1','class':'Resistor','value':220,'nodes':(2,0)}
        ]
    else:
        return [{'name':'R1','class':'Resistor','value':1000,'nodes':(1,0)}]

if __name__=='__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--power-plus', nargs=2, type=int, help='net x')
    parser.add_argument('--power-minus', nargs=2, type=int, help='net x')
    parser.add_argument('--voltage', type=float, default=5)
    parser.add_argument('--output', type=str, default='diagram.jpg')
    args = parser.parse_args()
    # 예시 사용
    circuit = create_example_circuit()
    p_plus = tuple(args.power_plus) if args.power_plus else (1,0)
    p_minus= tuple(args.power_minus) if args.power_minus else(0,0)
    d = drawDiagram(args.voltage,circuit,[],power_plus=p_plus,power_minus=p_minus)
    d.draw(); d.save(args.output)
    print(f"Saved: {args.output}")
