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

    Args:
        img: 클릭할 이미지 (NumPy array)
        window_name: OpenCV 창 제목
        prompts: 클릭 순서에 따라 출력할 프롬프트 리스트

    Returns:
        pts: 클릭한 좌표들의 리스트
    """
    pts = []
    clone = img.copy()

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < len(prompts):
            pts.append((x, y))
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, clone)
            if len(pts) >= len(prompts):
                cv2.destroyWindow(window_name)

    cv2.namedWindow(window_name)
    cv2.imshow(window_name, clone)
    cv2.setMouseCallback(window_name, click_event)
    for msg in prompts:
        print(msg)
    cv2.waitKey(0)
    return pts



def drawDiagram(voltage, comps, wires=None, power_plus=None, power_minus=None):
    """
    회로도를 그리는 함수
    CLI 호출 시 add_argument를 통해 인수를 넘길 수 있도록 처리합니다.
    """
    # CLI 인자 파싱: power_plus/power_minus 미전달 시 처리
    if power_plus is not None or power_minus is not None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--wires', nargs=2, type=int, action='append')
        parser.add_argument('--power-plus', nargs=2, type=float)
        parser.add_argument('--power-minus', nargs=2, type=float)
        args, _ = parser.parse_known_args()
        wires = args.wires if args.wires is not None else wires
        power_plus = tuple(args.power_plus) if args.power_plus else power_plus
        power_minus = tuple(args.power_minus) if args.power_minus else power_minus

    if wires is None:
        wires = []

    # 전원 소자 추가
    if power_plus and power_minus:
        net_p, xp = power_plus
        net_m, xm = power_minus
        comps = list(comps)
        comps.append({'name': 'V1', 'class': 'VoltageSource', 'value': voltage, 'nodes': (net_p, net_m)})

    # 넷 집합 구성
    net_set = {n for comp in comps for n in comp['nodes']}
    for u, v in wires:
        net_set.update((u, v))
    nets = sorted(net_set)

    # y 위치
    y_positions = {n: -i * 1.5 for i, n in enumerate(nets)}

    # 컴포넌트 위치 계산
    comp_positions = {}
    x_start, spacing = 1, 2
    for comp in comps:
        if comp['class'] == 'VoltageSource' and power_plus and power_minus:
            net_p, xp = power_plus
            net_m, xm = power_minus
            comp_positions[comp['name']] = (xp + xm) / 2
        else:
            n1, n2 = comp['nodes']
            idx1, idx2 = nets.index(n1), nets.index(n2)
            comp_positions[comp['name']] = x_start + spacing * ((idx1 + idx2) / 2)

    # Drawing
    d = schemdraw.Drawing()
    sym_map = {
        'Resistor': e.Resistor,
        'Capacitor': e.Capacitor,
        'Diode': e.Diode,
        'LED': e.LED,
        'IC': lambda: e.RBox(width=2, height=1),
        'VoltageSource': e.SourceV
    }
    # 버스 라인 그리기
    for n in nets:
        xs = [comp_positions[c['name']] for c in comps if n in c['nodes']]
        if power_plus and n == power_plus[0]: xs.append(power_plus[1])
        if power_minus and n == power_minus[0]: xs.append(power_minus[1])
        if len(xs) >= 2:
            d += e.Line().at((min(xs), y_positions[n])).to((max(xs), y_positions[n]))

    # 컴포넌트 추가
    for comp in comps:
        x = comp_positions[comp['name']]
        n1, n2 = comp['nodes']
        y1, y2 = y_positions[n1], y_positions[n2]
        sym_cls = sym_map.get(comp['class'], lambda: e.Dot)
        elem = sym_cls().at((x, y1)).to((x, y2))
        label = f"{comp['name']} {comp['value']}Ω" if comp['class']=='Resistor' else comp['name']
        elem.label(label)
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
