import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import schemdraw
import schemdraw.elements as e

def get_n_clicks(img, window_name, prompts):
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


def drawDiagram(voltage, comps,wires, power_plus=None, power_minus=None):
    """회로도를 그리는 함수
    
    Args:
        voltage: 공급 전압 (V)
        comps: 회로 컴포넌트 리스트, 각 항목은 'name','class','value','nodes' 키를 가진 딕셔너리
        power_plus: (net_id, x_pos) - 전원 +를 연결할 네트워크 ID와 x 위치
        power_minus: (net_id, x_pos) - 전원 -를 연결할 네트워크 ID와 x 위치
        
    Returns:
        schemdraw.Drawing: 생성된 회로도 객체
    """
    d = schemdraw.Drawing()
    # 심볼 매핑
    sym_map = {
        'Resistor': e.Resistor,
        'Capacitor': e.Capacitor,
        'Diode': e.Diode,
        'LED': e.LED,
        'IC': lambda: e.RBox(width=2, height=1),
        'VoltageSource': e.SourceV
    }
    
    # 0) 컴포넌트 위치 계획
    comp_positions = {}
    x_start = 1
    for idx, comp in enumerate(comps):
        comp_positions[comp['name']] = x_start + idx * 2
    
    # 전원 위치 설정
    power_x = 0
    if power_plus and power_plus[1] > 0:
        power_x = power_plus[1]
    elif power_minus and power_minus[1] > 0:
        power_x = power_minus[1]
    else:
        # 기본 전원 위치
        max_comp_x = max(comp_positions.values()) if comp_positions else 1
        power_x = max_comp_x + 2
    
    # 1) 필요한 노드와 범위 파악
    #nets = sorted({n for comp in comps for n in comp['nodes']})
    # — 컴포넌트 노드 + 전원 터미널 네트 포함 —
    net_set = {n for comp in comps for n in comp['nodes']}
    if power_plus:   net_set.add(power_plus[0])
    if power_minus:  net_set.add(power_minus[0])
    # (선택) 와이어 네트까지 포함하려면, drawDiagram 호출 시 wires 인자를 추가로 넘기고:
    if wires:
        for u,v in wires: net_set.update((u,v))
    nets = sorted(net_set)
    y_positions = {n: -i * 1.5 for i, n in enumerate(nets)}
    
    # 노드별 사용 범위 파악
    # 각 노드별로 실제 연결된 컴포넌트들의 x 위치를 추적
    node_connections = {n: [] for n in nets}
    
    # 컴포넌트의 위치로 각 노드의 연결 포인트 수집
    for comp in comps:
        x_pos = comp_positions[comp['name']]
        for node in comp['nodes']:
            node_connections[node].append(x_pos)
    
    # 전원 위치 고려
    if power_plus:
        net_plus = power_plus[0]
        if net_plus in node_connections:
            node_connections[net_plus].append(power_x)
    
    if power_minus:
        net_minus = power_minus[0]
        if net_minus in node_connections:
            node_connections[net_minus].append(power_x)
    
    # 2) 개선된 버스 라인 그리기 - 연결 포인트 사이에만 선 그리기
    for n, y in y_positions.items():
        # 연결 포인트 정렬
        points = sorted(node_connections[n])
        
        if len(points) > 1:  # 최소 2개 이상의 연결점이 있어야 선을 그림
            # 연속된 연결점 사이에 선 그리기
            for i in range(len(points) - 1):
                start_x = points[i]
                end_x = points[i + 1]
                # 충분히 가까운 점들은 연결하지 않음 (같은 컴포넌트의 두 연결점으로 간주)
                if end_x - start_x > 0.1:  # 최소 간격 설정
                    d += e.Line().at((start_x, y)).to((end_x, y))
                    
    
    # 3) 컴포넌트 그리기
    for comp in comps:
        x = comp_positions[comp['name']]
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
    
    # 4) 전원 연결
    if power_plus is not None and power_minus is not None:
        # 사용자 지정 위치 사용
        net_plus, x_plus = power_plus
        net_minus, x_minus = power_minus
        
        # 해당 네트워크의 y 위치 찾기
        if net_plus in y_positions:
            y_plus = y_positions[net_plus]
        else:
            print(f"Warning: Net {net_plus} not found. Using first net.")
            y_plus = y_positions[nets[0]]
            
        if net_minus in y_positions:
            y_minus = y_positions[net_minus]
        else:
            print(f"Warning: Net {net_minus} not found. Using last net.")
            y_minus = y_positions[nets[-1]]
        
        # 전원 배치 - 두 버스 라인 사이에
        d += e.SourceV().at((power_x, y_plus)).to((power_x, y_minus)).label(f"{voltage}V")

    return d

def load_circuit_from_spice(path: str) -> list[dict]:
    comps = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            # 주석(*)이나 .op/.end 같은 제어문은 건너뛰기
            if not line or line.startswith('*') or line.startswith('.'):
                continue
            tokens = line.split()
            name = tokens[0]
            # 노드 파싱
            try:
                n1 = int(tokens[1]); n2 = int(tokens[2])
            except ValueError:
                # 잘못된 노드 정보 스킵
                continue
            # 값 파싱 및 클래스 결정
            if name.upper().startswith('V'):
                # 전압원: 토큰[3]이 'DC'일 때 토큰[4]이 값
                if len(tokens) >= 5 and tokens[3].upper() == 'DC':
                    try:
                        val = float(tokens[4])
                    except ValueError:
                        continue
                else:
                    try:
                        val = float(tokens[3])
                    except ValueError:
                        continue
                cls = 'VoltageSource'
            else:
                # 일반 소자: 토큰[3]에 값이 있다고 가정
                try:
                    val = float(tokens[3])
                except (IndexError, ValueError):
                    continue
                cls = (
                    'Resistor'   if name.startswith('R') else
                    'Capacitor'  if name.startswith('C') else
                    'Diode'      if name.startswith('D') else
                    'LED'        if name.startswith('L') else
                    'IC'         if name.startswith('X') else
                    'Unknown'
                )
            comps.append({
                'name': name,
                'class': cls,
                'value': val,
                'nodes': (n1, n2)
            })
    return comps

def create_example_circuit(circuit_type='voltage_divider'):
    """예시 회로를 생성합니다.
    
    Args:
        circuit_type: 생성할 회로 타입 ('voltage_divider', 'rc_filter', 'led_circuit')
        
    Returns:
        list: 회로 컴포넌트 리스트
    """
    if circuit_type == 'voltage_divider':
        # 간단한 전압 분배기 회로
        return [
            {'name': 'R1', 'class': 'Resistor', 'value': 10000, 'nodes': (1, 2)},
            {'name': 'R2', 'class': 'Resistor', 'value': 5000, 'nodes': (2, 0)}
        ]
    elif circuit_type == 'rc_filter':
        # RC 저역통과 필터
        return [
            {'name': 'R1', 'class': 'Resistor', 'value': 4700, 'nodes': (1, 2)},
            {'name': 'C1', 'class': 'Capacitor', 'value': 0.000001, 'nodes': (2, 0)}
        ]
    elif circuit_type == 'led_circuit':
        # LED 구동 회로 - 이미지에서 보이는 것처럼 배치
        return [
            {'name': 'D1', 'class': 'LED', 'value': 2.0, 'nodes': (1, 2)},
            {'name': 'R1', 'class': 'Resistor', 'value': 220, 'nodes': (2, 0)}
        ]
    elif circuit_type == 'complex':
        # 좀 더 복잡한 회로
        return [
            {'name': 'R1', 'class': 'Resistor', 'value': 1000, 'nodes': (1, 2)},
            {'name': 'R2', 'class': 'Resistor', 'value': 4700, 'nodes': (2, 3)},
            {'name': 'C1', 'class': 'Capacitor', 'value': 0.0000047, 'nodes': (2, 0)},
            {'name': 'D1', 'class': 'LED', 'value': 2.0, 'nodes': (3, 0)}
        ]
    else:
        # 기본 회로
        return [
            {'name': 'R1', 'class': 'Resistor', 'value': 1000, 'nodes': (1, 0)}
        ]

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--power-plus', type=int, nargs=2, help='net_id and x position for power + (e.g. 1 8)')
    parser.add_argument('--power-minus', type=int, nargs=2, help='net_id and x position for power - (e.g. 0 8)')
    parser.add_argument('--spice-file', type=str, help='SPICE netlist file (optional)')
    parser.add_argument('--voltage', type=float, default=5, help='Supply voltage in volts')
    parser.add_argument('--output', type=str, default='diagram_output.jpg', help='Output filename')
    parser.add_argument('--example', type=str, default='complex', 
                      choices=['voltage_divider', 'rc_filter', 'led_circuit', 'complex'],
                      help='Example circuit type to generate if no SPICE file is provided')
    args = parser.parse_args()

    # 전압 설정
    voltage = args.voltage
    
    # 회로 로드 (SPICE 파일이 있으면 파일에서, 없으면 예제 생성)
    if args.spice_file and os.path.exists(args.spice_file):
        print(f"Loading circuit from {args.spice_file}")
        circuit = load_circuit_from_spice(args.spice_file)
    else:
        print(f"Creating example {args.example} circuit")
        circuit = create_example_circuit(args.example)
    
    # 컴포넌트 정보 출력
    print(f"Circuit components: {len(circuit)}")
    for comp in circuit:
        print(f"  {comp['name']} ({comp['class']}): {comp['value']} Ω/F/V, nodes: {comp['nodes']}")
    
    # 전원 위치 설정
    if args.power_plus and args.power_minus:
        p_plus = tuple(args.power_plus)
        p_minus = tuple(args.power_minus)
    else:
        # 예제별 기본 전원 위치 설정
        if args.example == 'led_circuit':
            # LED 회로의 경우 왼쪽에 전원 배치
            width = len(circuit) * 2 + 2
            p_plus = (1, 0)  # 네트워크 1, x 위치 0
            p_minus = (0, 0)  # 네트워크 0, x 위치 0
        else:
            # 기본 위치: 오른쪽 끝
            width = len(circuit) * 2 + 2
            p_plus = (1, width)  # 네트워크 1, 오른쪽 끝
            p_minus = (0, width)  # 네트워크 0, 오른쪽 끝

    # 회로도 그리기
    drawing = drawDiagram(voltage, circuit, wires=[], power_plus=p_plus, power_minus=p_minus)
    drawing.draw()
    drawing.save(args.output)
    print(f"Circuit diagram saved to {args.output}")