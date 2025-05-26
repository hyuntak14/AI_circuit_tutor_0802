from typing import List, Dict
import schemdraw
import schemdraw.elements as e
import networkx as nx
from spice_parser import SpiceParser


def parse_spice(spice_file: str) -> List[Dict]:
    """
    SPICE 넷리스트 파일을 파싱하여 컴포넌트 리스트를 반환합니다.
    """
    parser = SpiceParser()
    circuit = parser.parse_file(spice_file)
    return circuit.get('components', [])


def build_graph_from_spice(components: List[Dict]) -> nx.Graph:
    """
    SPICE 컴포넌트 리스트로부터 NetworkX 그래프를 생성합니다.
    노드: 회로 노드 번호
    엣지 속성: component 객체
    """
    G = nx.Graph()
    for comp in components:
        nodes = comp.get('nodes', ())
        if len(nodes) == 2:
            n1, n2 = nodes
            G.add_edge(n1, n2, component=comp)
    return G


def find_parallel_components(components: List[Dict]) -> List[List[Dict]]:
    """
    동일한 두 노드를 공유하는 컴포넌트를 병렬 그룹으로 묶습니다.
    """
    groups = []
    used = set()
    for i, c1 in enumerate(components):
        if i in used:
            continue
        group = [c1]
        used.add(i)
        for j, c2 in enumerate(components):
            if j in used:
                continue
            if set(c1['nodes']) == set(c2['nodes']):
                group.append(c2)
                used.add(j)
        groups.append(group)
    return groups


def order_series_components(G: nx.Graph, voltage_source: Dict) -> List[List[Dict]]:
    """
    전압원에서 다른 전압 단자로 가는 최단 경로상의 컴포넌트를 순서대로 정렬하고,
    경로 상에서 병렬 그룹을 감지하여 레벨로 반환합니다.
    """
    start, end = voltage_source.get('nodes', (None, None))
    if start is None or end is None:
        return []
    try:
        node_path = nx.shortest_path(G, source=start, target=end)
    except nx.NetworkXNoPath:
        return []
    levels = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        edge_data = G.get_edge_data(u, v)
        comps = []
        if isinstance(edge_data, dict):
            # 단일 Edge: 속성 dict에 'component' 키
            if 'component' in edge_data:
                comps = [edge_data['component']]
            else:
                # MultiEdge 같은 구조: key->attr dict
                for attr in edge_data.values():
                    if isinstance(attr, dict) and 'component' in attr:
                        comps.append(attr['component'])
        parallel = find_parallel_components(comps)
        levels.extend(parallel)
    return levels


def analyze_spice_topology(components: List[Dict]) -> List[List[Dict]]:
    """
    SPICE 컴포넌트의 토폴로지를 분석하여 직렬/병렬 레벨 리스트를 반환합니다.
    """
    G = build_graph_from_spice(components)
    vsources = [c for c in components if c.get('type') == 'VoltageSource']
    if not vsources:
        return [components]
    vs = vsources[0]
    return order_series_components(G, vs)


def _get_element_for_comp(comp: Dict) -> schemdraw.elements.Element:
    """
    SPICE 컴포넌트 정보를 기반으로 schemdraw 엘리먼트를 반환합니다.
    LED 등 주요 컴포넌트 지원.
    """
    ctype = comp.get('type', '')
    name = comp.get('name', '')
    value = comp.get('value', '')
    label = f"{name}\n{value}" if value not in (None, '') else name

    if ctype == 'Resistor':
        return e.Resistor().right().label(label)
    elif ctype == 'Capacitor':
        return e.Capacitor().right().label(label)
    elif ctype == 'Inductor':
        return e.Inductor2().right().label(label)
    elif ctype == 'Diode':
        return e.Diode().right().label(label)
    elif ctype == 'LED':
        return e.LED().right().label(label)
    elif ctype == 'VoltageSource':
        return e.SourceV().right().label(label)
    elif ctype == 'CurrentSource':
        return e.SourceI().right().label(label)
    else:
        return e.Line().right().label(label)


def draw_new_diagram(spice_file: str, output_path: str = None) -> schemdraw.Drawing:
    """
    SPICE 넷리스트로부터 병렬/직렬 구성이 반영된 회로도를 생성합니다.
    LED도 포함된 회로도 지원.

    Args:
        spice_file: SPICE 넷리스트 파일 경로
        output_path: 저장할 이미지 경로 (선택)

    Returns:
        schemdraw.Drawing
    """
    comps = parse_spice(spice_file)
    levels = analyze_spice_topology(comps)

    d = schemdraw.Drawing()
    for level in levels:
        if len(level) == 1:
            d += _get_element_for_comp(level[0])
        else:
            d += e.Line().right(d.unit/4).linewidth(0)
            d.push()
            spacing = d.unit
            for i, comp in enumerate(level):
                if i > 0:
                    d.pop(); d.push()
                offset = (i - (len(level)-1)/2) * spacing
                if offset:
                    d += e.Line().up(offset).linewidth(0)
                d += _get_element_for_comp(comp)
                if offset:
                    d += e.Line().down(offset).linewidth(0)
            d.pop()
            d += e.Line().right(d.unit/4).linewidth(0)
    if any(c.get('type') == 'VoltageSource' for c in comps):
        d += e.Line().down().length(d.unit/2)
        d += e.Line().left().length(d.unit * max(1, len(levels)))

    if output_path:
        d.save(output_path)
    return d

# 모듈 테스트용
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python new_diagram.py <spice_file> [<output_image>]')
    else:
        img = sys.argv[2] if len(sys.argv) >= 3 else None
        draw_new_diagram(sys.argv[1], img)
