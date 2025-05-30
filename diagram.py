# diagram.py 파일 상단에 추가할 코드

import matplotlib
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg', 'WxAgg' 등 다른 대화형 백엔드
# 이후 schemdraw 코드 실행
# Streamlit 환경에서 GUI 오류를 방지하기 위해 Agg 백엔드 사용
import os
from typing import List, Dict

#if 'STREAMLIT_SERVER_PORT' in os.environ:
#    matplotlib.use('Agg')
#else:
#    matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import cv2
import schemdraw
import schemdraw.elements as e
import networkx as nx
import argparse
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
from collections import defaultdict

# diagram.py에 추가할 함수들

def drawDiagramFromSpice(spice_file: str, output_path: str = None) -> schemdraw.Drawing:
    """
    SPICE 넷리스트 파일로부터 회로도를 생성합니다.
    
    Args:
        spice_file: SPICE 넷리스트 파일 경로
        output_path: 출력 이미지 파일 경로 (선택사항)
    
    Returns:
        schemdraw.Drawing 객체
    """
    from spice_parser import SpiceParser
    
    # SPICE 파일 파싱
    parser = SpiceParser()
    circuit_data = parser.parse_file(spice_file)
    
    # 컴포넌트를 networkx 그래프로 변환
    G = build_graph_from_spice(circuit_data['components'])
    
    # 회로 토폴로지 분석
    circuit_levels = analyze_spice_topology(G, circuit_data['components'])
    
    # 회로도 그리기
    d = drawDiagramFromSpiceComponents(circuit_data['components'], circuit_levels)
    
    if output_path:
        d.save(output_path)
    
    return d


def build_graph_from_spice(components: List[Dict]) -> nx.Graph:
    """
    SPICE 컴포넌트 리스트로부터 networkx 그래프를 생성합니다.
    """
    G = nx.Graph()
    
    # 노드(넷) 추가
    all_nodes = set()
    for comp in components:
        if isinstance(comp['nodes'], tuple):
            for node in comp['nodes']:
                all_nodes.add(node)
    
    G.add_nodes_from(all_nodes)
    
    # 컴포넌트를 엣지로 추가
    for comp in components:
        if comp['type'] in ['Resistor', 'Capacitor', 'Inductor', 'VoltageSource', 
                           'CurrentSource', 'Diode']:
            n1, n2 = comp['nodes']
            G.add_edge(n1, n2, component=comp)
    
    return G


def analyze_spice_topology(G: nx.Graph, components: List[Dict]) -> List[List[Dict]]:
    """
    SPICE 컴포넌트의 토폴로지를 분석하여 직렬/병렬 구조를 파악합니다.
    """
    # 그라운드 노드 (0) 찾기
    ground_node = 0
    
    # 전압원 찾기
    voltage_sources = [c for c in components if c['type'] == 'VoltageSource']
    
    if not voltage_sources:
        # 전압원이 없으면 모든 컴포넌트를 하나의 레벨로
        return [components]
    
    # 병렬 그룹 찾기
    parallel_groups = find_parallel_components_spice(components)
    
    # 직렬 순서 결정
    circuit_levels = order_series_components_spice(G, parallel_groups, voltage_sources[0])
    
    return circuit_levels


def find_parallel_components_spice(components: List[Dict]) -> List[List[Dict]]:
    """
    같은 두 노드에 연결된 컴포넌트들을 병렬 그룹으로 묶습니다.
    """
    groups = []
    processed = set()
    
    for i, comp1 in enumerate(components):
        if i in processed or comp1['type'] == 'VoltageSource':
            continue
        
        group = [comp1]
        processed.add(i)
        
        # 같은 노드에 연결된 다른 컴포넌트 찾기
        for j, comp2 in enumerate(components):
            if j in processed or j == i or comp2['type'] == 'VoltageSource':
                continue
            
            # 두 컴포넌트가 같은 두 노드에 연결되어 있는지 확인
            if set(comp1['nodes'][:2]) == set(comp2['nodes'][:2]):
                group.append(comp2)
                processed.add(j)
        
        groups.append(group)
    
    return groups


def order_series_components_spice(G: nx.Graph, parallel_groups: List[List[Dict]], 
                                  voltage_source: Dict) -> List[List[Dict]]:
    """
    전압원으로부터의 전류 경로를 따라 컴포넌트 그룹을 정렬합니다.
    """
    # 전압원의 양극에서 시작
    start_node = voltage_source['nodes'][0]
    end_node = voltage_source['nodes'][1]
    
    # 각 병렬 그룹의 대표 노드 쌍 추출
    group_nodes = []
    for group in parallel_groups:
        if group:
            nodes = group[0]['nodes'][:2]
            group_nodes.append((group, set(nodes)))
    
    # 노드 연결 순서에 따라 정렬
    ordered_groups = []
    current_node = start_node
    used_groups = set()
    
    while len(ordered_groups) < len(parallel_groups):
        for i, (group, nodes) in enumerate(group_nodes):
            if i in used_groups:
                continue
            
            if current_node in nodes:
                ordered_groups.append(group)
                used_groups.add(i)
                # 다음 노드로 이동
                for node in nodes:
                    if node != current_node:
                        current_node = node
                        break
                break
    
    # 정렬되지 않은 그룹 추가
    for i, (group, _) in enumerate(group_nodes):
        if i not in used_groups:
            ordered_groups.append(group)
    
    return ordered_groups


def drawDiagramFromSpiceComponents(components: List[Dict], 
                                   circuit_levels: List[List[Dict]]) -> schemdraw.Drawing:
    """
    SPICE 컴포넌트로부터 회로도를 그립니다.
    """
    d = schemdraw.Drawing()
    
    # 전압원 찾기
    voltage_sources = [c for c in components if c['type'] == 'VoltageSource']
    
    if voltage_sources:
        # 전압원 그리기
        vs = voltage_sources[0]
        d += e.SourceV().label(f"{vs['name']}\n{vs['value']}V")
        d.push()
    
    # 각 레벨의 컴포넌트 그리기
    for level_idx, level in enumerate(circuit_levels):
        if not level:
            continue
        
        level_size = len(level)
        
        if level_size == 1:
            # 단일 컴포넌트
            comp = level[0]
            element = get_spice_component_element(comp)
            d += element
            
        else:
            # 병렬 컴포넌트
            d += e.Line().right(d.unit/4).linewidth(0)
            d.push()
            
            # 병렬 브랜치 그리기
            spacing = 1.0
            for i, comp in enumerate(level):
                if i > 0:
                    d.pop()
                    d.push()
                
                # 수직 오프셋 계산
                offset = (i - (level_size-1)/2) * spacing
                
                if offset != 0:
                    d += e.Line().up(offset * d.unit).linewidth(0)
                
                element = get_spice_component_element(comp)
                d += element
                
                if offset != 0:
                    d += e.Line().down(offset * d.unit).linewidth(0)
            
            d.pop()
            d += e.Line().right(d.unit/4).linewidth(0)
    
    # 회로 닫기
    if voltage_sources:
        d.pop()
        d += e.Line().down()
        d += e.Line().left()
    
    return d


def get_spice_component_element(comp: Dict) -> schemdraw.elements.Element:
    """
    SPICE 컴포넌트 정보로부터 schemdraw 엘리먼트를 생성합니다.
    """
    comp_type = comp['type']
    name = comp['name']
    
    if comp_type == 'Resistor':
        value = comp.get('value', 0)
        label = f"{name}\n{format_value(value, 'Ω')}"
        return e.Resistor().right().label(label)
    
    elif comp_type == 'Capacitor':
        value = comp.get('value', 0)
        label = f"{name}\n{format_value(value, 'F')}"
        return e.Capacitor().right().label(label)
    
    elif comp_type == 'Inductor':
        value = comp.get('value', 0)
        label = f"{name}\n{format_value(value, 'H')}"
        return e.Inductor2().right().label(label)
    
    elif comp_type == 'Diode':
        model = comp.get('model', '')
        label = f"{name}\n{model}" if model else name
        return e.Diode().right().label(label)
    
    elif comp_type == 'CurrentSource':
        value = comp.get('value', 0)
        label = f"{name}\n{format_value(value, 'A')}"
        return e.SourceI().right().label(label)
    
    elif comp_type == 'VoltageSource':
        value = comp.get('value', 0)
        label = f"{name}\n{value}V"
        return e.SourceV().right().label(label)
    
    else:
        # 기본값으로 저항 사용
        return e.Resistor().right().label(name)


def format_value(value: float, unit: str) -> str:
    """
    숫자 값을 적절한 단위와 함께 포맷팅합니다.
    """
    if value == 0:
        return f"0{unit}"
    
    # 단위 접두사
    prefixes = [
        (1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'k'),
        (1, ''), (1e-3, 'm'), (1e-6, 'μ'), (1e-9, 'n'), (1e-12, 'p')
    ]
    
    for scale, prefix in prefixes:
        if abs(value) >= scale:
            return f"{value/scale:.2g}{prefix}{unit}"
    
    return f"{value:.2e}{unit}"


# 기존 generate_circuit 함수 수정
def generate_circuit_from_spice(spice_file: str, output_img: str):
    """
    SPICE 파일로부터 회로도를 생성하는 간단한 래퍼 함수
    """
    try:
        # SPICE 파일로부터 회로도 생성
        d = drawDiagramFromSpice(spice_file)
        
        if d:
            d.draw()
            d.save(output_img)
            print(f"✅ SPICE 기반 회로도 생성 완료: {output_img}")
            
            # OpenCV 버전도 저장
            try:
                img_cv = render_drawing_to_cv2(d)
                cv2.imwrite(output_img.replace('.jpg', '_cv.jpg'), img_cv)
            except Exception as e:
                print(f"OpenCV 변환 실패: {e}")
        else:
            print("❌ 회로도 생성 실패")
            
    except Exception as e:
        print(f"❌ SPICE 파싱 오류: {e}")
        import traceback
        traceback.print_exc()

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

    cv2.setMouseCallback(window_name, click_event)
    for msg in prompts:
        print(msg)
    cv2.waitKey(0)
    return pts

def find_parallel_groups(component_nets):
    """
    같은 두 넷에 연결된 컴포넌트들을 병렬 그룹으로 묶습니다.
    """
    groups = []
    processed = set()
    
    for comp_name, nets in component_nets.items():
        if comp_name in processed:
            continue
        
        # 현재 컴포넌트와 같은 넷에 연결된 다른 컴포넌트들 찾기
        current_group = [comp_name]
        processed.add(comp_name)
        
        for other_comp, other_nets in component_nets.items():
            if other_comp != comp_name and other_comp not in processed:
                # 정확히 같은 넷에 연결되어 있으면 병렬
                if nets == other_nets:
                    current_group.append(other_comp)
                    processed.add(other_comp)
        
        groups.append(current_group)
    
    return groups

def analyze_circuit_topology_fixed(G):
    """
    networkx Graph를 분석하여 직렬/병렬 구조를 올바르게 추출합니다.
    수정된 버전: 병렬 연결을 정확히 검출
    """
    # 1) 전압원 찾기
    voltage_nodes = []
    other_components = []
    
    for node, data in G.nodes(data=True):
        if data.get('comp_class') == 'VoltageSource':
            voltage_nodes.append(node)
        else:
            other_components.append({
                'name': node,
                'class': data.get('comp_class'),
                'value': data.get('value', 0)
            })
    
    if not voltage_nodes:
        print("Warning: No voltage source found")
        return []
    
    voltage_node = voltage_nodes[0]
    print(f"DEBUG: Voltage node: {voltage_node}")
    
    # 2) 넷 분석을 통한 병렬 구조 검출
    # 각 컴포넌트가 연결된 넷 정보 추출
    component_nets = {}
    for comp_name in [c['name'] for c in other_components]:
        nets_str = G.nodes[comp_name].get('nets', '')
        if nets_str:
            nets = [int(net) for net in nets_str.split(',')]
            component_nets[comp_name] = set(nets)
            print(f"DEBUG: {comp_name} connected to nets: {nets}")
    
    # 3) 병렬 그룹 찾기
    parallel_groups = find_parallel_groups(component_nets)
    print(f"DEBUG: Parallel groups: {parallel_groups}")
    
    # 4) 직렬 순서 결정
    circuit_levels = build_circuit_levels(parallel_groups, component_nets, other_components)
    print(f"DEBUG: Circuit levels: {circuit_levels}")
    
    return circuit_levels

def build_circuit_levels(parallel_groups, component_nets, other_components):
    """
    병렬 그룹들을 직렬 순서로 배열하여 circuit levels를 구성합니다.
    """
    # 컴포넌트 정보 매핑
    comp_info = {comp['name']: comp for comp in other_components}
    
    # 각 그룹의 넷 연결 정보
    group_nets = {}
    for i, group in enumerate(parallel_groups):
        # 그룹의 첫 번째 컴포넌트의 넷 정보를 그룹 대표로 사용
        group_nets[i] = component_nets[group[0]]
    
    # 넷 기준으로 그룹들을 정렬
    # 가장 작은 넷 번호 기준으로 정렬
    sorted_groups = sorted(enumerate(parallel_groups), 
                          key=lambda x: min(group_nets[x[0]]))
    
    # 컴포넌트 정보 포함하여 최종 레벨 구성
    circuit_levels = []
    for _, group in sorted_groups:
        level = []
        for comp_name in group:
            if comp_name in comp_info:
                level.append(comp_info[comp_name])
        if level:
            circuit_levels.append(level)
    
    return circuit_levels




def find_path_excluding_voltage(G, start, end, voltage_nodes):
    """
    전압원을 제외하고 start에서 end까지의 경로를 찾습니다.
    """
    from collections import deque
    
    queue = deque([(start, [start])])
    visited = set([start])
    
    while queue:
        node, path = queue.popleft()
        
        if node == end:
            return path
        
        for neighbor in G.neighbors(node):
            if neighbor not in visited and neighbor not in voltage_nodes:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return None


def group_parallel_components(G, ordered_components, voltage_nodes):
    """
    순서가 정해진 컴포넌트들 중에서 병렬 연결된 것들을 그룹화합니다.
    """
    if len(ordered_components) <= 1:
        return [ordered_components] if ordered_components else []
    
    # net 연결 정보를 이용해 병렬 구조 찾기
    levels = []
    i = 0
    
    while i < len(ordered_components):
        current_group = [ordered_components[i]]
        current_node = ordered_components[i]['name']
        
        # 같은 두 노드에 연결된 다른 컴포넌트들 찾기
        current_nets = get_component_nets(G, current_node)
        
        j = i + 1
        while j < len(ordered_components):
            other_node = ordered_components[j]['name']
            other_nets = get_component_nets(G, other_node)
            
            # 같은 net에 연결되어 있으면 병렬
            if current_nets == other_nets:
                current_group.append(ordered_components[j])
                ordered_components.pop(j)
            else:
                j += 1
        
        levels.append(current_group)
        i += 1
    
    return levels


def get_component_nets(G, node_name):
    """
    컴포넌트가 연결된 net 정보를 반환합니다.
    """
    node_data = G.nodes[node_name]
    nets_str = node_data.get('nets', '')
    
    if nets_str:
        return set(nets_str.split(','))
    else:
        return set()


def drawDiagramFromGraph_fixed(G, voltage=5.0):
    """
    수정된 networkx Graph 기반 회로도 생성 함수 (병렬 구조 개선)
    """
    # 1) 수정된 회로 토폴로지 분석
    circuit_levels = analyze_circuit_topology_fixed(G)
    
    if not circuit_levels:
        print("No circuit levels found")
        return None
    
    print(f"DEBUG: Found {len(circuit_levels)} levels:")
    for i, level in enumerate(circuit_levels):
        print(f"  Level {i}: {[comp['name'] for comp in level]}")
    
    # 2) diagram_origin.py 스타일로 회로도 생성
    return drawDiagram_fixed_v2(voltage, circuit_levels)


def drawDiagram_fixed_v2(voltage, circuit_levels):
    """
    수정된 회로도 그리기 함수 - 병렬 구조를 정확히 반영
    """
    import schemdraw
    import schemdraw.elements as e
    
    d = schemdraw.Drawing()
    d.push()
    
    components = []
    
    for level_idx, level in enumerate(circuit_levels):
        level_size = len(level)
        
        print(f"DEBUG: Drawing level {level_idx} with {level_size} components: {[c['name'] for c in level]}")
        
        if level_size == 1:
            # 단일 컴포넌트 (직렬)
            comp = level[0]
            with schemdraw.Drawing(show=False) as single_comp:
                element = get_component_element(comp)
                single_comp += element
            components.append(single_comp)
            
        elif level_size == 2:
            # 병렬 컴포넌트 2개
            with schemdraw.Drawing(show=False) as parallel_2:
                parallel_2 += e.Line().right(parallel_2.unit/4)
                parallel_2.push()
                
                # 위쪽 컴포넌트
                parallel_2 += e.Line().up(parallel_2.unit/2)
                element1 = get_component_element(level[0])
                parallel_2 += element1
                parallel_2 += e.Line().down(parallel_2.unit/2)
                parallel_2.pop()
                
                # 아래쪽 컴포넌트  
                parallel_2 += e.Line().down(parallel_2.unit/2)
                element2 = get_component_element(level[1])
                parallel_2 += element2
                parallel_2 += e.Line().up(parallel_2.unit/2)
                
                # 연결선
                parallel_2 += e.Line().right(parallel_2.unit/4)
            components.append(parallel_2)
            
        elif level_size == 3:
            # 병렬 컴포넌트 3개
            with schemdraw.Drawing(show=False) as parallel_3:
                parallel_3 += e.Line().right(parallel_3.unit/4)
                parallel_3.push()
                
                # 위쪽
                parallel_3 += e.Line().up(parallel_3.unit/2)
                element1 = get_component_element(level[0])
                parallel_3 += element1
                parallel_3 += e.Line().down(parallel_3.unit/2)
                parallel_3.pop()
                
                # 중간
                parallel_3.push()
                element2 = get_component_element(level[1])
                parallel_3 += element2
                parallel_3.pop()
                
                # 아래쪽
                parallel_3 += e.Line().down(parallel_3.unit/2)
                element3 = get_component_element(level[2])
                parallel_3 += element3
                parallel_3 += e.Line().up(parallel_3.unit/2)
                
                # 연결선
                parallel_3 += e.Line().right(parallel_3.unit/4)
            components.append(parallel_3)
            
        else:
            # 4개 이상의 병렬 컴포넌트
            with schemdraw.Drawing(show=False) as multi_parallel:
                multi_parallel += e.Line().right(multi_parallel.unit/4)
                multi_parallel.push()
                
                # 수직 간격 계산
                spacing = 0.8
                for i, comp in enumerate(level):
                    if i > 0:
                        multi_parallel.pop()
                        multi_parallel.push()
                    
                    # 중심을 기준으로 대칭 배치
                    vertical_offset = (i - (level_size-1)/2) * spacing
                    
                    if vertical_offset != 0:
                        multi_parallel += e.Line().up(vertical_offset * multi_parallel.unit)
                    
                    element = get_component_element(comp)
                    multi_parallel += element
                    
                    if vertical_offset != 0:
                        multi_parallel += e.Line().down(vertical_offset * multi_parallel.unit)
                
                multi_parallel.pop()
                multi_parallel += e.Line().right(multi_parallel.unit/4)
            components.append(multi_parallel)
    
    # 컴포넌트들을 직렬로 연결
    for comp in components:
        d += e.ElementDrawing(comp)
    
    # 전원 및 연결선 추가
    d += (n1 := e.Dot())
    d += e.Line().down().at(n1.end)
    d += (n2 := e.Dot())
    d.pop()
    d += (n3 := e.Dot())
    d += e.SourceV().down().label(f"{voltage}V").at(n3.end).reverse()
    d += (n4 := e.Dot())
    d += e.Line().right().endpoints(n4.end, n2.end)
    
    return d

def detect_parallel_groups(level_components, G):
    """
    한 레벨 내에서 병렬 연결된 컴포넌트 그룹을 찾습니다.
    """
    if len(level_components) <= 1:
        return [level_components]
    
    # 각 컴포넌트의 연결 상태 분석
    groups = []
    ungrouped = level_components.copy()
    
    while ungrouped:
        current_group = [ungrouped[0]]
        ungrouped.remove(ungrouped[0])
        
        # 같은 노드에 연결된 컴포넌트들 찾기
        for comp in current_group:
            comp_neighbors = set(G.neighbors(comp['name']))
            
            for other in ungrouped.copy():
                other_neighbors = set(G.neighbors(other['name']))
                
                # 공통 이웃이 있으면 병렬 연결
                if comp_neighbors & other_neighbors:
                    current_group.append(other)
                    ungrouped.remove(other)
        
        groups.append(current_group)
    
    return groups


def drawDiagramFromGraph(G, voltage=5.0):
    """
    networkx Graph로부터 회로도를 그립니다 (diagram_origin.py 스타일)
    """
    # 1) 회로 토폴로지 분석
    circuit_levels = analyze_circuit_topology_fixed(G)
    
    if not circuit_levels:
        print("No circuit levels found")
        return None
    
    # 2) 각 레벨에서 병렬 그룹 검출
    processed_levels = []
    for level in circuit_levels:
        parallel_groups = detect_parallel_groups(level, G)
        
        # 그룹별로 컴포넌트 정리
        level_groups = []
        for group in parallel_groups:
            if len(group) == 1:
                # 단일 컴포넌트
                level_groups.append(group)
            else:
                # 병렬 그룹
                level_groups.append(group)
        
        processed_levels.extend(level_groups)
    
    # 3) schemdraw로 그리기
    return drawDiagram(voltage, processed_levels)


def drawDiagram(voltage, circuit_levels):
    """
    diagram_origin.py 기반의 회로도 그리기 함수 (개선된 버전)
    """
    d = schemdraw.Drawing()
    d.push()
    
    components = []
    
    for level in circuit_levels:
        level_size = len(level)
        
        if level_size == 1:
            # 단일 컴포넌트
            comp = level[0]
            with schemdraw.Drawing(show=False) as single_comp:
                element = get_component_element(comp)
                single_comp += element
            components.append(single_comp)
            
        elif level_size == 2:
            # 병렬 저항 2개
            with schemdraw.Drawing(show=False) as parallel_2:
                parallel_2 += e.Line().right(parallel_2.unit/4)
                parallel_2.push()
                
                # 위쪽 컴포넌트
                parallel_2 += e.Line().up(parallel_2.unit/2)
                element1 = get_component_element(level[0])
                parallel_2 += element1
                parallel_2 += e.Line().down(parallel_2.unit/2)
                parallel_2.pop()
                
                # 아래쪽 컴포넌트  
                parallel_2 += e.Line().down(parallel_2.unit/2)
                element2 = get_component_element(level[1])
                parallel_2 += element2
                parallel_2 += e.Line().up(parallel_2.unit/2)
                
                # 연결선
                parallel_2 += e.Line().right(parallel_2.unit/4)
            components.append(parallel_2)
            
        elif level_size == 3:
            # 병렬 저항 3개
            with schemdraw.Drawing(show=False) as parallel_3:
                parallel_3 += e.Line().right(parallel_3.unit/4)
                parallel_3.push()
                
                # 위쪽
                parallel_3 += e.Line().up(parallel_3.unit/2)
                element1 = get_component_element(level[0])
                parallel_3 += element1
                parallel_3 += e.Line().down(parallel_3.unit/2)
                parallel_3.pop()
                
                # 중간
                parallel_3.push()
                element2 = get_component_element(level[1])
                parallel_3 += element2
                parallel_3.pop()
                
                # 아래쪽
                parallel_3 += e.Line().down(parallel_3.unit/2)
                element3 = get_component_element(level[2])
                parallel_3 += element3
                parallel_3 += e.Line().up(parallel_3.unit/2)
                
                # 연결선
                parallel_3 += e.Line().right(parallel_3.unit/4)
            components.append(parallel_3)
            
        else:
            # 4개 이상인 경우 간단히 처리
            with schemdraw.Drawing(show=False) as multi_parallel:
                multi_parallel += e.Line().right(multi_parallel.unit/4)
                multi_parallel.push()
                
                spacing = 1.0 / level_size
                for i, comp in enumerate(level):
                    if i > 0:
                        multi_parallel.pop()
                        multi_parallel.push()
                    
                    offset = (i - (level_size-1)/2) * spacing
                    if offset != 0:
                        multi_parallel += e.Line().up(offset * multi_parallel.unit)
                    
                    element = get_component_element(comp)
                    multi_parallel += element
                    
                    if offset != 0:
                        multi_parallel += e.Line().down(offset * multi_parallel.unit)
                
                multi_parallel.pop()
                multi_parallel += e.Line().right(multi_parallel.unit/4)
            components.append(multi_parallel)
    
    # 컴포넌트들을 직렬로 연결
    for comp in components:
        d += e.ElementDrawing(comp)
    
    # 전원 및 연결선 추가
    d += (n1 := e.Dot())
    d += e.Line().down().at(n1.end)
    d += (n2 := e.Dot())
    d.pop()
    d += (n3 := e.Dot())
    d += e.SourceV().down().label(f"{voltage}V").at(n3.end).reverse()
    d += (n4 := e.Dot())
    d += e.Line().right().endpoints(n4.end, n2.end)
    
    return d


def get_component_element(comp):
    """
    컴포넌트 정보로부터 schemdraw 엘리먼트를 생성합니다.
    """
    comp_class = comp.get('class', '')
    name = comp.get('name', 'Unknown')
    value = comp.get('value', 0)
    
    if comp_class == 'Resistor':
        label_text = f"{name}\n{value}Ω"
        return e.Resistor().right().label(label_text)
    elif comp_class == 'Capacitor':
        label_text = f"{name}\n{value}F"
        return e.Capacitor().right().label(label_text)
    elif comp_class == 'Diode':
        return e.Diode().right().label(name)
    elif comp_class == 'LED':
        return e.LED().right().label(name)
    elif comp_class == 'IC':
        return e.RBox(width=2, height=1).right().label(name)
    else:
        return e.Resistor().right().label(f"{name}\n{value}")


# 기존 함수들 유지 (호환성을 위해)
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
    #plt.show()
    return G

def render_drawing_to_cv2(drawing: schemdraw.Drawing, dpi: int = 200) -> np.ndarray:
    """
    Schemdraw Drawing 객체를 PIL→OpenCV BGR 이미지(Numpy)로 변환하여 반환합니다.
    """
    try:
        # 방법 1: schemdraw의 내장 get_imagedata 메서드 사용
        if hasattr(drawing, 'get_imagedata'):
            try:
                # PNG 형태로 이미지 데이터 얻기
                img_data = drawing.get_imagedata('png')
                
                # 바이트 데이터를 PIL Image로 변환
                pil_img = Image.open(BytesIO(img_data)).convert('RGB')
                arr = np.array(pil_img)
                
                # RGB → BGR(OpenCV)로 변환
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except:
                pass
        
        # 방법 2: matplotlib을 이용한 변환
        # 먼저 drawing을 그리기
        if not hasattr(drawing, '_drawn') or not drawing._drawn:
            drawing.draw()
        
        # 현재 matplotlib figure 가져오기
        import matplotlib.pyplot as plt
        fig = plt.gcf()
        
        # 메모리 버퍼에 PNG로 저장
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        
        # PIL로 읽어서 numpy array 변환
        pil_img = Image.open(buf).convert('RGB')
        arr = np.array(pil_img)
        
        # 버퍼 정리
        buf.close()
        
        # RGB → BGR(OpenCV)로 변환
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        
    except Exception as e:
        print(f"Warning: Failed to convert drawing to OpenCV format: {e}")
        
        # 방법 3: 빈 이미지 반환 (fallback)
        # 대안으로 빈 흰색 이미지 생성
        height, width = 400, 600
        blank_img = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # 중앙에 "Error" 텍스트 추가
        cv2.putText(blank_img, "Diagram Generation Error", 
                   (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 0, 255), 2)
        
        return blank_img


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


# diagram.py에 추가할 개선된 함수들

import networkx as nx
from collections import defaultdict

def analyze_circuit_connectivity(G):
    """
    회로 그래프의 연결 상태를 분석하여 연결된 컴포넌트 그룹들을 반환합니다.
    
    Returns:
        list: 각 원소는 연결된 컴포넌트 그룹 (disconnected인 경우 여러 그룹)
        bool: 전체 회로가 연결되어 있는지 여부
    """
    # 1) 전압원 제외한 그래프 생성 (전압원은 회로 연결성과 별개)
    non_voltage_nodes = [
        node for node, data in G.nodes(data=True) 
        if data.get('comp_class') != 'VoltageSource'
    ]
    
    if not non_voltage_nodes:
        return [], False
    
    # 전압원을 제외한 서브그래프
    sub_G = G.subgraph(non_voltage_nodes)
    
    # 2) 연결된 컴포넌트 그룹 찾기
    connected_components = list(nx.connected_components(sub_G))
    
    # 3) 각 그룹을 컴포넌트 정보로 변환
    component_groups = []
    for component_set in connected_components:
        group = []
        for node in component_set:
            node_data = G.nodes[node]
            group.append({
                'name': node,
                'class': node_data.get('comp_class'),
                'value': node_data.get('value', 0)
            })
        component_groups.append(group)
    
    # 4) 전체 연결성 확인
    is_fully_connected = len(connected_components) == 1
    
    return component_groups, is_fully_connected


def analyze_circuit_topology_improved(G):
    """
    개선된 회로 토폴로지 분석 - 연결 상태를 고려함
    """
    print("=== 회로 연결성 분석 시작 ===")
    
    # 1) 연결 상태 분석
    component_groups, is_connected = analyze_circuit_connectivity(G)
    
    print(f"연결된 그룹 수: {len(component_groups)}")
    print(f"전체 연결 상태: {'연결됨' if is_connected else '끊어짐'}")
    
    if not is_connected:
        print("⚠️  경고: 회로가 완전히 연결되지 않았습니다!")
        for i, group in enumerate(component_groups):
            print(f"  그룹 {i+1}: {[comp['name'] for comp in group]}")
    
    # 2) 각 연결된 그룹에 대해 직렬/병렬 분석
    all_circuit_levels = []
    
    for group_idx, group in enumerate(component_groups):
        print(f"\n--- 그룹 {group_idx+1} 분석 ---")
        
        # 해당 그룹만의 서브그래프 생성
        group_nodes = [comp['name'] for comp in group]
        group_graph = G.subgraph(group_nodes)
        
        # 병렬 구조 분석
        component_nets = {}
        for comp_name in group_nodes:
            nets_str = G.nodes[comp_name].get('nets', '')
            if nets_str:
                nets = [int(net) for net in nets_str.split(',')]
                component_nets[comp_name] = set(nets)
        
        # 병렬 그룹 찾기
        parallel_groups = find_parallel_groups(component_nets)
        print(f"병렬 그룹: {parallel_groups}")
        
        # 컴포넌트 정보와 결합
        group_levels = []
        for p_group in parallel_groups:
            level = []
            for comp_name in p_group:
                comp_info = next(comp for comp in group if comp['name'] == comp_name)
                level.append(comp_info)
            group_levels.append(level)
        
        all_circuit_levels.extend(group_levels)
    
    return all_circuit_levels, is_connected


def drawDiagramFromGraph_with_connectivity_check(G, voltage=5.0):
    """
    연결 상태를 확인하여 끊어진 회로는 별도로 표시하는 개선된 함수
    """
    # 1) 연결성 분석
    circuit_levels, is_connected = analyze_circuit_topology_improved(G)
    
    if not circuit_levels:
        print("❌ 그릴 수 있는 회로 요소가 없습니다.")
        return None
    
    # 2) 연결되지 않은 경우 경고 메시지와 함께 부분 회로도 생성
    if not is_connected:
        print("⚠️  주의: 연결되지 않은 회로 요소들이 있습니다. 연결된 부분만 그립니다.")
        
        # 각 연결된 그룹별로 별도 다이어그램 생성 가능
        # 또는 모든 그룹을 하나의 다이어그램에 표시 (점선으로 구분)
    
    # 3) 회로도 그리기
    return drawDiagram_with_disconnection_indicator(voltage, circuit_levels, is_connected)


def drawDiagram_with_disconnection_indicator(voltage, circuit_levels, is_connected):
    """
    연결 끊김을 시각적으로 표시하는 회로도 그리기
    """
    import schemdraw
    import schemdraw.elements as e
    
    d = schemdraw.Drawing()
    d.config(unit=3.0)  # 기본 단위 크기를 3배로 확대
    d.config(fontsize=14)  # 폰트 크기도 키우기
    
    d.push()
    # 연결되지 않은 경우 제목에 경고 표시
    if not is_connected:
        d += e.Label().label("⚠️ DISCONNECTED CIRCUIT ⚠️").color('red').at((0, 1))
    
    d.push()
    
    components = []
    
    for level_idx, level in enumerate(circuit_levels):
        level_size = len(level)
        
        # 연결 끊김을 나타내는 특별한 처리
        if level_idx > 0 and not is_connected:
            # 점선으로 끊어진 연결 표시
            d += e.Line().linestyle('--').color('red').length(1)
            d += e.Label().label("BREAK").color('red').fontsize(8)
        
        # 기존 레벨 그리기 로직
        if level_size == 1:
            comp = level[0]
            element = get_component_element(comp)
            d += element
            
        elif level_size == 2:
            # 병렬 2개
            d += e.Line().right(d.unit/4)
            d.push()
            
            # 위쪽
            d += e.Line().up(d.unit/2)
            element1 = get_component_element(level[0])
            d += element1
            d += e.Line().down(d.unit/2)
            d.pop()
            
            # 아래쪽  
            d += e.Line().down(d.unit/2)
            element2 = get_component_element(level[1])
            d += element2
            d += e.Line().up(d.unit/2)
            
            d += e.Line().right(d.unit/4)
            
        # ... (기타 병렬 조합 처리)
    
    # 전원 연결 (연결된 경우만)
    if is_connected:
        d += (n1 := e.Dot())
        d += e.Line().down().at(n1.end)
        d += (n2 := e.Dot())
        d.pop()
        d += (n3 := e.Dot())
        d += e.SourceV().down().label(f"{voltage}V").at(n3.end).reverse()
        d += (n4 := e.Dot())
        d += e.Line().right().endpoints(n4.end, n2.end)
    else:
        # 연결되지 않은 경우 전원을 별도로 표시하거나 생략
        d += e.Label().label("전원 연결 불가 - 회로 끊어짐").color('red')
    
    return d


def validate_circuit_connectivity(G):
    """
    회로의 연결성을 검증하고 문제점을 리포트
    """
    component_groups, is_connected = analyze_circuit_connectivity(G)
    
    report = {
        'is_connected': is_connected,
        'num_groups': len(component_groups),
        'groups': component_groups,
        'issues': []
    }
    
    if not is_connected:
        report['issues'].append(f"회로가 {len(component_groups)}개 그룹으로 분리됨")
        
        # 각 그룹의 크기 분석
        group_sizes = [len(group) for group in component_groups]
        isolated_components = [i for i, size in enumerate(group_sizes) if size == 1]
        
        if isolated_components:
            isolated_names = [component_groups[i][0]['name'] for i in isolated_components]
            report['issues'].append(f"고립된 컴포넌트: {isolated_names}")
        
        # 전압원 연결 확인
        has_voltage_source = any(
            data.get('comp_class') == 'VoltageSource' 
            for node, data in G.nodes(data=True)
        )
        
        if has_voltage_source and not is_connected:
            report['issues'].append("전압원이 일부 컴포넌트와 연결되지 않음")
    
    return report


# circuit_generator.py의 generate_circuit 함수에 추가할 검증 코드
def add_connectivity_validation_to_generate_circuit():
    """
    generate_circuit 함수에 추가할 연결성 검증 코드
    """
    # generate_circuit 함수 내에서 그래프 생성 후 추가:
    
    # G = build_circuit_graph(mapped) 다음에 추가
    
    # 연결성 검증
    connectivity_report = validate_circuit_connectivity(G)
    
    if not connectivity_report['is_connected']:
        print("\n🚨 회로 연결성 문제 감지!")
        for issue in connectivity_report['issues']:
            print(f"  - {issue}")
        
        print(f"\n연결된 그룹별 컴포넌트:")
        for i, group in enumerate(connectivity_report['groups']):
            component_names = [comp['name'] for comp in group]
            print(f"  그룹 {i+1}: {component_names}")
    else:
        print("\n✅ 회로가 올바르게 연결되었습니다.")
    
    return connectivity_report

# 새로운 함수 추가
def draw_connectivity_graph_from_nx(G, output_path=None, show = False):
    """
    이미 생성된 networkx Graph를 시각화
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    if len(G.nodes()) == 0:
        print("Empty graph - skipping visualization")
        return
    
    # 컴포넌트 타입별 색상 정의
    color_map = {
        'VoltageSource': '#FF6B6B',  # 빨간색
        'Resistor': '#4ECDC4',       # 청록색
        'Capacitor': '#45B7D1',      # 파란색
        'Diode': '#96CEB4',          # 초록색
        'LED': '#FFEAA7',            # 노란색
        'IC': '#DDA0DD',             # 보라색
        'Wire': '#95A5A6',           # 회색
        'Unknown': '#BDC3C7'         # 연회색
    }
    
    # 레이아웃 계산
    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)
    
    # 노드 색상 설정
    node_colors = []
    node_labels = {}
    
    for node, data in G.nodes(data=True):
        # 클래스 정규화
        comp_type = data.get('comp_class') or data.get('type') or 'Unknown'
        if comp_type in ['VoltageSource', 'V+', 'V-']:
            comp_type = 'VoltageSource'
        
        node_colors.append(color_map.get(comp_type, '#BDC3C7'))
        
        # 라벨 생성
        label = str(node)
        if 'value' in data and data['value'] != 0:
            if comp_type == 'Resistor':
                label += f"\n{data['value']}Ω"
            elif comp_type == 'VoltageSource':
                label += f"\n{data['value']}V"
            elif comp_type == 'Capacitor':
                label += f"\n{data['value']}F"
        node_labels[node] = label
    
    # 그래프 그리기
    plt.figure(figsize=(20, 16))
    
    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                          node_size=1500, alpha=0.9, edgecolors='black', linewidths=1.5)
    
    # 엣지 그리기
    nx.draw_networkx_edges(G, pos, edge_color='#7F8C8D', 
                          width=2, alpha=0.7)
    
    # 라벨 그리기
    nx.draw_networkx_labels(G, pos, labels=node_labels, 
                           font_size=9, font_weight='bold')
    
    # 엣지 라벨 (nets 정보가 있으면)
    edge_labels = {}
    for u, v, data in G.edges(data=True):
        if 'nets' in data:
            edge_labels[(u, v)] = str(data['nets'])
    
    if edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels, 
                                    font_size=8, font_color='red')
    
    # 제목 및 정보
    plt.title(f"Circuit Connectivity Graph\nNodes: {len(G.nodes())}, Edges: {len(G.edges())}", 
             fontsize=14, fontweight='bold')
    plt.axis('off')
    
    # 범례 추가
    legend_elements = []
    used_types = set()
    for _, data in G.nodes(data=True):
        comp_type = data.get('comp_class') or data.get('type') or 'Unknown'
        if comp_type in ['VoltageSource', 'V+', 'V-']:
            comp_type = 'VoltageSource'
        used_types.add(comp_type)
    
    for comp_type in sorted(used_types):
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color_map.get(comp_type, '#BDC3C7'),
                                        markersize=10, label=comp_type))
    
    plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Connectivity graph saved: {output_path}")
    #노드 그래프 시각화 부분
    '''if show == True:
        plt.show()'''
    return plt.gcf()




if __name__=='__main__':
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--power-plus', nargs=2, type=int, help='net x')
    parser.add_argument('--power-minus', nargs=2, type=int, help='net x')
    parser.add_argument('--voltage', type=float, default=5)
    parser.add_argument('--output', type=str, default='diagram.jpg')
    args = parser.parse_args()
    
    # 예시 사용 - networkx Graph로부터 회로도 생성
    from circuit_generator import build_circuit_graph
    
    # 예시 회로 데이터
    mapped = [
        {"name":"R1","class":"Resistor","value":100,"nodes":(1,2)},
        {"name":"R2","class":"Resistor","value":200,"nodes":(2,3)},
        {"name":"R3","class":"Resistor","value":300,"nodes":(2,3)},  # R2와 병렬
        {"name":"V1","class":"VoltageSource","value":5,"nodes":(1,3)}
    ]
    
    G = build_circuit_graph(mapped)
    d = drawDiagramFromGraph_fixed(G, args.voltage)
    
    if d:
        d.draw()
        d.save(args.output)
        print(f"Saved: {args.output}")
    else:
        print("Failed to generate diagram")