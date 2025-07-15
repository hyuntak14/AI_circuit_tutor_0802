#!/usr/bin/env python3
"""
SPICE Netlist to Circuit Diagram Converter
Converts all SPICE netlist files in the current directory to SVG or Schemdraw schematics
"""

# [FIXED] 미사용 re 모듈 제거, 중복 import 정리
import os
import math
import collections
import glob
import sys
import cairosvg
import argparse
from typing import List, Set, Dict, Tuple
import re
import schemdraw
import schemdraw.elements as elm
import matplotlib.pyplot as plt  # PNG 저장을 위해 추가

class SpiceComponent:
    """Represents a SPICE component"""
    def __init__(self, name: str, type_: str, nodes: List[str], value: str):
        self.name = name
        self.type = type_
        self.nodes = nodes
        self.node1 = nodes[0] if len(nodes) > 0 else None
        self.node2 = nodes[1] if len(nodes) > 1 else None
        self.value = value

class SpiceParser:
    """Parses SPICE netlist files"""
    def __init__(self, filename: str):
        self.filename = filename
        self.components: List[SpiceComponent] = []
        self.nodes: Set[str] = set()
        self.ground_nodes: Set[str] = set()

    # [IMPROVED] 4단자 이상 소자 및 값 없는 소자도 파싱 가능하도록 개선
    def parse(self) -> List[SpiceComponent]:
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip().lower() # SPICE는 대소문자 구분이 없으므로 소문자로 통일
            if not line or line.startswith('*') or line.startswith('.'):
                continue
            
            parts = line.split()
            name = parts[0]
            type_ = name[0].upper()
            nodes = []
            value = ''
            
            # 마지막 부분이 값인지 노드인지 확인
            if len(parts) > 1:
                # 간단한 값 형식 감지 (숫자로 시작하거나 'k', 'm', 'u', 'n', 'p', 'f' 등 포함)
                if re.match(r'^[0-9\.]+[a-z]*$', parts[-1]):
                    value = parts[-1]
                    nodes = parts[1:-1]
                else:
                    nodes = parts[1:]

            comp = SpiceComponent(name, type_, nodes, value)
            self.components.append(comp)
            self.nodes.update(nodes)

        # [FIXED] Ground 노드 감지 로직 수정
        # 1. 명시적인 ground 노드 찾기
        explicit_gnd = {'0', 'gnd', 'ground'}
        self.ground_nodes = {n for n in self.nodes if n in explicit_gnd}

        # 2. 명시적인 ground가 없을 경우, 전압원의 공통 노드를 ground로 추정
        if not self.ground_nodes:
            voltages = [c for c in self.components if c.type == 'V']
            if voltages:
                # 모든 전압원의 음극(-) 노드를 후보로 탐색
                common_nodes = set(v.node2 for v in voltages if v.node2 is not None)
                if len(common_nodes) == 1:
                     self.ground_nodes.add(common_nodes.pop())

        return self.components

# [IMPROVED] CircuitLayout은 schemdraw 방식에서는 불필요하므로 SVG 전용으로 둠
class CircuitLayout:
    """Handles automatic layout of circuit components for custom SVG rendering."""
    def __init__(self, components: List[SpiceComponent], nodes: Set[str]):
        self.components = components
        self.nodes = nodes
        self.node_positions: Dict[str, Tuple[int, int]] = {}

    def calculate_layout(self):
        """A simple grid-based layout algorithm."""
        import math
        node_list = sorted(list(self.nodes))
        n_nodes = len(node_list)
        if n_nodes == 0: return

        cols = math.ceil(math.sqrt(n_nodes))
        rows = math.ceil(n_nodes / cols)
        
        for i, node in enumerate(node_list):
            x = (i % cols) * 150  # 간격 증가
            y = (i // cols) * 150 # 간격 증가
            self.node_positions[node] = (x, y)

# [IMPROVED] SVG 생성자는 이제 소자 좌표만 받고, 연결점(터미널) 좌표를 반환
# [IMPROVED] SVG 생성자는 이제 소자 좌표만 받고, 연결점(터미널) 좌표를 반환
# [IMPROVED] SVG 생성자는 이제 소자 좌표만 받고, 연결점(터미널) 좌표를 반환
class SVGGenerator:
    """Generates SVG circuit diagrams with automatic bounding & offset."""
    def __init__(self, width=800, height=600, x_offset=0, y_offset=0):
        self.width = width
        self.height = height
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.svg_content = []

    def start_svg(self):
        self.svg_content.append(f'<svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">')
        #self.svg_content.append('<defs><pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse"><path d="M 20 0 L 0 0 0 20" fill="none" stroke="#e0e0e0" stroke-width="0.5"/></pattern></defs>')
        self.svg_content.append('<rect width="100%" height="100%" fill="url(#grid)" />')

    def _apply_off(self, x, y):
        return x + self.x_offset, y + self.y_offset

    def draw_wire(self, x1, y1, x2, y2):
        x1o, y1o = self._apply_off(x1, y1)
        x2o, y2o = self._apply_off(x2, y2)
        self.svg_content.append(f'<line x1="{x1o}" y1="{y1o}" x2="{x2o}" y2="{y2o}" stroke="black" stroke-width="1.5"/>')

    def draw_node_label(self, x, y, label):
        xo, yo = self._apply_off(x, y)
        self.svg_content.append(f'<text x="{xo+5}" y="{yo-5}" font-family="sans-serif" font-size="10" fill="blue">{label}</text>')

    def draw_node_dot(self, x, y):
        xo, yo = self._apply_off(x, y)
        self.svg_content.append(f'<circle cx="{xo}" cy="{yo}" r="3" fill="black"/>')
    
    # [IMPROVED] 컴포넌트 드로잉 함수가 터미널 좌표를 반환하도록 수정

    def draw_resistor(self, x, y, name, value, horizontal=True, angle=0):
        # 지그재그 선 파라미터
        segments = 6
        length = 40
        zigzag_amp = 10
        term_len = 10

        # 좌표 계산
        if horizontal:
            x0, y0 = x - length/2, y
            dx = length / segments
            points = [(x0 + i*dx, y0 + (zigzag_amp if i % 2 else -zigzag_amp)) for i in range(1, segments)]
            pts = [(x0, y0)] + points + [(x0+length, y0)]
        else:
            x0, y0 = x, y - length/2
            dy = length / segments
            points = [(x0 + (zigzag_amp if i % 2 else -zigzag_amp), y0 + i*dy) for i in range(1, segments)]
            pts = [(x0, y0)] + points + [(x0, y0+length)]

        # 터미널 좌표
        term1, term2 = pts[0], pts[-1]

        # 회전 그룹 시작
        cx, cy = x + self.x_offset, y + self.y_offset
        if angle != 0:
            angle_deg = math.degrees(angle)
            self.svg_content.append(f'<g transform="rotate({angle_deg}, {cx}, {cy})">')

        # 선 그리기
        path = 'M ' + ' L '.join(f"{px+ self.x_offset},{py+ self.y_offset}" for px,py in pts)
        self.svg_content.append(f'<path d="{path}" stroke="black" fill="none" stroke-width="1.5"/>')

        # 터미널 연장선
        self.draw_wire(term1[0], term1[1], term1[0] - (term_len if horizontal else 0), term1[1] - (term_len if not horizontal else 0))
        self.draw_wire(term2[0], term2[1], term2[0] + (term_len if horizontal else 0), term2[1] + (term_len if not horizontal else 0))

        # 레이블
        self.svg_content.append(f'<text x="{cx}" y="{cy - 15}" text-anchor="middle" font-weight="bold" font-size="12">{name}</text>')
        self.svg_content.append(f'<text x="{cx}" y="{cy + 15}" text-anchor="middle" font-size="10">{value}</text>')

        # 회전 그룹 종료
        if angle != 0:
            self.svg_content.append('</g>')

        return (term1[0] - (term_len if horizontal else 0), term1[1] - (term_len if not horizontal else 0)), \
               (term2[0] + (term_len if horizontal else 0), term2[1] + (term_len if not horizontal else 0))

    def draw_capacitor(self, x, y, name, value, horizontal=True, angle=0):
        plate_w = 10  # 평판 너비
        gap = 4       # 평판 간격
        term_len = 20 # 터미널 길이

        # 회전 그룹 시작
        cx, cy = x + self.x_offset, y + self.y_offset
        if angle != 0:
            angle_deg = math.degrees(angle)
            self.svg_content.append(f'<g transform="rotate({angle_deg}, {cx}, {cy})">')

        # 레이블
        self.svg_content.append(f'<text x="{cx}" y="{cy - 15}" text-anchor="middle" font-weight="bold" font-size="12">{name}</text>')
        self.svg_content.append(f'<text x="{cx}" y="{cy + 20}" text-anchor="middle" font-size="10">{value}</text>')
        
        if horizontal:
            # 터미널 1
            p1_start = (x - term_len - gap/2, y)
            p1_end = (x - gap/2, y)
            self.draw_wire(p1_start[0], p1_start[1], p1_end[0], p1_end[1])
            # 평판 1
            x1o, y1o = self._apply_off(p1_end[0], y - plate_w/2)
            x2o, y2o = self._apply_off(p1_end[0], y + plate_w/2)
            self.svg_content.append(f'<line x1="{x1o}" y1="{y1o}" x2="{x2o}" y2="{y2o}" stroke="black" stroke-width="1.5"/>')

            # 터미널 2
            p2_start = (x + term_len + gap/2, y)
            p2_end = (x + gap/2, y)
            self.draw_wire(p2_start[0], p2_start[1], p2_end[0], p2_end[1])
            # 평판 2
            x1o, y1o = self._apply_off(p2_end[0], y - plate_w/2)
            x2o, y2o = self._apply_off(p2_end[0], y + plate_w/2)
            self.svg_content.append(f'<line x1="{x1o}" y1="{y1o}" x2="{x2o}" y2="{y2o}" stroke="black" stroke-width="1.5"/>')

            result = p1_start, p2_start
        else: # Vertical
            # 터미널 1
            p1_start = (x, y - term_len - gap/2)
            p1_end = (x, y - gap/2)
            self.draw_wire(p1_start[0], p1_start[1], p1_end[0], p1_end[1])
            # 평판 1
            x1o, y1o = self._apply_off(x - plate_w/2, p1_end[1])
            x2o, y2o = self._apply_off(x + plate_w/2, p1_end[1])
            self.svg_content.append(f'<line x1="{x1o}" y1="{y1o}" x2="{x2o}" y2="{y2o}" stroke="black" stroke-width="1.5"/>')

            # 터미널 2
            p2_start = (x, y + term_len + gap/2)
            p2_end = (x, y + gap/2)
            self.draw_wire(p2_start[0], p2_start[1], p2_end[0], p2_end[1])
            # 평판 2
            x1o, y1o = self._apply_off(x - plate_w/2, p2_end[1])
            x2o, y2o = self._apply_off(x + plate_w/2, p2_end[1])
            self.svg_content.append(f'<line x1="{x1o}" y1="{y1o}" x2="{x2o}" y2="{y2o}" stroke="black" stroke-width="1.5"/>')

            result = p1_start, p2_start

        # 회전 그룹 종료
        if angle != 0:
            self.svg_content.append('</g>')

        return result

    # [수정] 다이오드 좌표 계산 오류, 미정의 변수, 터미널 반환 오류 수정
    def draw_diode(self, x, y, name, value, horizontal=True, angle=0):
        size = 15  # 삼각형 한 변 길이
        term_len = 15 # 터미널 길이

        # 회전 변환
        if angle != 0:
            angle_deg = math.degrees(angle)
            transform = f'transform="rotate({angle_deg}, {x + self.x_offset}, {y + self.y_offset})"'
        else:
            transform = ''

        # 레이블 (회전 적용)
        lx, ly = self._apply_off(x, y)
        self.svg_content.append(f'<text x="{lx}" y="{ly-20}" text-anchor="middle" font-weight="bold" font-size="12" {transform}>{name}</text>')
        self.svg_content.append(f'<text x="{lx}" y="{ly+20}" text-anchor="middle" font-size="10" {transform}>{value}</text>')

        if horizontal:
            # 터미널 (Anode)
            anode_pt = (x - size*0.866/2, y)
            term1 = (anode_pt[0] - term_len, y)
            self.draw_wire(term1[0], term1[1], anode_pt[0], anode_pt[1])

            # 삼각형 몸체 (회전 적용)
            p1 = anode_pt
            p2 = (x + size*0.866/2, y - size/2)
            p3 = (x + size*0.866/2, y + size/2)
            p1_off, p2_off, p3_off = self._apply_off(*p1), self._apply_off(*p2), self._apply_off(*p3)
            path = f"M {p1_off[0]},{p1_off[1]} L {p2_off[0]},{p2_off[1]} L {p3_off[0]},{p3_off[1]} Z"
            self.svg_content.append(f'<path d="{path}" fill="none" stroke="black" stroke-width="1.5" {transform}/>')
            
            # 막대 (Cathode) (회전 적용)
            bar_x = x + size*0.866/2
            bar_x_off, bar_y1_off = self._apply_off(bar_x, y - size/2)
            bar_x_off, bar_y2_off = self._apply_off(bar_x, y + size/2)
            self.svg_content.append(f'<line x1="{bar_x_off}" y1="{bar_y1_off}" x2="{bar_x_off}" y2="{bar_y2_off}" stroke="black" stroke-width="1.5" {transform}/>')
            
            # 터미널 (Cathode)
            term2 = (bar_x + term_len, y)
            self.draw_wire(bar_x, y, term2[0], term2[1])

            return term1, term2
        else: # Vertical
            # 터미널 (Anode)
            anode_pt = (x, y - size*0.866/2)
            term1 = (x, anode_pt[1] - term_len)
            self.draw_wire(term1[0], term1[1], anode_pt[0], anode_pt[1])
            
            # 삼각형 몸체 (회전 적용)
            p1 = anode_pt
            p2 = (x - size/2, y + size*0.866/2)
            p3 = (x + size/2, y + size*0.866/2)
            p1_off, p2_off, p3_off = self._apply_off(*p1), self._apply_off(*p2), self._apply_off(*p3)
            path = f"M {p1_off[0]},{p1_off[1]} L {p2_off[0]},{p2_off[1]} L {p3_off[0]},{p3_off[1]} Z"
            self.svg_content.append(f'<path d="{path}" fill="none" stroke="black" stroke-width="1.5" {transform}/>')
            
            # 막대 (Cathode) (회전 적용)
            bar_y = y + size*0.866/2
            bar_x1_off, bar_y_off = self._apply_off(x - size/2, bar_y)
            bar_x2_off, bar_y_off = self._apply_off(x + size/2, bar_y)
            self.svg_content.append(f'<line x1="{bar_x1_off}" y1="{bar_y_off}" x2="{bar_x2_off}" y2="{bar_y_off}" stroke="black" stroke-width="1.5" {transform}/>')
            
            # 터미널 (Cathode)
            term2 = (x, bar_y + term_len)
            self.draw_wire(x, bar_y, term2[0], term2[1])
            
            return term1, term2

    # [수정] 미구현된 화살표 그리기 기능 추가
    def draw_led(self, x, y, name, value, horizontal=True, angle=0):
        # 다이오드 먼저 그리기
        term1, term2 = self.draw_diode(x, y, name, value, horizontal, angle)

        # 회전 변환
        if angle != 0:
            angle_deg = math.degrees(angle)
            transform = f'transform="rotate({angle_deg}, {x + self.x_offset}, {y + self.y_offset})"'
        else:
            transform = ''

        # LED를 상징하는 화살표 추가 (회전 적용)
        arrow_len = 10
        
        if horizontal:
            ax1, ay1 = x - 5, y - 10
            ax2, ay2 = ax1 - arrow_len, ay1 - arrow_len
            ax1o, ay1o = self._apply_off(ax1, ay1)
            ax2o, ay2o = self._apply_off(ax2, ay2)
            self.svg_content.append(f'<line x1="{ax1o}" y1="{ay1o}" x2="{ax2o}" y2="{ay2o}" stroke="black" stroke-width="1" {transform}/>')
            self.svg_content.append(f'<line x1="{ax2o}" y1="{ay2o}" x2="{ax2o+5}" y2="{ay2o}" stroke="black" stroke-width="1" {transform}/>')
            self.svg_content.append(f'<line x1="{ax2o}" y1="{ay2o}" x2="{ax2o}" y2="{ay2o+5}" stroke="black" stroke-width="1" {transform}/>')
            
            ax1, ay1 = x + 5, y - 10
            ax2, ay2 = ax1 - arrow_len, ay1 - arrow_len
            ax1o, ay1o = self._apply_off(ax1, ay1)
            ax2o, ay2o = self._apply_off(ax2, ay2)
            self.svg_content.append(f'<line x1="{ax1o}" y1="{ay1o}" x2="{ax2o}" y2="{ay2o}" stroke="black" stroke-width="1" {transform}/>')
            self.svg_content.append(f'<line x1="{ax2o}" y1="{ay2o}" x2="{ax2o+5}" y2="{ay2o}" stroke="black" stroke-width="1" {transform}/>')
            self.svg_content.append(f'<line x1="{ax2o}" y1="{ay2o}" x2="{ax2o}" y2="{ay2o+5}" stroke="black" stroke-width="1" {transform}/>')
        else:
            # 수직 방향 화살표 (생략, 필요시 추가 구현)
            pass

        return term1, term2

    def draw_ic(self, x, y, name, pins=8, width=60, height=40, angle=0):
        # 회전 변환
        if angle != 0:
            angle_deg = math.degrees(angle)
            transform = f'transform="rotate({angle_deg}, {x + self.x_offset}, {y + self.y_offset})"'
        else:
            transform = ''

        xo, yo = self._apply_off(x, y)
        self.svg_content.append(f'<rect x="{xo-width/2}" y="{yo-height/2}" width="{width}" height="{height}" fill="white" stroke="black" stroke-width="1.5" {transform}/>')
        
        # 핀 그리기 (회전 적용)
        pin_spacing = height / (pins/2 + 1)
        for i in range(int(pins/2)):
            py = yo - height/2 + (i+1)*pin_spacing
            self.svg_content.append(f'<line x1="{xo-width/2}" y1="{py}" x2="{xo-width/2-10}" y2="{py}" stroke="black" stroke-width="1" {transform}/>')
            self.svg_content.append(f'<line x1="{xo+width/2}" y1="{py}" x2="{xo+width/2+10}" y2="{py}" stroke="black" stroke-width="1" {transform}/>')
        
        # 레이블 (회전 적용)
        self.svg_content.append(f'<text x="{xo}" y="{yo-10}" text-anchor="middle" font-weight="bold" font-size="12" {transform}>{name}</text>')
        return (x-width/2-10, y), (x+width/2+10, y)

    def draw_voltage_source(self, x, y, name, value, angle=0):
        xo, yo = self._apply_off(x, y)
        r = 15
        
        # 회전과 이동을 하나의 transform으로 결합
        if angle != 0:
            angle_deg = math.degrees(angle)
            transform = f'transform="translate({xo},{yo}) rotate({angle_deg})"'
        else:
            transform = f'transform="translate({xo},{yo})"'

        self.svg_content.append(f'<g {transform}>')
        self.svg_content.append(f'<circle cx="0" cy="0" r="{r}" fill="white" stroke="black" stroke-width="1.5"/>')
        self.svg_content.append(f'<line x1="0" y1="{-r+5}" x2="0" y2="{-r+10}" stroke="black" stroke-width="1.5"/>') # + sign vertical
        self.svg_content.append(f'<line x1="{-2.5}" y1="{-r+7.5}" x2="{2.5}" y2="{-r+7.5}" stroke="black" stroke-width="1.5"/>') # + sign horizontal
        self.svg_content.append(f'<line x1="{-2.5}" y1="{r-7.5}" x2="{2.5}" y2="{r-7.5}" stroke="black" stroke-width="1.5"/>') # - sign
        self.svg_content.append(f'<text x="{r+5}" y="4" font-size="10">{value}</text>')
        self.svg_content.append(f'<text x="0" y="{-r-5}" text-anchor="middle" font-weight="bold" font-size="12">{name}</text>')
        self.svg_content.append('</g>')
        term1 = (x, y - r) # Positive terminal
        term2 = (x, y + r) # Negative terminal
        return term1, term2

    def draw_ground(self, x, y):
        xo, yo = self._apply_off(x, y)
        self.svg_content.append(f'<path d="M{xo-10} {yo} l 20 0 M{xo-5} {yo+5} l 10 0 M{xo} {yo+10} l 0 0" stroke="black" stroke-width="1.5"/>')

    def end_svg(self):
        self.svg_content.append('</svg>')

    def get_svg(self):
        return "\n".join(self.svg_content)


import schemdraw
import schemdraw.elements as elm

def convert_spice_to_schemdraw_png(spice_file: str, output_file: str):
    """
    convert_spice_to_svg와 동일한 로직으로 schemdraw를 사용하여 회로도 생성
    """
    parser = SpiceParser(spice_file)
    components = parser.parse()
    layout = CircuitLayout(components, parser.nodes)
    layout.calculate_layout()
    node_pos = layout.node_positions

    if not node_pos:
        print("No nodes to draw.")
        return

    # Drawing 생성
    d = schemdraw.Drawing()
    d.config(unit=2.5, inches_per_unit=0.3, lw=1.5, font='sans-serif')
    
    # 병렬 부품 처리를 위한 그룹화
    parallel_groups = collections.defaultdict(list)
    for comp in components:
        if comp.node1 and comp.node2:
            key = tuple(sorted((comp.node1, comp.node2)))
            parallel_groups[key].append(comp)
    
    # 그려진 컴포넌트와 그 element 저장
    drawn_elements = {}
    
    # 각 병렬 그룹별로 컴포넌트 그리기
    for nodes, group in parallel_groups.items():
        p1 = node_pos[nodes[0]]
        p2 = node_pos[nodes[1]]
        
        num_comps = len(group)
        is_parallel = num_comps > 1
        
        # 병렬 배치를 위한 수직 방향 계산
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.sqrt(dx*dx + dy*dy) if (dx*dx + dy*dy) > 0 else 1
        perp_dx, perp_dy = -dy/dist * 0.8, dx/dist * 0.8  # 병렬 간격
        
        for i, comp in enumerate(group):
            # 병렬 컴포넌트 오프셋 계산
            if is_parallel:
                offset = (i - (num_comps - 1) / 2.0)
                offset_x = offset * perp_dx
                offset_y = offset * perp_dy
            else:
                offset_x = offset_y = 0
            
            # 컴포넌트 중심점
            center_x = (p1[0] + p2[0]) / 2 + offset_x
            center_y = (p1[1] + p2[1]) / 2 + offset_y
            
            # 각도 계산 (도 단위)
            angle = math.degrees(math.atan2(dy, dx))
            
            # 컴포넌트 생성
            label = f'{comp.name}\n{comp.value}' if comp.value else comp.name
            
            if comp.type == 'R':
                elem = elm.Resistor(label=label)
            elif comp.type == 'C':
                elem = elm.Capacitor(label=label)
            elif comp.type == 'L':
                elem = elm.LED(label=label)
            elif comp.type == 'V':
                elem = elm.SourceV(label=label)
            elif comp.type == 'I':
                elem = elm.SourceI(label=label)
            elif comp.type == 'D':
                elem = elm.Diode(label=label)
            else:
                elem = elm.Resistor(label=label)  # 기본값
            
            # schemdraw 좌표계에 맞춰 변환 (픽셀 -> 단위)
            unit_scale = 50  # SVG 픽셀을 schemdraw 단위로 변환
            elem_x = center_x / unit_scale
            elem_y = -center_y / unit_scale  # Y축 반전
            
            # 엘리먼트 배치
            elem = elem.at((elem_x, elem_y)).theta(angle)
            d.add(elem)
            drawn_elements[comp.name] = elem
            
            # 연결선 그리기 (병렬 컴포넌트의 경우)
            if is_parallel:
                # 노드에서 컴포넌트까지 연결선
                start1_x, start1_y = p1[0] / unit_scale, -p1[1] / unit_scale
                start2_x, start2_y = p2[0] / unit_scale, -p2[1] / unit_scale
                
                # 짧은 연결선 추가
                if i == 0:  # 첫 번째 병렬 컴포넌트에만
                    d.add(elm.Line().at((start1_x, start1_y)).to((elem.start)))
                    d.add(elm.Line().at((start2_x, start2_y)).to((elem.end)))
    
    # 노드 표시
    for node, pos in node_pos.items():
        x, y = pos[0] / unit_scale, -pos[1] / unit_scale
        
        # 노드점 표시
        d.add(elm.Dot().at((x, y)))
        
        # 노드 레이블
        if node in parser.ground_nodes:
            # Ground 심볼
            d.add(elm.Ground().at((x, y)))
            d.add(elm.Label(f"Node {node} (GND)").at((x, y-0.5)))
        else:
            d.add(elm.Label(f"Node {node}").at((x, y-0.3)))
    
    # 저장
    try:
        # PNG로 저장
        base = os.path.splitext(output_file)[0]
        png_path = f"{base}.png"
        d.save(png_path)
        
        # SVG도 저장 (선택사항)
        svg_path = f"{base}.svg"
        d.save(svg_path)
        
        print(f"Schemdraw circuit saved to {png_path} and {svg_path}")
        
    except Exception as e:
        print(f"Error saving schematic: {e}")
# [IMPROVED] SVG 생성 로직을 더 견고하게 수정
def calculate_rotation_angle(p1, p2):
    """두 점 사이의 벡터 각도를 계산하여 소자를 연결선과 평행하게 회전"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    # 연결선의 각도를 계산
    angle = math.atan2(dy, dx)
    
    # 각도를 -π/2 ~ π/2 범위로 정규화 (소자가 위아래로 뒤집히지 않도록)
    # 이렇게 하면 연속적인 회전이 가능
    if angle > math.pi/2:
        angle = angle - math.pi
    elif angle < -math.pi/2:
        angle = angle + math.pi
    
    return angle

def rotate_point(x, y, cx, cy, angle):
    """점 (x,y)를 중심점 (cx,cy)를 기준으로 angle만큼 회전"""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    dx = x - cx
    dy = y - cy
    return cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a

def convert_spice_to_svg(spice_file: str, output_file: str):
    parser = SpiceParser(spice_file)
    components = parser.parse()
    layout = CircuitLayout(components, parser.nodes)
    layout.calculate_layout()
    node_pos = layout.node_positions

    if not node_pos:
        print("No nodes to draw.")
        return

    # Bounding box 계산
    all_x = [pos[0] for pos in node_pos.values()]
    all_y = [pos[1] for pos in node_pos.values()]
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    margin = 100
    width = (max_x - min_x) + 2 * margin
    height = (max_y - min_y) + 2 * margin
    offset_x = margin - min_x
    offset_y = margin - min_y
    svg = SVGGenerator(width=width, height=height, x_offset=offset_x, y_offset=offset_y)
    svg.start_svg()

    # 병렬 부품 처리 로직
    parallel_groups = collections.defaultdict(list)
    for comp in components:
        if comp.node1 and comp.node2:
            key = tuple(sorted((comp.node1, comp.node2)))
            parallel_groups[key].append(comp)

    for nodes, group in parallel_groups.items():
        p1 = node_pos[nodes[0]]
        p2 = node_pos[nodes[1]]
        
        num_comps = len(group)
        is_parallel = num_comps > 1
        
        # 두 노드 간의 실제 각도 계산
        connection_angle = calculate_rotation_angle(p1, p2)
        
        # 병렬 부품을 배치할 오프셋 방향 계산
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.sqrt(dx*dx + dy*dy) if (dx*dx + dy*dy) > 0 else 1
        perp_dx, perp_dy = -dy/dist, dx/dist
        spacing = 30  # 병렬 부품 간 간격

        for i, comp in enumerate(group):
            # 중심점 계산
            center_x, center_y = (p1[0]+p2[0])/2, (p1[1]+p2[1])/2
            
            # 병렬일 경우 오프셋 적용
            if is_parallel:
                offset = i - (num_comps - 1) / 2.0
                comp_x = center_x + offset * perp_dx * spacing
                comp_y = center_y + offset * perp_dy * spacing
            else:
                comp_x, comp_y = center_x, center_y

            # 수정된 draw 함수들을 사용
            term1, term2 = None, None
            
            if comp.type == 'R':
                term1, term2 = draw_resistor_fixed(svg, comp_x, comp_y, comp.name, comp.value, connection_angle)
            elif comp.type == 'C':
                term1, term2 = draw_capacitor_fixed(svg, comp_x, comp_y, comp.name, comp.value, connection_angle)
            elif comp.type == 'L':
                term1, term2 = draw_led_fixed(svg, comp_x, comp_y, comp.name, comp.value, connection_angle)
            elif comp.type == 'D':
                term1, term2 = draw_diode_fixed(svg, comp_x, comp_y, comp.name, comp.value, connection_angle)
            elif comp.type in ('U','X'):
                term1, term2 = draw_ic_fixed(svg, comp_x, comp_y, comp.name, connection_angle)
            elif comp.type == 'V':
                term1, term2 = draw_voltage_source_fixed(svg, comp_x, comp_y, comp.name, comp.value, connection_angle)
            
            # 와이어 연결
            if term1 and term2:
                # 전압원의 경우 극성을 고려한 연결
                if comp.type == 'V':
                    # 전압원: node1이 positive (+), node2가 negative (-)
                    # term1은 positive 터미널, term2는 negative 터미널
                    node1_pos = node_pos[comp.node1]
                    node2_pos = node_pos[comp.node2]
                    
                    # 각 터미널에서 가장 가까운 노드로 연결
                    dist_n1_t1 = math.hypot(node1_pos[0]-term1[0], node1_pos[1]-term1[1])
                    dist_n1_t2 = math.hypot(node1_pos[0]-term2[0], node1_pos[1]-term2[1])
                    
                    if dist_n1_t1 < dist_n1_t2:
                        # term1이 node1에 가까움 - 올바른 연결
                        svg.draw_wire(node1_pos[0], node1_pos[1], term1[0], term1[1])
                        svg.draw_wire(node2_pos[0], node2_pos[1], term2[0], term2[1])
                    else:
                        # term2가 node1에 가까움 - 반대로 연결
                        svg.draw_wire(node1_pos[0], node1_pos[1], term2[0], term2[1])
                        svg.draw_wire(node2_pos[0], node2_pos[1], term1[0], term1[1])
                elif is_parallel:
                    # 병렬 컴포넌트의 경우 원래 노드 위치에서 컴포넌트 터미널까지 연결
                    # 각 터미널에서 가장 가까운 노드로 연결
                    dist_p1_t1 = math.hypot(p1[0]-term1[0], p1[1]-term1[1])
                    dist_p1_t2 = math.hypot(p1[0]-term2[0], p1[1]-term2[1])
                    if dist_p1_t1 < dist_p1_t2:
                        svg.draw_wire(p1[0], p1[1], term1[0], term1[1])
                        svg.draw_wire(p2[0], p2[1], term2[0], term2[1])
                    else:
                        svg.draw_wire(p1[0], p1[1], term2[0], term2[1])
                        svg.draw_wire(p2[0], p2[1], term1[0], term1[1])
                else:
                    # 단일 컴포넌트는 직접 연결
                    svg.draw_wire(p1[0], p1[1], term1[0], term1[1])
                    svg.draw_wire(p2[0], p2[1], term2[0], term2[1])

    # 노드 그리기
    for node, pos in node_pos.items():
        gnd = node in parser.ground_nodes
        label = f"Node {node}" + (" (GND)" if gnd else "")
        svg.draw_node_label(pos[0], pos[1], label)
        svg.draw_node_dot(pos[0], pos[1])
        if gnd:
            svg.draw_ground(pos[0], pos[1]+5)

    svg.end_svg()

    # 저장
    with open(output_file, 'w') as f:
        f.write(svg.get_svg())
    
    try:
        png_file = os.path.splitext(output_file)[0] + '.png'
        cairosvg.svg2png(url=output_file, write_to=png_file, background_color='white')
        print(f"Custom SVG schematic saved to {output_file} and {png_file}")
    except Exception as e:
        print(f"Failed to save PNG: {e}")


# 수정된 draw 함수들
def draw_resistor_fixed(svg, x, y, name, value, angle):
    """회전이 올바르게 적용되는 저항 그리기 (텍스트는 수평 유지)"""
    # 기본 파라미터
    segments = 6
    length = 40
    zigzag_amp = 10
    term_len = 10
    
    # 터미널 위치 계산 (회전 적용)
    half_len = (length + 2 * term_len) / 2
    term1 = rotate_point(x - half_len, y, x, y, angle)
    term2 = rotate_point(x + half_len, y, x, y, angle)
    
    # SVG 그룹으로 회전 적용 (도형만)
    angle_deg = math.degrees(angle)
    cx, cy = x + svg.x_offset, y + svg.y_offset
    
    svg.svg_content.append(f'<g transform="rotate({angle_deg}, {cx}, {cy})">')
    
    # 지그재그 그리기 (수평 기준)
    x0 = x - length/2
    dx = length / segments
    
    # 터미널 1
    svg.draw_wire(x - length/2 - term_len, y, x - length/2, y)
    
    # 지그재그
    path_points = [(x0, y)]
    for i in range(1, segments):
        px = x0 + i * dx
        py = y + (zigzag_amp if i % 2 else -zigzag_amp)
        path_points.append((px, py))
    path_points.append((x0 + length, y))
    
    path = 'M ' + ' L '.join(f"{px + svg.x_offset},{py + svg.y_offset}" for px, py in path_points)
    svg.svg_content.append(f'<path d="{path}" stroke="black" fill="none" stroke-width="1.5"/>')
    
    # 터미널 2
    svg.draw_wire(x + length/2, y, x + length/2 + term_len, y)
    
    svg.svg_content.append('</g>')
    
    # 텍스트는 회전 그룹 밖에서 그리기 (수평 유지)
    svg.svg_content.append(f'<text x="{cx}" y="{cy - 15}" text-anchor="middle" font-weight="bold" font-size="12">{name}</text>')
    svg.svg_content.append(f'<text x="{cx}" y="{cy + 15}" text-anchor="middle" font-size="10">{value}</text>')
    
    return term1, term2

def draw_capacitor_fixed(svg, x, y, name, value, angle):
    """회전이 올바르게 적용되는 커패시터 그리기 (텍스트는 수평 유지)"""
    plate_h = 20
    gap = 6
    term_len = 20
    
    # 터미널 위치 계산 (회전 적용)
    half_len = (gap + 2 * term_len) / 2
    term1 = rotate_point(x - half_len, y, x, y, angle)
    term2 = rotate_point(x + half_len, y, x, y, angle)
    
    # SVG 그룹으로 회전 적용 (도형만)
    angle_deg = math.degrees(angle)
    cx, cy = x + svg.x_offset, y + svg.y_offset
    
    svg.svg_content.append(f'<g transform="rotate({angle_deg}, {cx}, {cy})">')
    
    # 터미널 1
    svg.draw_wire(x - gap/2 - term_len, y, x - gap/2, y)
    # 평판 1
    x1o, y1o = svg._apply_off(x - gap/2, y - plate_h/2)
    x2o, y2o = svg._apply_off(x - gap/2, y + plate_h/2)
    svg.svg_content.append(f'<line x1="{x1o}" y1="{y1o}" x2="{x2o}" y2="{y2o}" stroke="black" stroke-width="1.5"/>')
    
    # 터미널 2
    svg.draw_wire(x + gap/2, y, x + gap/2 + term_len, y)
    # 평판 2
    x1o, y1o = svg._apply_off(x + gap/2, y - plate_h/2)
    x2o, y2o = svg._apply_off(x + gap/2, y + plate_h/2)
    svg.svg_content.append(f'<line x1="{x1o}" y1="{y1o}" x2="{x2o}" y2="{y2o}" stroke="black" stroke-width="1.5"/>')
    
    svg.svg_content.append('</g>')
    
    # 텍스트는 회전 그룹 밖에서 그리기 (수평 유지)
    svg.svg_content.append(f'<text x="{cx}" y="{cy - 15}" text-anchor="middle" font-weight="bold" font-size="12">{name}</text>')
    svg.svg_content.append(f'<text x="{cx}" y="{cy + 20}" text-anchor="middle" font-size="10">{value}</text>')
    
    return term1, term2

def draw_diode_fixed(svg, x, y, name, value, angle):
    """회전이 올바르게 적용되는 다이오드 그리기 (텍스트는 수평 유지)"""
    size = 15
    term_len = 15
    total_len = size * 0.866 + 2 * term_len
    
    # 터미널 위치 계산 (회전 적용)
    term1 = rotate_point(x - total_len/2, y, x, y, angle)
    term2 = rotate_point(x + total_len/2, y, x, y, angle)
    
    # SVG 그룹으로 회전 적용 (도형만)
    angle_deg = math.degrees(angle)
    cx, cy = x + svg.x_offset, y + svg.y_offset
    
    svg.svg_content.append(f'<g transform="rotate({angle_deg}, {cx}, {cy})">')
    
    # 터미널 (Anode)
    svg.draw_wire(x - size*0.866/2 - term_len, y, x - size*0.866/2, y)
    
    # 삼각형
    p1 = (x - size*0.866/2, y)
    p2 = (x + size*0.866/2, y - size/2)
    p3 = (x + size*0.866/2, y + size/2)
    p1_off, p2_off, p3_off = svg._apply_off(*p1), svg._apply_off(*p2), svg._apply_off(*p3)
    path = f"M {p1_off[0]},{p1_off[1]} L {p2_off[0]},{p2_off[1]} L {p3_off[0]},{p3_off[1]} Z"
    svg.svg_content.append(f'<path d="{path}" fill="none" stroke="black" stroke-width="1.5"/>')
    
    # 막대 (Cathode)
    bar_x = x + size*0.866/2
    bar_x_off, bar_y1_off = svg._apply_off(bar_x, y - size/2)
    bar_x_off, bar_y2_off = svg._apply_off(bar_x, y + size/2)
    svg.svg_content.append(f'<line x1="{bar_x_off}" y1="{bar_y1_off}" x2="{bar_x_off}" y2="{bar_y2_off}" stroke="black" stroke-width="1.5"/>')
    
    # 터미널 (Cathode)
    svg.draw_wire(bar_x, y, bar_x + term_len, y)
    
    svg.svg_content.append('</g>')
    
    # 텍스트는 회전 그룹 밖에서 그리기 (수평 유지)
    svg.svg_content.append(f'<text x="{cx}" y="{cy-20}" text-anchor="middle" font-weight="bold" font-size="12">{name}</text>')
    svg.svg_content.append(f'<text x="{cx}" y="{cy+20}" text-anchor="middle" font-size="10">{value}</text>')
    
    return term1, term2

def draw_led_fixed(svg, x, y, name, value, angle):
    """회전이 올바르게 적용되는 LED 그리기 (텍스트는 수평 유지)"""
    # 먼저 다이오드를 그림 (텍스트는 나중에 그림)
    size = 15
    term_len = 15
    total_len = size * 0.866 + 2 * term_len
    
    # 터미널 위치 계산 (회전 적용)
    term1 = rotate_point(x - total_len/2, y, x, y, angle)
    term2 = rotate_point(x + total_len/2, y, x, y, angle)
    
    # SVG 그룹으로 회전 적용 (도형만)
    angle_deg = math.degrees(angle)
    cx, cy = x + svg.x_offset, y + svg.y_offset
    
    svg.svg_content.append(f'<g transform="rotate({angle_deg}, {cx}, {cy})">')
    
    # 다이오드 본체
    # 터미널 (Anode)
    svg.draw_wire(x - size*0.866/2 - term_len, y, x - size*0.866/2, y)
    
    # 삼각형
    p1 = (x - size*0.866/2, y)
    p2 = (x + size*0.866/2, y - size/2)
    p3 = (x + size*0.866/2, y + size/2)
    p1_off, p2_off, p3_off = svg._apply_off(*p1), svg._apply_off(*p2), svg._apply_off(*p3)
    path = f"M {p1_off[0]},{p1_off[1]} L {p2_off[0]},{p2_off[1]} L {p3_off[0]},{p3_off[1]} Z"
    svg.svg_content.append(f'<path d="{path}" fill="none" stroke="black" stroke-width="1.5"/>')
    
    # 막대 (Cathode)
    bar_x = x + size*0.866/2
    bar_x_off, bar_y1_off = svg._apply_off(bar_x, y - size/2)
    bar_x_off, bar_y2_off = svg._apply_off(bar_x, y + size/2)
    svg.svg_content.append(f'<line x1="{bar_x_off}" y1="{bar_y1_off}" x2="{bar_x_off}" y2="{bar_y2_off}" stroke="black" stroke-width="1.5"/>')
    
    # 터미널 (Cathode)
    svg.draw_wire(bar_x, y, bar_x + term_len, y)
    
    # LED 화살표
    # 화살표 1
    ax1, ay1 = x - 5, y - 15
    ax2, ay2 = ax1 - 8, ay1 - 8
    ax1o, ay1o = svg._apply_off(ax1, ay1)
    ax2o, ay2o = svg._apply_off(ax2, ay2)
    svg.svg_content.append(f'<line x1="{ax1o}" y1="{ay1o}" x2="{ax2o}" y2="{ay2o}" stroke="black" stroke-width="1"/>')
    svg.svg_content.append(f'<line x1="{ax2o}" y1="{ay2o}" x2="{ax2o+4}" y2="{ay2o}" stroke="black" stroke-width="1"/>')
    svg.svg_content.append(f'<line x1="{ax2o}" y1="{ay2o}" x2="{ax2o}" y2="{ay2o+4}" stroke="black" stroke-width="1"/>')
    
    # 화살표 2
    ax1, ay1 = x + 5, y - 15
    ax2, ay2 = ax1 - 8, ay1 - 8
    ax1o, ay1o = svg._apply_off(ax1, ay1)
    ax2o, ay2o = svg._apply_off(ax2, ay2)
    svg.svg_content.append(f'<line x1="{ax1o}" y1="{ay1o}" x2="{ax2o}" y2="{ay2o}" stroke="black" stroke-width="1"/>')
    svg.svg_content.append(f'<line x1="{ax2o}" y1="{ay2o}" x2="{ax2o+4}" y2="{ay2o}" stroke="black" stroke-width="1"/>')
    svg.svg_content.append(f'<line x1="{ax2o}" y1="{ay2o}" x2="{ax2o}" y2="{ay2o+4}" stroke="black" stroke-width="1"/>')
    
    svg.svg_content.append('</g>')
    
    # 텍스트는 회전 그룹 밖에서 그리기 (수평 유지)
    svg.svg_content.append(f'<text x="{cx}" y="{cy-25}" text-anchor="middle" font-weight="bold" font-size="12">{name}</text>')
    svg.svg_content.append(f'<text x="{cx}" y="{cy+20}" text-anchor="middle" font-size="10">{value}</text>')
    
    return term1, term2

def draw_ic_fixed(svg, x, y, name, angle, pins=8, width=60, height=40):
    """회전이 올바르게 적용되는 IC 그리기 (텍스트는 수평 유지)"""
    # 좌우 터미널 위치 (단순화)
    term1 = rotate_point(x - width/2 - 10, y, x, y, angle)
    term2 = rotate_point(x + width/2 + 10, y, x, y, angle)
    
    angle_deg = math.degrees(angle)
    cx, cy = x + svg.x_offset, y + svg.y_offset
    
    svg.svg_content.append(f'<g transform="rotate({angle_deg}, {cx}, {cy})">')
    
    xo, yo = svg._apply_off(x, y)
    svg.svg_content.append(f'<rect x="{xo-width/2}" y="{yo-height/2}" width="{width}" height="{height}" fill="white" stroke="black" stroke-width="1.5"/>')
    
    # 핀 그리기
    pin_spacing = height / (pins/2 + 1)
    for i in range(int(pins/2)):
        py = yo - height/2 + (i+1)*pin_spacing
        svg.svg_content.append(f'<line x1="{xo-width/2}" y1="{py}" x2="{xo-width/2-10}" y2="{py}" stroke="black" stroke-width="1"/>')
        svg.svg_content.append(f'<line x1="{xo+width/2}" y1="{py}" x2="{xo+width/2+10}" y2="{py}" stroke="black" stroke-width="1"/>')
    
    svg.svg_content.append('</g>')
    
    # 텍스트는 회전 그룹 밖에서 그리기 (수평 유지)
    svg.svg_content.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" font-weight="bold" font-size="12">{name}</text>')
    
    return term1, term2

def draw_voltage_source_fixed(svg, x, y, name, value, angle):
    """회전이 올바르게 적용되는 전압원 그리기"""
    r = 15
    term_len = 10
    
    # 전압원은 수평 방향을 기본으로 하여 회전
    # 터미널 위치 계산 (좌우 터미널로 변경)
    half_len = r + term_len
    term1 = rotate_point(x - half_len, y, x, y, angle)  # 왼쪽 터미널 (positive)
    term2 = rotate_point(x + half_len, y, x, y, angle)  # 오른쪽 터미널 (negative)
    
    angle_deg = math.degrees(angle)
    xo, yo = svg._apply_off(x, y)
    
    svg.svg_content.append(f'<g transform="rotate({angle_deg}, {xo}, {yo})">')
    
    # 터미널 연결선 (수평)
    svg.draw_wire(x - r - term_len, y, x - r, y)
    svg.draw_wire(x + r, y, x + r + term_len, y)
    
    # 원
    svg.svg_content.append(f'<circle cx="{xo}" cy="{yo}" r="{r}" fill="white" stroke="black" stroke-width="1.5"/>')
    
    # + 기호 (왼쪽)
    svg.svg_content.append(f'<line x1="{xo-7}" y1="{yo}" x2="{xo-2}" y2="{yo}" stroke="black" stroke-width="1.5"/>')
    svg.svg_content.append(f'<line x1="{xo-4.5}" y1="{yo-2.5}" x2="{xo-4.5}" y2="{yo+2.5}" stroke="black" stroke-width="1.5"/>')
    
    # - 기호 (오른쪽)
    svg.svg_content.append(f'<line x1="{xo+2}" y1="{yo}" x2="{xo+7}" y2="{yo}" stroke="black" stroke-width="1.5"/>')
    
    # 레이블
    svg.svg_content.append(f'<text x="{xo}" y="{yo-r-5}" text-anchor="middle" font-weight="bold" font-size="12">{name}</text>')
    svg.svg_content.append(f'<text x="{xo}" y="{yo+r+12}" text-anchor="middle" font-size="10">{value}</text>')
    
    svg.svg_content.append('</g>')
    
    return term1, term2


# [MAJOR REFACTOR] schemdraw를 제대로 활용하는 방식으로 전면 재작성
def convert_spice_to_schemdraw(spice_file: str, output_file: str):
    """
    convert_spice_to_svg와 동일한 로직으로 schemdraw를 사용하여 회로도 생성
    """
    parser = SpiceParser(spice_file)
    components = parser.parse()
    layout = CircuitLayout(components, parser.nodes)
    layout.calculate_layout()
    node_pos = layout.node_positions

    if not node_pos:
        print("No nodes to draw.")
        return

    # Drawing 생성
    d = schemdraw.Drawing()
    d.config(unit=2.5, inches_per_unit=0.3, lw=1.5, font='sans-serif')
    
    # 병렬 부품 처리를 위한 그룹화
    parallel_groups = collections.defaultdict(list)
    for comp in components:
        if comp.node1 and comp.node2:
            key = tuple(sorted((comp.node1, comp.node2)))
            parallel_groups[key].append(comp)
    
    # 그려진 컴포넌트와 그 element 저장
    drawn_elements = {}
    
    # 각 병렬 그룹별로 컴포넌트 그리기
    for nodes, group in parallel_groups.items():
        p1 = node_pos[nodes[0]]
        p2 = node_pos[nodes[1]]
        
        num_comps = len(group)
        is_parallel = num_comps > 1
        
        # 병렬 배치를 위한 수직 방향 계산
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = math.sqrt(dx*dx + dy*dy) if (dx*dx + dy*dy) > 0 else 1
        perp_dx, perp_dy = -dy/dist * 0.8, dx/dist * 0.8  # 병렬 간격
        
        for i, comp in enumerate(group):
            # 병렬 컴포넌트 오프셋 계산
            if is_parallel:
                offset = (i - (num_comps - 1) / 2.0)
                offset_x = offset * perp_dx
                offset_y = offset * perp_dy
            else:
                offset_x = offset_y = 0
            
            # 컴포넌트 중심점
            center_x = (p1[0] + p2[0]) / 2 + offset_x
            center_y = (p1[1] + p2[1]) / 2 + offset_y
            
            # 각도 계산 (도 단위)
            angle = math.degrees(math.atan2(dy, dx))
            
            # 컴포넌트 생성
            label = f'{comp.name}\n{comp.value}' if comp.value else comp.name
            
            if comp.type == 'R':
                elem = elm.Resistor(label=label)
            elif comp.type == 'C':
                elem = elm.Capacitor(label=label)
            elif comp.type == 'L':
                elem = elm.LED(label=label)
            elif comp.type == 'V':
                elem = elm.SourceV(label=label)
            elif comp.type == 'I':
                elem = elm.SourceI(label=label)
            elif comp.type == 'D':
                elem = elm.Diode(label=label)
            else:
                elem = elm.Resistor(label=label)  # 기본값
            
            # schemdraw 좌표계에 맞춰 변환 (픽셀 -> 단위)
            unit_scale = 50  # SVG 픽셀을 schemdraw 단위로 변환
            elem_x = center_x / unit_scale
            elem_y = -center_y / unit_scale  # Y축 반전
            
            # 엘리먼트 배치
            elem = elem.at((elem_x, elem_y)).theta(angle)
            d.add(elem)
            drawn_elements[comp.name] = elem
            
            # 연결선 그리기 (병렬 컴포넌트의 경우)
            if is_parallel:
                # 노드에서 컴포넌트까지 연결선
                start1_x, start1_y = p1[0] / unit_scale, -p1[1] / unit_scale
                start2_x, start2_y = p2[0] / unit_scale, -p2[1] / unit_scale
                
                # 짧은 연결선 추가
                if i == 0:  # 첫 번째 병렬 컴포넌트에만
                    d.add(elm.Line().at((start1_x, start1_y)).to((elem.start)))
                    d.add(elm.Line().at((start2_x, start2_y)).to((elem.end)))
    
    # 노드 표시
    for node, pos in node_pos.items():
        x, y = pos[0] / unit_scale, -pos[1] / unit_scale
        
        # 노드점 표시
        d.add(elm.Dot().at((x, y)))
        
        # 노드 레이블
        if node in parser.ground_nodes:
            # Ground 심볼
            d.add(elm.Ground().at((x, y)))
            d.add(elm.Label(f"Node {node} (GND)").at((x, y-0.5)))
        else:
            d.add(elm.Label(f"Node {node}").at((x, y-0.3)))
    
    # 저장
    try:
        # PNG로 저장
        base = os.path.splitext(output_file)[0]
        png_path = f"{base}.png"
        d.save(png_path)
        
        # SVG도 저장 (선택사항)
        svg_path = f"{base}.svg"
        d.save(svg_path)
        
        print(f"Schemdraw circuit saved to {png_path} and {svg_path}")
        
    except Exception as e:
        print(f"Error saving schematic: {e}")


def convert_spice_to_schemdraw_auto_png(spice_file: str, output_file: str):
    """
    개선된 schemdraw 자동 레이아웃 PNG 생성
    - push/pop을 활용한 안정적인 위치 관리
    - 충분한 간격으로 겹침 방지
    """
    parser = SpiceParser(spice_file)
    components = parser.parse()
    
    if not components:
        print("No components found.")
        return
    
    # 병렬 그룹 찾기
    parallel_groups = collections.defaultdict(list)
    for comp in components:
        if len(comp.nodes) >= 2:
            key = tuple(sorted([comp.node1, comp.node2]))
            parallel_groups[key].append(comp)
    
    with schemdraw.Drawing() as d:
        d.config(unit=3.5, inches_per_unit=0.4, lw=1.5, font='sans-serif', fontsize=10)
        
        # Ground 노드 찾기
        ground_node = next(iter(parser.ground_nodes), '0') if parser.ground_nodes else None
        
        # 시작점 설정
        if ground_node:
            gnd = d.add(elm.Ground())
            d.push()  # 현재 위치 저장
            d.add(elm.Dot())
            start_pos = d.here
        else:
            d.add(elm.Dot())
            d.push()
            start_pos = d.here
        
        # 그려진 컴포넌트 추적
        drawn_components = set()
        
        # 병렬 그룹별로 처리
        group_count = 0
        for node_pair, comps in parallel_groups.items():
            if any(c.name in drawn_components for c in comps):
                continue
            
            # 새 그룹 시작 위치
            if group_count > 0:
                # 이전 그룹과 간격 두기
                d.pop()  # 시작점으로
                d.push()
                if group_count % 3 == 0:  # 3개마다 새 행
                    d.move(0, -4)
                else:
                    d.move(5, 0)
            
            # 병렬 컴포넌트 그리기
            if len(comps) > 1:
                # 병렬 회로 그리기
                d.push()  # 분기 시작점 저장
                
                # 노드 1
                d.add(elm.Dot().label(f'N{node_pair[0]}', loc='left'))
                
                # 병렬 경로들
                max_width = 0
                for i, comp in enumerate(comps):
                    d.pop()  # 분기점으로 돌아가기
                    d.push()  # 다시 저장
                    
                    # 위아래로 분산
                    offset = (i - (len(comps)-1)/2) * 1.2
                    if offset != 0:
                        d.add(elm.Line().length(0.3).theta(90 if offset > 0 else -90))
                        d.add(elm.Line().length(abs(offset)).theta(90 if offset > 0 else -90))
                        d.add(elm.Line().length(0.3).theta(0))
                    
                    # 컴포넌트
                    label = f'{comp.name}\n{comp.value}' if comp.value else comp.name
                    elem = create_schemdraw_element(comp, label)
                    el = d.add(elem.right().length(3))
                    
                    # 오른쪽 연결
                    if offset != 0:
                        d.add(elm.Line().length(0.3).theta(0))
                        d.add(elm.Line().length(abs(offset)).theta(-90 if offset > 0 else 90))
                        d.add(elm.Line().length(0.3).theta(-90 if offset > 0 else 90))
                    
                    # 최대 너비 추적
                    if hasattr(el, 'end'):
                        end_x = el.end[0] if hasattr(el.end, '__getitem__') else 0
                        max_width = max(max_width, end_x)
                    
                    drawn_components.add(comp.name)
                
                # 병합점으로 이동
                d.pop()  # 분기점으로
                if max_width > 0:
                    d.move(max_width + 1, 0)
                else:
                    d.move(4, 0)
                    
                # 노드 2
                d.add(elm.Dot().label(f'N{node_pair[1]}', loc='right'))
                
            else:
                # 단일 컴포넌트
                comp = comps[0]
                
                # 노드 1
                d.add(elm.Dot().label(f'N{comp.node1}', loc='bottom'))
                
                # 컴포넌트
                label = f'{comp.name}\n{comp.value}' if comp.value else comp.name
                elem = create_schemdraw_element(comp, label)
                d.add(elem.right().length(3))
                
                # 노드 2
                d.add(elm.Dot().label(f'N{comp.node2}', loc='bottom'))
                
                drawn_components.add(comp.name)
            
            group_count += 1
    
    # 저장
    base = os.path.splitext(output_file)[0]
    try:
        d.save(f"{base}.png", dpi=150)
        d.save(f"{base}.svg")
        print(f"Circuit saved to {base}.png and {base}.svg")
    except Exception as e:
        print(f"Error saving circuit: {e}")


def convert_spice_to_schemdraw_simple_png(spice_file: str, output_file: str):
    """
    더 간단한 접근법 - 선형 배치
    """
    parser = SpiceParser(spice_file)
    components = parser.parse()
    
    if not components:
        print("No components found.")
        return
    
    # 병렬 그룹 찾기
    parallel_groups = collections.defaultdict(list)
    for comp in components:
        if len(comp.nodes) >= 2:
            key = tuple(sorted([comp.node1, comp.node2]))
            parallel_groups[key].append(comp)
    
    d = schemdraw.Drawing()
    d.config(unit=2.5, inches_per_unit=0.5, lw=1.5)
    
    # Ground 시작
    if parser.ground_nodes:
        d += elm.Ground()
        d += elm.Dot()
    
    # 각 그룹을 순차적으로 그리기
    for i, (node_pair, comps) in enumerate(parallel_groups.items()):
        if i > 0:
            d += elm.Line().right(0.5)
            d += elm.Dot()
        
        if len(comps) == 1:
            # 단일 컴포넌트
            comp = comps[0]
            label = f'{comp.name}\n{comp.value}' if comp.value else comp.name
            elem = create_schemdraw_element(comp, label)
            d += elem.right()
            
        else:
            # 병렬 컴포넌트 - 단순 표기
            d.push()
            
            # 첫 번째 컴포넌트 (위)
            d += elm.Line().up(0.5)
            comp = comps[0]
            label = f'{comp.name}\n{comp.value}' if comp.value else comp.name
            elem = create_schemdraw_element(comp, label)
            d += elem.right()
            d += elm.Line().down(0.5)
            
            # 나머지는 아래에
            d.pop()
            if len(comps) > 1:
                d += elm.Line().down(0.5)
                # 병렬 표시
                d += elm.Label(f'|| {len(comps)-1} more').right()
                d += elm.Line().up(0.5)
        
        d += elm.Dot()
    
    # 저장
    base = os.path.splitext(output_file)[0]
    d.save(f"{base}.png", dpi=150)
    d.save(f"{base}.svg")
    print(f"Simple circuit saved to {base}.png and {base}.svg")


def create_schemdraw_element(comp, label):
    """컴포넌트 타입에 따른 schemdraw element 생성"""
    elements = {
        'R': elm.Resistor,
        'C': elm.Capacitor,
        'L': elm.LED,
        'V': elm.SourceV,
        'I': elm.SourceI,
        'D': elm.Diode,
        'Q': elm.Bjt,

        
    }

    
    # 기본 엘리먼트 선택
    ElementClass = elements.get(comp.type, elm.Resistor)
    return ElementClass(label=label)

def convert_spice_to_schemdraw_grid_png(spice_file: str, output_file: str):
    """
    그리드 기반 레이아웃 - 더 체계적인 배치
    """
    parser = SpiceParser(spice_file)
    components = parser.parse()
    
    if not components:
        print("No components found.")
        return
    
    # 병렬 그룹화
    parallel_groups = collections.defaultdict(list)
    for comp in components:
        if len(comp.nodes) >= 2:
            key = tuple(sorted([comp.node1, comp.node2]))
            parallel_groups[key].append(comp)
    
    with schemdraw.Drawing() as d:
        d.config(unit=3.0, inches_per_unit=0.5, lw=1.5)
        
        # Ground 추가
        if parser.ground_nodes:
            d.add(elm.Ground())
            d.move(0, 1)
        
        # 그리드 레이아웃
        col = 0
        row = 0
        max_cols = 4  # 한 행에 최대 4개 컴포넌트 그룹
        
        for node_pair, comps in parallel_groups.items():
            # 시작 위치로 이동
            d.move_from((col * 4, -row * 3), dx=0, dy=0)
            
            if len(comps) == 1:
                # 단일 컴포넌트
                comp = comps[0]
                label = f'{comp.name}\n{comp.value}' if comp.value else comp.name
                elem = create_schemdraw_element(comp, label)
                
                # 수평 또는 수직 배치
                if col % 2 == 0:
                    d.add(elem.right(length=3))
                else:
                    d.add(elem.down(length=2))
                
            else:
                # 병렬 컴포넌트
                # 분기점
                start = d.here
                d.add(elm.Dot())
                
                # 각 병렬 컴포넌트
                for i, comp in enumerate(comps):
                    d.move_from(start, dx=0, dy=0)
                    
                    # 위아래로 분기
                    offset = (i - (len(comps)-1)/2) * 1.5
                    if offset != 0:
                        d.add(elm.Line().up(length=abs(offset)) if offset > 0 
                              else elm.Line().down(length=abs(offset)))
                    
                    # 컴포넌트
                    label = f'{comp.name}\n{comp.value}' if comp.value else comp.name
                    elem = create_schemdraw_element(comp, label)
                    d.add(elem.right(length=3))
                    
                    # 다시 합치기
                    end_x = d.here[0]
                    if offset != 0:
                        d.add(elm.Line().down(length=abs(offset)) if offset > 0 
                              else elm.Line().up(length=abs(offset)))
                
                # 끝점
                d.move_from((end_x, start[1]), dx=0, dy=0)
                d.add(elm.Dot())
            
            # 다음 위치로
            col += 1
            if col >= max_cols:
                col = 0
                row += 1
        
        # 노드 레이블 추가
        d.move(0, -row * 3 - 2)
        d.add(elm.Label('Nodes: ' + ', '.join(f'N{n}' for n in sorted(parser.nodes))))
    
    # 저장
    base = os.path.splitext(output_file)[0]
    d.save(f"{base}.png", dpi=150)
    d.save(f"{base}.svg")
    print(f"Grid-based circuit saved to {base}.png and {base}.svg")


def convert_spice_to_schemdraw_practical(spice_file: str, output_file: str):
    """
    실용적인 하이브리드 방식: 메인 경로 + 브랜치 처리
    """
    parser = SpiceParser(spice_file)
    components = parser.parse()
    if not components:
        print(f"No components found in {spice_file}.")
        return
    
    # 회로 분석: 직렬/병렬 구조 파악
    circuit_tree = analyze_circuit_topology(components, parser.nodes, parser.ground_nodes)
    
    with schemdraw.Drawing() as d:
        d.config(unit=2.5, inches_per_unit=0.5, lw=1.5)
        
        # Ground부터 시작
        if parser.ground_nodes:
            gnd = d.add(elm.Ground())
            d.add(elm.Dot())
        
        # 메인 경로 그리기
        draw_main_path(d, circuit_tree)
        
        # 브랜치 추가
        draw_branches(d, circuit_tree)
        
        # 저장
        try:
            base, _ = os.path.splitext(output_file)
            d.save(f"{base}.svg")
            d.save(f"{base}.png")
            print(f"Practical schemdraw schematic saved")
        except Exception as e:
            print(f"Error saving: {e}")

def analyze_circuit_topology(components, nodes, ground_nodes):
    """
    회로 토폴로지 분석 - 노드 간 연결 관계 파악
    """
    # 노드별 연결된 컴포넌트 리스트
    node_connections = collections.defaultdict(list)
    for comp in components:
        if len(comp.nodes) >= 2:
            node_connections[comp.node1].append(comp)
            node_connections[comp.node2].append(comp)
    
    # 병렬 컴포넌트 그룹화
    parallel_groups = collections.defaultdict(list)
    for comp in components:
        if len(comp.nodes) >= 2:
            key = tuple(sorted([comp.node1, comp.node2]))
            parallel_groups[key].append(comp)
    
    # 회로 트리 구조 생성
    circuit_tree = {
        'node_connections': node_connections,
        'parallel_groups': parallel_groups,
        'ground_nodes': ground_nodes,
        'all_nodes': nodes,
        'components': components
    }
    
    return circuit_tree

def draw_main_path(d, circuit_tree):
    """
    메인 경로 그리기 - 가장 많이 연결된 경로를 따라 그리기
    """
    drawn_components = set()
    parallel_groups = circuit_tree['parallel_groups']
    
    # 각 병렬 그룹에서 대표 컴포넌트만 선택하여 메인 경로 구성
    for node_pair, comp_group in parallel_groups.items():
        if not comp_group:
            continue
            
        # 첫 번째 컴포넌트를 대표로 선택
        main_comp = comp_group[0]
        
        # 컴포넌트 그리기
        label = f'{main_comp.name}\n{main_comp.value}' if main_comp.value else main_comp.name
        elem = create_element(main_comp, label)
        
        # 간단한 배치: 교대로 방향 전환
        if len(drawn_components) % 4 == 0:
            d.add(elem.up())
        elif len(drawn_components) % 4 == 1:
            d.add(elem.right())
        elif len(drawn_components) % 4 == 2:
            d.add(elem.down())
        else:
            d.add(elem.left())
        
        d.add(elm.Dot())  # 연결점 표시
        drawn_components.add(main_comp.name)
        
        # 나머지 병렬 컴포넌트는 branches에서 처리
        circuit_tree['drawn_main'] = drawn_components
        circuit_tree['branch_components'] = comp_group[1:] if len(comp_group) > 1 else []

def draw_branches(d, circuit_tree):
    """
    브랜치(병렬 컴포넌트) 추가
    """
    # 이 부분은 schemdraw의 제약으로 인해 구현이 복잡합니다
    # 간단한 대안: 병렬 컴포넌트를 주석으로만 표시
    parallel_groups = circuit_tree['parallel_groups']
    
    for node_pair, comp_group in parallel_groups.items():
        if len(comp_group) > 1:
            # 병렬 컴포넌트들을 텍스트로 표시
            parallel_text = "Parallel: " + ", ".join([c.name for c in comp_group[1:]])
            d.add(elm.Label(parallel_text).at((5, -1)))
            break  # 첫 번째 병렬 그룹만 표시

def create_element(comp, label):
    """컴포넌트 타입에 따른 element 생성"""
    elem_map = {
        'R': elm.Resistor,
        'C': elm.Capacitor,
        'L': elm.LED,
        'V': elm.SourceV,
        'I': elm.SourceI,
        'D': elm.Diode,
    }
    

    
    elem_class = elem_map.get(comp.type, elm.Resistor)
    return elem_class(label=label)


def get_element_class(comp_type, comp_name):
    """컴포넌트 타입에 따른 schemdraw element 클래스 반환"""
    if comp_type == 'R': return elm.Resistor
    elif comp_type == 'C': return elm.Capacitor
    elif comp_type == 'L': return elm.LED
    elif comp_type == 'D': return elm.Diode
    elif comp_type == 'V': return elm.SourceV
    elif comp_type == 'I': return elm.SourceI
    else: return elm.Resistor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert SPICE netlist files to circuit diagrams.'
    )
    parser.add_argument(
        '-s', '--style', 
        choices=['svg', 'schemdraw', 'png', 'auto'],
        default='123123',
        help="Output style"
    )
    args = parser.parse_args()
    
    files = glob.glob('*.spice') + glob.glob('*.net')
    if not files:
        print('No SPICE files found.')
        sys.exit(0)

    for spice_file in files:
        base = os.path.splitext(spice_file)[0]
        print(f"\nProcessing {spice_file}...")
        
        if args.style == 'auto':
            out_file = f"{base}.png"
            convert_spice_to_schemdraw_auto_png(spice_file, out_file)
        elif args.style == 'png':
            out_file = f"{base}.png"
            convert_spice_to_schemdraw_png(spice_file, out_file)
        elif args.style == 'schemdraw':
            out_file = f"{base}.svg"
            convert_spice_to_schemdraw(spice_file, out_file)
        else:
            out_file = f"{base}_custom.svg"
            convert_spice_to_svg(spice_file, out_file)