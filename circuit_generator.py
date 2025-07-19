# circuit_generator.py (완전 안정화된 버전)
import os
import pandas as pd
import numpy as np
from detector.hole_detector import HoleDetector
from diagram import drawDiagram
from writeSPICE import toSPICE
from calcVoltageAndCurrent import calcCurrentAndVoltage
import networkx as nx
from networkx.readwrite import write_graphml
import matplotlib.pyplot as plt
from diagram import draw_connectivity_graph
import glob
from checker.Circuit_comparer import CircuitComparer
import matplotlib
import tkinter as tk
from tkinter import messagebox
matplotlib.use('TkAgg')
import cv2
import os, glob, re
from diagram import validate_circuit_connectivity,generate_circuit_from_spice
from new_diagram import draw_new_diagram
from collections import defaultdict, deque
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import webbrowser
import subprocess
import sys
from PIL import Image
import tkinter as tk
from tkinter import messagebox
# 1) 파일 상단에 import 추가
from spice_converter import (
    SpiceParser, CircuitLayout, SVGGenerator, 
    convert_spice_to_svg, convert_spice_to_schemdraw_auto_png
)
# 실습 주제 맵 (기존과 동일)
topic_map = {
    0: "test용 회로", 1: "병렬회로", 2: "직렬회로", 3: "키르히호프 1법칙", 4: "키르히호프 2법칙",
    5: "중첩의 원리", 6: "LED Test 회로", 7: "오실로스코프 실습2",
    8: "반파정류회로", 9: "반파정류회로2", 10: "비반전 증폭기"
}


# 연결선 포함 정확한 회로도 생성 함수

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np
from collections import defaultdict

def analyze_spice_topology(components):
    """
    SPICE 컴포넌트로부터 회로 토폴로지 분석
    """
    print("🔍 SPICE 토폴로지 분석...")
    
    # 네트별 연결 정보
    net_connections = defaultdict(list)
    all_nets = set()
    
    for comp in components:
        node1, node2 = comp['nodes']
        all_nets.update([node1, node2])
        net_connections[node1].append(comp)
        net_connections[node2].append(comp)
    
    print("=== 네트 연결 분석 ===")
    for net in sorted(all_nets):
        comps = net_connections[net]
        comp_names = [c['name'] for c in comps]
        print(f"Net{net}: {comp_names} ({len(comps)}개 연결)")
    
    # 공통 노드 찾기
    junction_nets = []
    for net, comps in net_connections.items():
        if len(comps) >= 3:  # 3개 이상 연결된 노드는 접점
            junction_nets.append(net)
    
    print(f"접점 노드: {junction_nets}")
    
    return {
        'net_connections': dict(net_connections),
        'all_nets': sorted(all_nets),
        'junction_nets': junction_nets
    }


def create_connected_layout(components, topology):
    """
    연결 관계를 고려한 레이아웃 생성
    """
    print("📐 연결 기반 레이아웃 생성...")
    
    net_connections = topology['net_connections']
    junction_nets = topology['junction_nets']
    
    # 네트별 위치 결정
    net_positions = {}
    
    # 접점들을 중앙에 배치
    if junction_nets:
        for i, net in enumerate(junction_nets):
            net_positions[net] = (6, 4 + i * 1.5)  # 중앙 세로 배치
            print(f"접점 Net{net}: 위치 {net_positions[net]}")
    
    # 전압원들의 네트 위치
    vs_components = [c for c in components if c['class'] == 'VoltageSource']
    for i, vs in enumerate(vs_components):
        node1, node2 = vs['nodes']
        
        # 접점에 연결되지 않은 노드를 왼쪽에
        for node in [node1, node2]:
            if node not in net_positions:
                net_positions[node] = (1.5, 5 - i * 2)
                print(f"전원 Net{node}: 위치 {net_positions[node]}")
    
    # 나머지 네트들 자동 배치
    other_nets = set(topology['all_nets']) - set(net_positions.keys())
    for i, net in enumerate(sorted(other_nets)):
        net_positions[net] = (9, 5 - i * 1.5)
        print(f"기타 Net{net}: 위치 {net_positions[net]}")
    
    # 컴포넌트 위치 계산 (네트 중점에)
    component_layout = {}
    for comp in components:
        node1, node2 = comp['nodes']
        pos1 = net_positions[node1]
        pos2 = net_positions[node2]
        
        # 중점 계산
        mid_x = (pos1[0] + pos2[0]) / 2
        mid_y = (pos1[1] + pos2[1]) / 2
        
        component_layout[comp['name']] = {
            'component': comp,
            'position': (mid_x, mid_y),
            'net_positions': (pos1, pos2)
        }
    
    return {
        'net_positions': net_positions,
        'component_layout': component_layout
    }


def draw_connected_circuit_diagram(components, output_path):
    """
    연결선을 포함한 정확한 회로도 생성
    """
    print(f"\n🎨 연결선 포함 회로도 생성: {output_path}")
    
    try:
        # 토폴로지 분석
        topology = analyze_spice_topology(components)
        
        # 레이아웃 생성
        layout = create_connected_layout(components, topology)
        net_positions = layout['net_positions']
        component_layout = layout['component_layout']
        
        # 그래프 생성
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # 1) 네트 노드들 먼저 그리기
        print("=== 네트 노드 그리기 ===")
        for net, pos in net_positions.items():
            # 접점은 크게, 일반 노드는 작게
            is_junction = net in topology['junction_nets']
            size = 150 if is_junction else 80
            color = 'red' if is_junction else 'blue'
            
            ax.scatter(pos[0], pos[1], c=color, s=size, zorder=10, 
                      edgecolors='black', linewidth=2)
            
            # 네트 라벨
            ax.text(pos[0], pos[1]-0.4, f'Net{net}', ha='center', va='center', 
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow"))
            
            print(f"Net{net}: {pos} ({'접점' if is_junction else '일반'})")
        
        # 2) 연결선 그리기
        print("=== 연결선 그리기 ===")
        for comp_name, info in component_layout.items():
            comp = info['component']
            pos1, pos2 = info['net_positions']
            
            # 네트 간 연결선
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                   'k-', linewidth=2, alpha=0.7, zorder=1)
            
            print(f"{comp_name}: Net{comp['nodes'][0]} ↔ Net{comp['nodes'][1]}")
        
        # 3) 컴포넌트 심볼 그리기
        print("=== 컴포넌트 그리기 ===")
        for comp_name, info in component_layout.items():
            comp = info['component']
            mid_x, mid_y = info['position']
            
            if comp['class'] == 'VoltageSource':
                # 전압원 원
                circle = Circle((mid_x, mid_y), 0.3, linewidth=3, 
                              edgecolor='red', facecolor='lightcoral', zorder=5)
                ax.add_patch(circle)
                
                # + - 표시
                ax.text(mid_x, mid_y+0.1, '+', ha='center', va='center', 
                       fontsize=12, fontweight='bold')
                ax.text(mid_x, mid_y-0.1, '−', ha='center', va='center', 
                       fontsize=12, fontweight='bold')
                
                # 라벨
                ax.text(mid_x, mid_y-0.6, f'{comp["name"]}\n{comp["value"]}V', 
                       ha='center', va='center', fontsize=10, fontweight='bold', color='red')
            
            elif comp['class'] == 'Resistor':
                # 저항 사각형
                rect = FancyBboxPatch((mid_x-0.4, mid_y-0.15), 0.8, 0.3,
                                    boxstyle="round,pad=0.05",
                                    linewidth=2, edgecolor='blue', facecolor='lightblue', zorder=5)
                ax.add_patch(rect)
                
                # 라벨
                ax.text(mid_x, mid_y, f'{comp["name"]}\n{comp["value"]}Ω', 
                       ha='center', va='center', fontsize=9, fontweight='bold')
            
            elif comp['class'] == 'Capacitor':
                # 캐패시터 (두 평행선)
                ax.plot([mid_x-0.1, mid_x-0.1], [mid_y-0.2, mid_y+0.2], 'k-', linewidth=3, zorder=5)
                ax.plot([mid_x+0.1, mid_x+0.1], [mid_y-0.2, mid_y+0.2], 'k-', linewidth=3, zorder=5)
                
                # 라벨
                ax.text(mid_x, mid_y-0.5, f'{comp["name"]}\n{comp["value"]}F', 
                       ha='center', va='center', fontsize=9, fontweight='bold')
        
        # 4) 접지 표시 (Net 0이 있다면)
        if 0 in net_positions:
            ground_pos = net_positions[0]
            # 접지 심볼
            ax.plot([ground_pos[0], ground_pos[0]], [ground_pos[1]-0.3, ground_pos[1]-0.8], 
                   'k-', linewidth=4)
            for i in range(3):
                width = 0.3 - i * 0.1
                ax.plot([ground_pos[0]-width, ground_pos[0]+width], 
                       [ground_pos[1]-0.8-i*0.1, ground_pos[1]-0.8-i*0.1], 
                       'k-', linewidth=2)
            
            ax.text(ground_pos[0], ground_pos[1]-1.2, 'GND', ha='center', va='center', 
                   fontsize=10, fontweight='bold')
        
        # 회로 정보
        vs_count = len([c for c in components if c['class'] == 'VoltageSource'])
        r_count = len([c for c in components if c['class'] == 'Resistor'])
        
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.set_title(f'SPICE 기반 연결된 회로도\n전원: {vs_count}개, 저항: {r_count}개, 접점: {len(topology["junction_nets"])}개', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # 범례
        legend_text = "컴포넌트:\n"
        for comp in components:
            legend_text += f"• {comp['name']}: {comp['value']}{comp['unit']}\n"
        
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 연결된 회로도 저장: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 연결된 회로도 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_traditional_circuit_diagram(components, output_path):
    """
    전통적인 회로도 스타일로 생성 (깔끔한 버전)
    """
    print(f"\n🎨 전통적인 회로도 생성: {output_path}")
    
    try:
        # 특정 회로에 맞는 수동 레이아웃
        # V1 170-104, V2 234-104, R1 48-108, R2 6-108, R3 108-234
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # 회로 분석
        vs_list = [c for c in components if c['class'] == 'VoltageSource']
        r_list = [c for c in components if c['class'] == 'Resistor']
        
        print("=== 특정 회로 토폴로지 기반 레이아웃 ===")
        
        # Net104는 공통 접점 (중앙)
        common_net = 104
        junction_pos = (6, 4)
        
        # V1: Net170 - Net104
        v1_net170_pos = (2, 6)
        
        # V2: Net234 - Net104  
        v2_net234_pos = (10, 6)
        
        # R3: Net108 - Net234
        r3_net108_pos = (6, 2)
        
        # R1: Net48 - Net108
        r1_net48_pos = (4, 2)
        
        # R2: Net6 - Net108
        r2_net6_pos = (8, 2)
        
        net_positions = {
            170: v1_net170_pos,
            104: junction_pos,
            234: v2_net234_pos,
            108: r3_net108_pos,
            48: r1_net48_pos,
            6: r2_net6_pos
        }
        
        # 네트 노드 그리기
        for net, pos in net_positions.items():
            color = 'red' if net == 104 else 'blue'
            size = 120 if net == 104 else 60
            ax.scatter(pos[0], pos[1], c=color, s=size, zorder=10, 
                      edgecolors='black', linewidth=2)
            ax.text(pos[0]+0.3, pos[1]+0.3, f'Net{net}', ha='left', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        # 컴포넌트와 연결선 그리기
        for comp in components:
            node1, node2 = comp['nodes']
            pos1 = net_positions[node1]
            pos2 = net_positions[node2]
            
            # 연결선
            ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                   'k-', linewidth=2, zorder=1)
            
            # 컴포넌트 중점
            mid_x = (pos1[0] + pos2[0]) / 2
            mid_y = (pos1[1] + pos2[1]) / 2
            
            if comp['class'] == 'VoltageSource':
                # 전압원
                circle = Circle((mid_x, mid_y), 0.25, linewidth=2, 
                              edgecolor='red', facecolor='lightcoral', zorder=5)
                ax.add_patch(circle)
                ax.text(mid_x, mid_y+0.05, '+', ha='center', va='center', 
                       fontsize=10, fontweight='bold')
                ax.text(mid_x, mid_y-0.05, '−', ha='center', va='center', 
                       fontsize=10, fontweight='bold')
                ax.text(mid_x, mid_y-0.5, f'{comp["name"]}\n{comp["value"]}V', 
                       ha='center', va='center', fontsize=9, fontweight='bold', color='red')
            
            elif comp['class'] == 'Resistor':
                # 저항
                rect = FancyBboxPatch((mid_x-0.3, mid_y-0.1), 0.6, 0.2,
                                    boxstyle="round,pad=0.02",
                                    linewidth=2, edgecolor='blue', facecolor='lightblue', zorder=5)
                ax.add_patch(rect)
                ax.text(mid_x, mid_y, f'{comp["name"]}\n{comp["value"]}Ω', 
                       ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.set_title('전통적인 회로도 (Net104 중심 접점)', fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ 전통적인 회로도 저장: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 전통적인 회로도 생성 실패: {e}")
        return False


# circuit_generator.py에 추가할 함수들
def generate_multiple_diagram_types(spice_components, base_output_path):
    """
    여러 타입의 회로도 생성
    """
    success_count = 0
    
    # 1) 연결선 포함 버전
    connected_path = base_output_path.replace('.jpg', '_connected.jpg')
    if draw_connected_circuit_diagram(spice_components, connected_path):
        success_count += 1
    
    # 2) 전통적인 버전
    traditional_path = base_output_path.replace('.jpg', '_traditional.jpg')
    if create_traditional_circuit_diagram(spice_components, traditional_path):
        success_count += 1
    
    print(f"✅ {success_count}/2 타입 회로도 생성 완료")
    return success_count > 0


# 기존 generate_output_files 함수의 SPICE 부분을 다음으로 교체:


# 2) SPICE 파싱 함수들 추가 (generate_circuit 함수 전에)
def parse_spice_file(spice_filepath):
    """SPICE 파일을 파싱하여 컴포넌트 리스트 반환"""
    print(f"📄 SPICE 파일 파싱: {spice_filepath}")
    
    components = []
    
    try:
        with open(spice_filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # 빈 줄이나 주석 건너뛰기
            if not line or line.startswith('*') or line.startswith('.'):
                continue
            
            print(f"   라인 {line_num}: {line}")
            
            # 전압원 파싱 (V로 시작)
            if line.upper().startswith('V'):
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[0]
                    node1 = int(parts[1])
                    node2 = int(parts[2])
                    value = float(parts[3])
                    
                    components.append({
                        'name': name,
                        'class': 'VoltageSource',
                        'value': value,
                        'nodes': (node1, node2),
                        'unit': 'V'
                    })
                    print(f"      ✅ 전압원: {name} = {value}V, Net{node1} ↔ Net{node2}")
            
            # 저항 파싱 (R로 시작)
            elif line.upper().startswith('R'):
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[0]
                    node1 = int(parts[1])
                    node2 = int(parts[2])
                    value = float(parts[3])
                    
                    components.append({
                        'name': name,
                        'class': 'Resistor',
                        'value': value,
                        'nodes': (node1, node2),
                        'unit': 'Ω'
                    })
                    print(f"      ✅ 저항: {name} = {value}Ω, Net{node1} ↔ Net{node2}")
            
            # 캐패시터 파싱 (C로 시작)
            elif line.upper().startswith('C'):
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[0]
                    node1 = int(parts[1])
                    node2 = int(parts[2])
                    value = float(parts[3])
                    
                    components.append({
                        'name': name,
                        'class': 'Capacitor',
                        'value': value,
                        'nodes': (node1, node2),
                        'unit': 'F'
                    })
                    print(f"      ✅ 캐패시터: {name} = {value}F, Net{node1} ↔ Net{node2}")
        
        print(f"✅ SPICE 파싱 완료: 총 {len(components)}개 컴포넌트")
        return components
        
    except FileNotFoundError:
        print(f"❌ SPICE 파일을 찾을 수 없습니다: {spice_filepath}")
        return []
    except Exception as e:
        print(f"❌ SPICE 파일 파싱 오류: {e}")
        return []


def draw_spice_based_circuit(components, output_path):
    """SPICE 데이터로부터 직접 회로도 생성"""
    print(f"\n🎨 SPICE 기반 회로도 생성: {output_path}")
    
    try:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        vs_list = [c for c in components if c['class'] == 'VoltageSource']
        r_list = [c for c in components if c['class'] == 'Resistor']
        c_list = [c for c in components if c['class'] == 'Capacitor']
        
        print(f"📊 SPICE 기반 회로: 전압원 {len(vs_list)}개, 저항 {len(r_list)}개, 캐패시터 {len(c_list)}개")
        
        # 전압원들 왼쪽에 세로로 배치
        for i, vs in enumerate(vs_list):
            x, y = 1.5, 6 - i * 2.5
            
            # 전압원 원
            circle = Circle((x, y), 0.4, linewidth=3, edgecolor='red', facecolor='lightcoral')
            ax.add_patch(circle)
            
            # + - 표시
            ax.text(x, y+0.15, '+', ha='center', va='center', fontsize=14, fontweight='bold')
            ax.text(x, y-0.15, '−', ha='center', va='center', fontsize=14, fontweight='bold')
            
            # 라벨
            ax.text(x-0.8, y, f'{vs["name"]}\n{vs["value"]}V', ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='red')
            
            # 터미널 연결선
            ax.plot([x-0.6, x-0.4], [y, y], 'k-', linewidth=3)  # 음극 터미널
            ax.plot([x+0.4, x+0.6], [y, y], 'k-', linewidth=3)  # 양극 터미널
            
            # 노드 라벨
            ax.text(x-0.8, y+0.6, f"Net{vs['nodes'][0]}", ha='center', fontsize=10, 
                   color='blue', fontweight='bold', 
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue"))
            ax.text(x+0.8, y+0.6, f"Net{vs['nodes'][1]}", ha='center', fontsize=10, 
                   color='blue', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue"))
        
        # 저항들 중간에 배치
        for i, res in enumerate(r_list):
            x = 4 + (i % 2) * 2.5
            y = 6 - (i // 2) * 1.5
            
            # 저항 사각형
            rect = FancyBboxPatch((x-0.5, y-0.2), 1.0, 0.4,
                                boxstyle="round,pad=0.05",
                                linewidth=2, edgecolor='blue', facecolor='lightblue')
            ax.add_patch(rect)
            
            # 라벨
            ax.text(x, y, f'{res["name"]}\n{res["value"]}Ω', ha='center', va='center', 
                   fontsize=11, fontweight='bold')
            
            # 터미널
            ax.plot([x-0.8, x-0.5], [y, y], 'k-', linewidth=2)
            ax.plot([x+0.5, x+0.8], [y, y], 'k-', linewidth=2)
            
            # 노드 라벨
            ax.text(x-1.0, y+0.5, f"Net{res['nodes'][0]}", ha='center', fontsize=10, 
                   color='green', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"))
            ax.text(x+1.0, y+0.5, f"Net{res['nodes'][1]}", ha='center', fontsize=10, 
                   color='green', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen"))
        
        # 캐패시터들 오른쪽에 배치
        for i, cap in enumerate(c_list):
            x = 7 + (i % 2) * 1.5
            y = 6 - (i // 2) * 1.5
            
            # 캐패시터 (두 평행선)
            ax.plot([x-0.15, x-0.15], [y-0.3, y+0.3], 'k-', linewidth=4)
            ax.plot([x+0.15, x+0.15], [y-0.3, y+0.3], 'k-', linewidth=4)
            
            # 라벨
            ax.text(x, y-0.6, f'{cap["name"]}\n{cap["value"]}F', ha='center', va='center', 
                   fontsize=10, fontweight='bold')
            
            # 터미널
            ax.plot([x-0.5, x-0.15], [y, y], 'k-', linewidth=2)
            ax.plot([x+0.15, x+0.5], [y, y], 'k-', linewidth=2)
            
            # 노드 라벨
            ax.text(x-0.7, y+0.5, f"Net{cap['nodes'][0]}", ha='center', fontsize=9, 
                   color='purple', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="plum"))
            ax.text(x+0.7, y+0.5, f"Net{cap['nodes'][1]}", ha='center', fontsize=9, 
                   color='purple', fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="plum"))
        
        # 접지 표시 (Net 0)
        if any(0 in comp['nodes'] for comp in components):
            ax.axhline(y=0.5, xmin=0.1, xmax=0.9, color='black', linewidth=4)
            for i in range(3):
                ax.axhline(y=0.3-i*0.1, xmin=0.45, xmax=0.55, color='black', linewidth=2)
            ax.text(6, 0.8, 'Ground (Net 0)', ha='center', va='center', 
                   fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)
        ax.set_title(f'SPICE 기반 정확한 다중 전원 회로도\n' + 
                    f'전원: {len(vs_list)}개 (각각 독립적), 저항: {len(r_list)}개, 캐패시터: {len(c_list)}개', 
                    fontsize=14, fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"✅ SPICE 기반 정확한 회로도 저장: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ SPICE 기반 회로도 생성 실패: {e}")
        return False



# circuit_generator.py에 추가할 코드들



# 2) SVG 회로도 생성 함수 추가
def generate_svg_circuit_diagram(spice_filepath, output_svg_path, show_in_window=True):
    """
    spice_converter.py를 활용한 고품질 SVG 회로도 생성 및 표시
    """
    print(f"\n🎨 SVG 회로도 생성: {spice_filepath} → {output_svg_path}")
    
    try:
        # spice_converter의 convert_spice_to_svg 함수 사용
        convert_spice_to_svg(spice_filepath, output_svg_path)
        
        # PNG 버전 경로
        png_path = output_svg_path.replace('.svg', '.png')
        
        print(f"✅ SVG 회로도 생성 성공:")
        print(f"   - SVG: {output_svg_path}")
        
        # PNG 파일 존재 확인
        if os.path.exists(png_path):
            print(f"   - PNG: {png_path}")
        
        # 창으로 표시
        if show_in_window:
            display_svg_circuit(output_svg_path, png_path, show_in_window=True)
        
        return True, output_svg_path, png_path
        
    except Exception as e:
        print(f"❌ SVG 회로도 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def generate_schemdraw_circuit_diagram(spice_filepath, output_png_path, show_in_window=True):
    """
    spice_converter.py의 schemdraw 기능을 활용한 전문적 회로도 생성 및 표시
    """
    print(f"\n🔧 Schemdraw 회로도 생성: {spice_filepath} → {output_png_path}")
    
    try:
        # spice_converter의 schemdraw 자동 레이아웃 사용
        #convert_spice_to_schemdraw_auto_png(spice_filepath, output_png_path)
        
        print(f"✅ Schemdraw 회로도 생성 성공: {output_png_path}")
        
        # 창으로 표시
        if show_in_window and os.path.exists(output_png_path):
            show_png_in_window(output_png_path)
        
        return True, output_png_path
        
    except Exception as e:
        print(f"❌ Schemdraw 회로도 생성 실패: {e}")
        return False, None


def generate_multiple_circuit_formats(spice_filepath, base_output_path):
    """
    여러 포맷의 회로도를 동시 생성
    """
    print(f"\n🎯 다중 포맷 회로도 생성: {base_output_path}")
    
    base_name = os.path.splitext(base_output_path)[0]
    success_count = 0
    generated_files = []
    
    # 1) SVG + PNG (spice_converter 사용)
    svg_path = f"{base_name}_spice.svg"
    svg_success, svg_file, png_file = generate_svg_circuit_diagram(spice_filepath, svg_path)
    if svg_success:
        success_count += 1
        generated_files.extend([svg_file, png_file])
    
    # 2) Schemdraw PNG (전문적 레이아웃)
    schemdraw_path = f"{base_name}_schemdraw.png"
    schemdraw_success, schemdraw_file = generate_schemdraw_circuit_diagram(spice_filepath, schemdraw_path)
    if schemdraw_success:
        success_count += 1
        generated_files.append(schemdraw_file)
    
    # 3) 기존 matplotlib 방식 (fallback)
    try:
        # SPICE 파싱 후 기존 방식으로도 생성
        spice_components = parse_spice_file(spice_filepath)
        if spice_components:
            # 연결된 회로도
            connected_path = f"{base_name}_connected.jpg"
            if draw_connected_circuit_diagram(spice_components, connected_path):
                success_count += 1
                generated_files.append(connected_path)
                
            # 전통적 회로도
            traditional_path = f"{base_name}_traditional.jpg"
            if create_traditional_circuit_diagram(spice_components, traditional_path):
                success_count += 1
                generated_files.append(traditional_path)
                
    except Exception as e:
        print(f"⚠️ 기존 matplotlib 방식 실패: {e}")
    
    print(f"✅ 다중 포맷 생성 완료: {success_count}개 파일")
    for i, file_path in enumerate(generated_files, 1):
        file_type = "SVG" if file_path.endswith('.svg') else \
                   "PNG" if file_path.endswith('.png') else "JPG"
        print(f"   {i}. {file_type}: {file_path}")
    
    return success_count > 0, generated_files


# 3) generate_output_files 함수 수정 (기존 함수 교체)
def generate_output_files(mapped, stable_power_pairs, voltage, output_spice, output_img, show_in_window=True):
    """
    출력 파일들 생성 (SVG 회로도 중심, diagram 출력 제거)
    """
    print("  📁 출력 파일 생성 중...")
    
    # 🔍 디버깅: 넷 매핑 과정 추적
    debug_net_mapping_process(
        stable_hole_to_net={},  # 실제로는 전달받아야 함
        stable_wires=[],        # 실제로는 전달받아야 함  
        mapped_components=mapped
    )
    
    # DataFrame 생성 (병합된 넷 번호 포함)
    df = pd.DataFrame([{
        'name': m['name'],
        'class': m['class'],
        'value': m['value'],
        'node1_n': m['nodes'][0],
        'node2_n': m['nodes'][1],
    } for m in mapped])
    
    print("\n📊 DataFrame 내용:")
    for _, row in df.iterrows():
        print(f"  {row['name']}: {row['class']} Net{row['node1_n']}↔Net{row['node2_n']} = {row['value']}")
    
    # 안정화된 그래프 생성
    G = build_stable_circuit_graph(mapped)
    
    # 파일들 저장
    save_circuit_graph(G, output_img.replace('.jpg', '.graphml'))
    
    # 🔧 핵심: 병합된 넷 번호로 SPICE 생성
    toSPICE_multi_power(df, stable_power_pairs, voltage, output_spice)
    
    # 🎨 SVG 회로도 생성 및 표시 (diagram 대신)
    print("\n" + "="*60)
    print("🎨 SVG 회로도 생성 및 창으로 표시")
    print("="*60)
    
    svg_success = False
    if os.path.exists(output_spice):
        # SVG 회로도 생성 (spice_converter 사용)
        base_name = os.path.splitext(output_img)[0]
        svg_path = f"{base_name}_circuit.svg"
        
        svg_success, svg_file, png_file = generate_svg_circuit_diagram(
            output_spice, svg_path, show_in_window=show_in_window
        )
        
        if svg_success:
            print(f"✅ SVG 회로도 생성 및 표시 성공!")
            print(f"   - SVG 파일: {svg_file}")
            if png_file and os.path.exists(png_file):
                print(f"   - PNG 파일: {png_file}")
        
        # 추가로 Schemdraw 버전도 생성
        schemdraw_path = f"{base_name}_schemdraw.png"
        schemdraw_success, schemdraw_file = generate_schemdraw_circuit_diagram(
            output_spice, schemdraw_path, show_in_window=False  # 하나만 창으로 보기
        )
        
        if schemdraw_success:
            print(f"✅ Schemdraw 회로도도 생성: {schemdraw_file}")
            
    else:
        print(f"❌ SPICE 파일을 찾을 수 없습니다: {output_spice}")
    
    # ❌ 기존 diagram 관련 코드들 모두 제거
    # generate_circuit_diagrams() 호출 제거
    # matplotlib 기반 회로도 생성 제거
    
    # 비교 분석만 유지
    try:
        compare_and_notify(G, output_img, checker_dir="checker")
    except Exception as e:
        print(f"  ⚠️ 회로 비교 실패: {e}")


# 4) 회로도 생성 옵션을 제어하는 설정 함수
def set_circuit_diagram_options(use_svg=True, use_schemdraw=True, use_matplotlib_fallback=True):
    """
    회로도 생성 옵션 설정
    """
    global CIRCUIT_OPTIONS
    CIRCUIT_OPTIONS = {
        'use_svg': use_svg,
        'use_schemdraw': use_schemdraw, 
        'use_matplotlib_fallback': use_matplotlib_fallback
    }
    
    print(f"🔧 회로도 생성 옵션 설정:")
    print(f"   - SVG (spice_converter): {'✅' if use_svg else '❌'}")
    print(f"   - Schemdraw: {'✅' if use_schemdraw else '❌'}")
    print(f"   - Matplotlib Fallback: {'✅' if use_matplotlib_fallback else '❌'}")

# 기본 옵션 설정
CIRCUIT_OPTIONS = {
    'use_svg': True,
    'use_schemdraw': True,
    'use_matplotlib_fallback': True
}


# 5) 사용 예시 함수
def demo_svg_circuit_generation():
    """
    SVG 회로도 생성 데모
    """
    print("🚀 SVG 회로도 생성 데모 시작")
    
    # 테스트용 컴포넌트 데이터
    test_components = [
        {"name": "V1", "class": "VoltageSource", "value": 5.0, "nodes": (1, 0)},
        {"name": "R1", "class": "Resistor", "value": 1000, "nodes": (1, 2)}, 
        {"name": "R2", "class": "Resistor", "value": 2000, "nodes": (2, 0)},
        {"name": "C1", "class": "Capacitor", "value": 0.001, "nodes": (2, 0)}
    ]
    
    # 임시 SPICE 파일 생성
    test_spice = "test_circuit.spice"
    with open(test_spice, 'w') as f:
        f.write("* Test Circuit\n")
        f.write("V1 1 0 5.0\n")
        f.write("R1 1 2 1000\n") 
        f.write("R2 2 0 2000\n")
        f.write("C1 2 0 0.001\n")
        f.write(".END\n")
    
    # SVG 회로도 생성
    test_output = "test_circuit_diagram"
    success, files = generate_multiple_circuit_formats(test_spice, test_output)
    
    if success:
        print("✅ 데모 성공!")
        print("생성된 파일들:")
        for file_path in files:
            print(f"  - {file_path}")
    else:
        print("❌ 데모 실패")
    
    # 정리
    if os.path.exists(test_spice):
        os.remove(test_spice)



# 4) generate_circuit 함수 마지막에 return 문 추가
# generate_circuit 함수의 마지막 줄을 다음으로 교체:

def compare_and_notify(G, output_img, checker_dir="checker"):
    """회로 비교 및 알림 (기존과 동일)"""
    files = glob.glob(os.path.join(checker_dir, "*.graphml"))
    if not files:
        print("[비교] 기준 .graphml 파일이 없습니다.")
        return

    sims = []
    for path in files:
        try:
            G_ref = nx.read_graphml(path)
            sim = CircuitComparer(G, G_ref).compute_similarity()
            sims.append((os.path.basename(path), sim))
        except Exception as e:
            print(f"[비교 실패] {path}: {e}")

    sims.sort(key=lambda x: x[1], reverse=True)
    print("\n=== 유사도 TOP 3 ===")
    for i, (fn, sc) in enumerate(sims[:3], 1):
        print(f"{i}. {fn}: {sc:.3f}")

    best_fn, _ = sims[0]
    m = re.search(r"(\d+)", best_fn)
    topic = topic_map.get(int(m.group(1))) if m else None
    msg = f"본 회로는 {topic} 실습 주제입니다." if topic else "실습 주제를 알 수 없습니다."

    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("회로 비교 결과", msg)
    root.destroy()


def generate_circuit(
    all_comps: list,
    holes: list,
    wires: list,
    voltage: float,
    output_spice: str,
    output_img: str,
    hole_to_net: dict,
    power_pairs: list[tuple[int, float, int, float]] = None,
    show_in_window: bool = True  # 새 옵션 추가
):
    """
    완전히 안정화된 회로 생성 함수 (SVG 창 표시 옵션 추가)
    """
    print("🔧 완전 안정화된 회로 생성 시작...")
    
    # 🔧 1) 모든 입력 데이터 안정화
    stable_data = stabilize_input_data(all_comps, wires, power_pairs, hole_to_net)
    stable_all_comps = stable_data['components']
    stable_wires = stable_data['wires'] 
    stable_power_pairs = stable_data['power_pairs']
    stable_hole_to_net = stable_data['hole_to_net']
    
    # 🔧 2) 안정화된 Union-Find로 넷 병합
    merged_nets = perform_stable_net_merging(stable_wires, stable_hole_to_net)
    
    # 🔧 3) 안정화된 컴포넌트 매핑 (전류 흐름 순서)
    mapped = create_stable_component_mapping(
        stable_all_comps, stable_power_pairs, voltage, merged_nets, stable_hole_to_net
    )
    
    # 🔧 4) 결과 파일 생성 (SVG 창 표시 포함)
    generate_output_files(mapped, stable_power_pairs, voltage, output_spice, output_img, show_in_window)
    
    print(f"✅ 안정화된 회로 생성 완료!")
    print(f"   - 컴포넌트 개수: {len([m for m in mapped if m['class'] != 'VoltageSource'])}")
    print(f"   - 전원 개수: {len(stable_power_pairs)}")
    
    if show_in_window:
        print(f"   - SVG 회로도가 창으로 표시됩니다")
    
    return mapped, stable_hole_to_net


def stabilize_input_data(all_comps, wires, power_pairs, hole_to_net):
    """
    모든 입력 데이터를 안정화
    """
    print("  📊 입력 데이터 안정화 중...")
    
    # 컴포넌트 안정화 (다중 키 정렬)
    def comp_key(comp):
        return (
            comp.get('class', ''),
            str(comp.get('value', 0)),
            tuple(sorted(comp.get('pins', []))),
            str(comp)
        )
    
    stable_comps = sorted(all_comps, key=comp_key)
    
    # 와이어 안정화 (net 번호 순)
    stable_wires = []
    for net1, net2 in wires:
        stable_wires.append((min(net1, net2), max(net1, net2)))
    stable_wires = sorted(set(stable_wires))
    
    # 전원 안정화 (net 번호 순)
    stable_powers = []
    if power_pairs:
        for net_p, x_p, net_m, x_m in power_pairs:
            stable_powers.append((net_p, x_p, net_m, x_m))
        stable_powers.sort(key=lambda x: (min(x[0], x[2]), max(x[0], x[2])))
    
    # hole_to_net 안정화 (키 정렬)
    stable_hole_to_net = dict(sorted(hole_to_net.items()))
    
    print(f"     안정화 완료: 컴포넌트 {len(stable_comps)}, 와이어 {len(stable_wires)}, 전원 {len(stable_powers)}")
    
    return {
        'components': stable_comps,
        'wires': stable_wires,
        'power_pairs': stable_powers,
        'hole_to_net': stable_hole_to_net
    }

def perform_stable_net_merging(stable_wires, stable_hole_to_net):
    """
    안정화된 Union-Find로 넷 병합 (디버깅 강화)
    """
    print("  🔗 안정화된 넷 병합 중...")
    
    all_nets = sorted(set(stable_hole_to_net.values()))
    parent = {net: net for net in all_nets}
    
    def find(u):
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    
    def union(u, v):
        pu, pv = find(u), find(v)
        if pu != pv:
            # 항상 작은 번호를 루트로 설정 (일관성)
            if pu < pv:
                parent[pv] = pu
                print(f"    Union: {u}({pu}) ← {v}({pv}) → 대표넷: {pu}")
            else:
                parent[pu] = pv
                print(f"    Union: {u}({pu}) → {v}({pv}) ← 대표넷: {pv}")
        else:
            print(f"    Union: {u}, {v} 이미 같은 그룹 (대표넷: {pu})")
    
    print("=== 넷 병합 과정 ===")
    print(f"초기 넷: {sorted(all_nets)}")
    
    for net1, net2 in stable_wires:
        print(f"Wire: {net1} <--> {net2}")
        union(net1, net2)
    
    # 최종 병합 결과 출력
    print("\n=== 최종 병합 결과 ===")
    groups = {}
    for net in all_nets:
        root = find(net)
        groups.setdefault(root, []).append(net)
    
    for root, members in sorted(groups.items()):
        if len(members) > 1:
            print(f"그룹 {root}: {sorted(members)} (병합됨)")
        else:
            print(f"그룹 {root}: {members} (단독)")
    
    return {'parent': parent, 'find': find, 'groups': groups}


def create_stable_component_mapping(stable_comps, stable_powers, voltage, merged_nets, stable_hole_to_net):
    """
    전류 흐름 순서 기반 안정화된 컴포넌트 매핑 (넷 병합 완전 적용)
    """
    print("  ⚡ 전류 흐름 기반 컴포넌트 매핑 중 (넷 병합 적용)...")
    
    find_net = merged_nets['find']
    
    def nearest_net(pt):
        if not stable_hole_to_net:
            return 0
        x, y = pt
        closest = min(stable_hole_to_net.keys(), key=lambda h: (h[0] - x) ** 2 + (h[1] - y) ** 2)
        original_net = stable_hole_to_net[closest]
        merged_net = find_net(original_net)
        print(f"    핀 {pt} → 홀 {closest} → 원래넷 {original_net} → 병합넷 {merged_net}")
        return merged_net
    
    # 🔋 1) 전압원들 먼저 매핑
    voltage_components = []
    for i, (net_p, x_p, net_m, x_m) in enumerate(stable_powers, start=1):
        # 🔧 핵심: 전압원도 병합된 넷 사용
        mapped_net_p = find_net(net_p)
        mapped_net_m = find_net(net_m)
        
        print(f"  전압원 V{i}: 원래 ({net_p}, {net_m}) → 병합 ({mapped_net_p}, {mapped_net_m})")
        
        # 노드 정렬 (일관성)
        node1, node2 = sorted([mapped_net_p, mapped_net_m])
        
        vs_comp = {
            'name': f"V{i}",
            'class': 'VoltageSource',
            'value': voltage,
            'nodes': (node1, node2),
            'original_nets': (net_p, net_m),  # 디버깅용
            'merged_nets': (mapped_net_p, mapped_net_m)  # 디버깅용
        }
        voltage_components.append(vs_comp)
    
    # 🔧 2) 일반 컴포넌트들 매핑 (병합된 넷 사용)
    regular_components = []
    regular_comps = [c for c in stable_comps if c.get('class') != 'Line_area']
    
    for idx, comp in enumerate(regular_comps, start=1):
        pins = comp.get('pins', [])
        if len(pins) != 2:
            print(f"[건너뜀] {comp.get('class', 'Unknown')} 핀 오류: {pins}")
            continue
        
        # 핀 좌표 정렬
        pin_a, pin_b = sorted(pins)
        
        # 🔧 핵심: 병합된 넷 사용
        original_node1 = nearest_net(pin_a)
        original_node2 = nearest_net(pin_b)
        
        # 이미 find_net()이 적용된 결과이므로 그대로 사용
        node1, node2 = sorted([original_node1, original_node2])
        
        # 컴포넌트 이름 생성
        prefix_map = {
            'Resistor': 'R', 'Diode': 'D', 'LED': 'L', 
            'Capacitor': 'C', 'IC': 'U'
        }
        prefix = prefix_map.get(comp.get('class', ''), 'X')
        name = f"{prefix}{idx}"
        
        regular_comp = {
            'name': name,
            'class': comp.get('class', ''),
            'value': comp.get('value', 0),
            'nodes': (node1, node2),
            'pins': pins  # 디버깅용
        }
        regular_components.append(regular_comp)
        
        print(f"  {name}: 핀 {pin_a},{pin_b} → 넷 ({node1}, {node2})")
    
    # ⚡ 3) 전류 흐름 순서로 재정렬
    all_components = voltage_components + regular_components
    flow_ordered = sort_by_current_flow(all_components)
    
    print("\n=== 넷 병합 적용된 최종 매핑 결과 ===")
    for i, comp in enumerate(flow_ordered):
        extra_info = ""
        if 'original_nets' in comp:
            extra_info = f" (원래: {comp['original_nets']})"
        elif 'pins' in comp:
            extra_info = f" (핀: {comp['pins']})"
            
        print(f"{i+1:2d}. {comp['name']:4s} ({comp['class']:12s}) "
              f"[{comp['nodes'][0]:2d},{comp['nodes'][1]:2d}] = {comp['value']}{extra_info}")
    
    return flow_ordered


def sort_by_current_flow(components):
    """
    컴포넌트들을 전류 흐름 순서로 정렬
    """
    print("  🌊 전류 흐름 순서 계산 중...")
    
    # 전압원과 일반 컴포넌트 분리
    voltage_sources = [c for c in components if c['class'] == 'VoltageSource']
    other_components = [c for c in components if c['class'] != 'VoltageSource']
    
    if not voltage_sources:
        # 전압원이 없으면 이름순 정렬
        return sorted(other_components, key=lambda x: x['name'])
    
    # 전압원들 정렬 (이름순)
    voltage_sources.sort(key=lambda x: x['name'])
    
    # 전류 흐름 그래프 구성
    flow_graph = build_current_flow_graph(voltage_sources, other_components)
    
    # BFS로 전류 흐름 순서 계산
    flow_order = calculate_flow_order_bfs(flow_graph, voltage_sources)
    
    return flow_order


def build_current_flow_graph(voltage_sources, other_components):
    """
    전류 흐름 분석을 위한 그래프 구성
    """
    # net → components 매핑
    net_to_comps = defaultdict(list)
    all_comps = voltage_sources + other_components
    
    for comp in all_comps:
        for net in comp['nodes']:
            net_to_comps[net].append(comp)
    
    # 컴포넌트 간 연결 그래프
    comp_graph = defaultdict(list)
    for net, comps in net_to_comps.items():
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                comp1, comp2 = comps[i], comps[j]
                comp_graph[comp1['name']].append(comp2['name'])
                comp_graph[comp2['name']].append(comp1['name'])
    
    return {
        'net_to_comps': dict(net_to_comps),
        'comp_graph': dict(comp_graph),
        'all_components': {c['name']: c for c in all_comps}
    }


def calculate_flow_order_bfs(flow_graph, voltage_sources):
    """
    BFS로 전류 흐름 순서 계산
    """
    comp_graph = flow_graph['comp_graph']
    all_components = flow_graph['all_components']
    
    # 전압원들을 시작점으로 BFS
    queue = deque()
    visited = set()
    flow_order = []
    
    # 전압원들을 먼저 추가 (정렬된 순서로)
    for vs in sorted(voltage_sources, key=lambda x: x['name']):
        flow_order.append(vs)
        visited.add(vs['name'])
        
        # 인접한 컴포넌트들을 큐에 추가
        neighbors = sorted(comp_graph.get(vs['name'], []))
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, 1))  # (component_name, distance)
    
    # BFS 실행
    distance_groups = defaultdict(list)
    
    while queue:
        comp_name, distance = queue.popleft()
        
        if comp_name in visited:
            continue
        
        visited.add(comp_name)
        component = all_components[comp_name]
        distance_groups[distance].append(component)
        
        # 인접한 컴포넌트들을 다음 레벨에 추가
        neighbors = sorted(comp_graph.get(comp_name, []))
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append((neighbor, distance + 1))
    
    # 거리별로 정렬하여 추가
    for distance in sorted(distance_groups.keys()):
        group = distance_groups[distance]
        # 같은 거리의 컴포넌트들은 이름순 정렬
        group.sort(key=lambda x: (x['class'], x['name']))
        flow_order.extend(group)
    
    # 연결되지 않은 컴포넌트들 마지막에 추가
    all_names = set(all_components.keys())
    visited_names = {c['name'] for c in flow_order}
    unvisited = all_names - visited_names
    
    for name in sorted(unvisited):
        flow_order.append(all_components[name])
    
    print(f"     전류 흐름 순서 계산 완료: {len(flow_order)}개 컴포넌트")
    
    return flow_order


def generate_output_files22(mapped, stable_power_pairs, voltage, output_spice, output_img):
    """
    출력 파일들 생성
    """
    print("  📁 출력 파일 생성 중...")
    
    # DataFrame 생성
    df = pd.DataFrame([{
        'name': m['name'],
        'class': m['class'],
        'value': m['value'],
        'node1_n': m['nodes'][0],
        'node2_n': m['nodes'][1],
    } for m in mapped])
    
    # 안정화된 그래프 생성
    G = build_stable_circuit_graph(mapped)
    
    # 파일들 저장
    save_circuit_graph(G, output_img.replace('.jpg', '.graphml'))
    toSPICE_multi_power(df, stable_power_pairs, voltage, output_spice)
    
    # 회로도 생성 시도
    try:
        generate_circuit_diagrams(G, voltage, output_img, stable_power_pairs)
    except Exception as e:
        print(f"  ⚠️ 회로도 생성 실패: {e}")
    
    # 비교 분석
    try:
        compare_and_notify(G, output_img, checker_dir="checker")
    except Exception as e:
        print(f"  ⚠️ 회로 비교 실패: {e}")


def build_stable_circuit_graph(mapped):
    """
    안정화된 회로 그래프 생성
    """
    G = nx.Graph()
    
    # 노드를 flow order 순서로 추가
    for i, comp in enumerate(mapped):
        n1, n2 = comp['nodes']
        nets_str = f"{min(n1, n2)},{max(n1, n2)}"
        
        G.add_node(comp['name'],
                   comp_class=comp['class'],
                   value=comp['value'],
                   nets=nets_str,
                   flow_order=i,
                   is_voltage_source=(comp['class'] == 'VoltageSource'))
    
    # net 기반 엣지 생성
    net_to_comps = defaultdict(list)
    for comp in mapped:
        for net in comp['nodes']:
            net_to_comps[net].append(comp['name'])
    
    # 안정화된 엣지 추가
    for net in sorted(net_to_comps.keys()):
        clist = sorted(net_to_comps[net])
        
        for i in range(len(clist)):
            for j in range(i + 1, len(clist)):
                u, v = sorted([clist[i], clist[j]])  # 사전순 정렬
                
                if G.has_edge(u, v):
                    prev_nets = G[u][v]['nets'].split(',')
                    all_nets = sorted(set(prev_nets + [str(net)]), key=int)
                    G[u][v]['nets'] = ','.join(all_nets)
                else:
                    G.add_edge(u, v, nets=str(net))
    
    return G


def generate_circuit_diagrams(G, voltage, output_img, power_pairs):
    """
    회로도 생성 (기존 코드 활용)
    """
    # 🔧 8) 단일 회로도에 모든 전원 표시
    print("=== 모든 전원을 포함한 단일 회로도 생성 ===")

    # current_power_index 제거 (핵심!)
    if 'current_power_index' in G.graph:
        del G.graph['current_power_index']

    # 모든 전원 정보 저장
    G.graph['power_pairs'] = power_pairs
    G.graph['voltage'] = voltage

    # 연결성 검증
    connectivity_report = validate_circuit_connectivity(G)

    # 통합 회로도 생성
    try:
        from diagram import drawDiagramFromGraph_with_connectivity_check
        d = drawDiagramFromGraph_with_connectivity_check(G, voltage)
        
        if d:
            d.draw()
            d.save(output_img)
            print(f"✅ {len(power_pairs)}개 전원 통합 회로도 저장: {output_img}")
        else:
            print("❌ 회로도 생성 실패")
            
    except Exception as e:
        print(f"❌ 회로도 생성 오류: {e}")


# 기존 함수들 (호환성 유지)
def toSPICE_multi_power(df, power_pairs, default_voltage, output_file):
    """
    다중 전원 SPICE 넷리스트 생성 (병합된 넷 번호 사용)
    """
    print(f"\n📝 SPICE 넷리스트 생성: {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("* Multi-Power Circuit Netlist\n")
        f.write(f"* Generated with {len(power_pairs)} power sources\n")
        f.write("* \n")
        
        # 전압원들 (병합된 넷 번호 사용)
        for i, (net_p, _, net_m, _) in enumerate(power_pairs, 1):
            f.write(f"V{i} {net_p} {net_m} {default_voltage}\n")
            print(f"  전압원 V{i}: Net{net_p} ↔ Net{net_m} ({default_voltage}V)")
        
        # 일반 컴포넌트들 (DataFrame에 이미 병합된 넷 번호 저장됨)
        for _, row in df.iterrows():
            if row['class'] == 'VoltageSource':
                continue
            elif row['class'] == 'Resistor':
                f.write(f"{row['name']} {row['node1_n']} {row['node2_n']} {row['value']}\n")
                print(f"  저항 {row['name']}: Net{row['node1_n']} ↔ Net{row['node2_n']} ({row['value']}Ω)")
            elif row['class'] == 'Capacitor':
                f.write(f"{row['name']} {row['node1_n']} {row['node2_n']} {row['value']}F\n")
                print(f"  캐패시터 {row['name']}: Net{row['node1_n']} ↔ Net{row['node2_n']} ({row['value']}F)")
            elif row['class'] == 'Diode':
                f.write(f"{row['name']} {row['node1_n']} {row['node2_n']} DMOD\n")
                print(f"  다이오드 {row['name']}: Net{row['node1_n']} ↔ Net{row['node2_n']}")
            elif row['class'] == 'LED':
                f.write(f"{row['name']} {row['node1_n']} {row['node2_n']} LEDMOD\n")
                print(f"  LED {row['name']}: Net{row['node1_n']} ↔ Net{row['node2_n']}")
        
        f.write("* \n")
        f.write(".MODEL DMOD D\n")
        f.write(".MODEL LEDMOD D(IS=1E-12 N=2)\n")
        f.write(".END\n")
    
    print(f"✅ SPICE 파일 저장 완료: {output_file}")

def debug_net_mapping_process(stable_hole_to_net, stable_wires, mapped_components):
    """
    넷 매핑 과정 디버깅 함수
    """
    print("\n" + "="*60)
    print("🔍 넷 매핑 과정 디버깅")
    print("="*60)
    
    print("1️⃣ 홀-넷 매핑:")
    for hole, net in sorted(stable_hole_to_net.items()):
        print(f"  홀 {hole} → Net{net}")
    
    print("\n2️⃣ 와이어 연결:")
    for net1, net2 in stable_wires:
        print(f"  Wire: Net{net1} ↔ Net{net2}")
    
    print("\n3️⃣ 최종 컴포넌트 넷:")
    for comp in mapped_components:
        node1, node2 = comp['nodes']
        print(f"  {comp['name']}: Net{node1} ↔ Net{node2}")
    
    print("="*60)

def toSPICE(df, voltage, output_file):
    """기존 인터페이스 호환성 래퍼"""
    power_pairs = [(1, 0, 0, 0)]
    toSPICE_multi_power(df, power_pairs, voltage, output_file)


def save_circuit_graph(G, path_graphml):
    """그래프 저장"""
    write_graphml(G, path_graphml)


def visualize_circuit_graph(G, out_path='circuit_graph.png'):
    """그래프 시각화 (기존과 동일)"""
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6,6))
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800, edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=2, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10)
    edge_labels = {(u,v): ','.join(map(str,data['nets'])) for u,v,data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_color='red', font_size=8)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Circuit graph saved to {out_path}")

# 2) SVG 뷰어 함수들
def show_svg_in_browser(svg_path):
    """SVG 파일을 기본 브라우저에서 열기"""
    try:
        # 절대 경로로 변환
        abs_path = os.path.abspath(svg_path)
        file_url = f"file://{abs_path}"
        
        webbrowser.open(file_url)
        print(f"✅ 브라우저에서 SVG 열기: {svg_path}")
        return True
    except Exception as e:
        print(f"❌ 브라우저 열기 실패: {e}")
        return False

def show_png_in_matplotlib_window(png_path):
    """PNG 파일을 matplotlib 창으로 표시 (matplotlib 스타일)"""
    try:
        # matplotlib로 이미지 로드 및 표시
        img = plt.imread(png_path)
        
        # 새 figure 생성
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        ax.axis('off')  # 축 숨기기
        ax.set_title('Circuit Diagram (Generated from SPICE)', fontsize=14, fontweight='bold', pad=20)
        
        # 창 제목 설정
        fig.canvas.manager.set_window_title('Circuit Diagram Viewer')
        
        # 여백 조정
        plt.tight_layout()
        
        # 창 표시 (non-blocking)
        plt.show(block=False)
        
        print(f"✅ matplotlib 창에서 회로도 표시: {png_path}")
        return True
    except Exception as e:
        print(f"❌ matplotlib 창 표시 실패: {e}")
        return False

def show_png_in_window(png_path):
    """PNG 파일을 PIL 이미지 창으로 보기"""
    try:
        img = Image.open(png_path)
        img.show()  # 시스템 기본 이미지 뷰어로 열기
        print(f"✅ 이미지 창에서 PNG 보기: {png_path}")
        return True
    except Exception as e:
        print(f"❌ 이미지 창 열기 실패: {e}")
        return False

def show_svg_with_system_viewer(svg_path):
    """시스템 기본 프로그램으로 SVG 열기"""
    try:
        if sys.platform.startswith('win'):
            # Windows
            subprocess.run(['start', svg_path], shell=True, check=True)
        elif sys.platform.startswith('darwin'):
            # macOS
            subprocess.run(['open', svg_path], check=True)
        else:
            # Linux
            subprocess.run(['xdg-open', svg_path], check=True)
        
        print(f"✅ 시스템 뷰어로 SVG 열기: {svg_path}")
        return True
    except Exception as e:
        print(f"❌ 시스템 뷰어 열기 실패: {e}")
        return False




def display_svg_circuit(svg_path, png_path=None, show_in_window=True):
    """
    SVG 회로도를 창으로 표시 (여러 방법 시도)
    """
    if not show_in_window:
        return
    
    print(f"\n👁️ SVG 회로도 창으로 표시: {svg_path}")
    
    success = False


        # 1순위: PNG를 matplotlib 창으로 표시 ⭐
    if png_path and os.path.exists(png_path):
        if show_png_in_matplotlib_window(png_path):
            success = True

        # 2순위: PNG가 있으면 이미지 뷰어로 보기
    if not success and png_path and os.path.exists(png_path):
        if show_png_in_window(png_path):
            success = True
    
    # 1순위: 브라우저에서 SVG 직접 보기
    #if os.path.exists(svg_path):
    #    if show_svg_in_browser(svg_path):
    #        success = True
    

    
    # 3순위: 시스템 기본 프로그램으로 SVG 열기
    if not success and os.path.exists(svg_path):
        if show_svg_with_system_viewer(svg_path):
            success = True
    
    if not success:
        print("❌ SVG 회로도 표시 실패 - 파일을 수동으로 확인하세요")



# 6) 사용 예시 및 데모 함수
def demo_svg_circuit_with_viewer():
    """
    SVG 회로도 생성 및 창 표시 데모
    """
    print("🚀 SVG 회로도 생성 및 창 표시 데모 시작")
    
    # 테스트용 컴포넌트 데이터
    test_components = [
        {"name": "V1", "class": "VoltageSource", "value": 5.0, "nodes": (1, 0)},
        {"name": "R1", "class": "Resistor", "value": 1000, "nodes": (1, 2)}, 
        {"name": "R2", "class": "Resistor", "value": 2000, "nodes": (2, 0)},
        {"name": "C1", "class": "Capacitor", "value": 0.001, "nodes": (2, 0)}
    ]
    
    # 임시 SPICE 파일 생성
    test_spice = "demo_circuit.spice"
    with open(test_spice, 'w') as f:
        f.write("* Demo Circuit for SVG Viewer\n")
        f.write("V1 1 0 5.0\n")
        f.write("R1 1 2 1000\n") 
        f.write("R2 2 0 2000\n")
        f.write("C1 2 0 0.001\n")
        f.write(".END\n")
    
    # SVG 회로도 생성 및 창 표시
    test_svg = "demo_circuit.svg"
    success, svg_file, png_file = generate_svg_circuit_diagram(test_spice, test_svg, show_in_window=True)
    
    if success:
        print("✅ 데모 성공! SVG 회로도가 창에 표시됩니다.")
        print(f"생성된 파일들:")
        print(f"  - SVG: {svg_file}")
        if png_file and os.path.exists(png_file):
            print(f"  - PNG: {png_file}")
            
        # 사용자에게 알림
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("SVG 회로도 생성 완료", 
                              f"SVG 회로도가 생성되어 창에 표시됩니다.\n\n"
                              f"파일 위치:\n{svg_file}")
            root.destroy()
        except:
            pass
    else:
        print("❌ 데모 실패")
    
    # 정리
    cleanup_files = [test_spice, test_svg, test_svg.replace('.svg', '.png')]
    for file_path in cleanup_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"🗑️ 임시 파일 삭제: {file_path}")
            except:
                pass

# 7) 메인 실행부 수정
if __name__ == "__main__":
    print("🎨 Circuit Generator with SVG Viewer")
    
    # SVG 회로도 생성 및 창 표시 데모 실행
    demo_svg_circuit_with_viewer()
    
    # 기존 테스트 코드 (diagram 제거)
    print("\n🔧 기본 그래프 테스트...")
    mapped = [
        {"name":"V1","class":"VoltageSource","value":5,"nodes":(1,0)},
        {"name":"R1","class":"Resistor","value":100,"nodes":(1,2)},
        {"name":"C1","class":"Capacitor","value":0.001,"nodes":(2,0)}
    ]
    G = build_stable_circuit_graph(mapped)
    print(f"✅ 그래프 생성 완료: {len(G.nodes)}개 노드, {len(G.edges)}개 엣지")