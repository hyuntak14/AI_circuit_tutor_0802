# checker/Circuit_saver.py (개선된 버전 - Netlist 지원)
import cv2
import numpy as np
import networkx as nx
from networkx.readwrite import write_graphml
import tkinter as tk
from tkinter import simpledialog, messagebox
from datetime import datetime
import os

class CircuitSaver:
    def __init__(self, canvas_size=(800, 600)):
        self.graph = nx.DiGraph()  # 방향성 그래프
        self.canvas_size = canvas_size
        self.node_positions = {}    # {node_name: (x, y)}
        self.node_classes = {}      # {node_name: class}
        self.node_models = {}       # {node_name: model} (for IC)
        self.node_values = {}       # {node_name: value} (for R, C)
        self.edge_pins = {}         # {(src, tgt): (src_pin, tgt_pin)}
        self.next_node_id = 1
        self.edge_start = None
        self.canvas = 255 * np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        
        # 컴포넌트 클래스 + 연결 노드 추가
        self.class_options = [
            'VoltageSource', 'V-', 'Resistor', 'Capacitor', 
            'Diode', 'LED', 'IC', 'Junction'  # Junction 노드 추가
        ]
        
        # 시각적 구분을 위한 색상
        self.node_colors = {
            'VoltageSource': (255, 0, 0),    # 빨강
            'V-': (0, 0, 255),               # 파랑
            'Resistor': (255, 165, 0),       # 주황
            'Capacitor': (128, 0, 128),      # 보라
            'Diode': (0, 255, 0),           # 초록
            'LED': (255, 255, 0),           # 노랑
            'IC': (255, 0, 255),            # 마젠타
            'Junction': (128, 128, 128)      # 회색 (연결 노드)
        }

        self.ua741_pins = {
            1: 'Offset Null', 2: 'Inverting', 3: 'Non-inverting',
            4: 'V-', 5: 'Offset Null', 6: 'Output', 7: 'V+', 8: 'NC'
        }

    def _draw_canvas(self):
        img = self.canvas.copy()
        
        # 노드 그리기
        for name, (x, y) in self.node_positions.items():
            cls = self.node_classes.get(name, '?')
            model = self.node_models.get(name, '')
            color = self.node_colors.get(cls, (200, 200, 200))
            
            # Junction 노드는 작은 원으로 표시
            if cls == 'Junction':
                cv2.circle(img, (x, y), 8, color, -1)
                cv2.circle(img, (x, y), 8, (0, 0, 0), 2)
            else:
                cv2.circle(img, (x, y), 15, color, -1)
                cv2.circle(img, (x, y), 15, (0, 0, 0), 2)
            
            # 텍스트 표시
            cv2.putText(img, name, (x - 10, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(img, cls, (x - 15, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 0, 0), 1)
            if model:
                cv2.putText(img, model, (x - 20, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # 엣지 그리기
        for u, v in self.graph.edges:
            x1, y1 = self.node_positions[u]
            x2, y2 = self.node_positions[v]
            pin_label = self.edge_pins.get((u, v), ('', ''))
            label = f"{pin_label[0]}->{pin_label[1]}" if pin_label[0] or pin_label[1] else ""
            
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            if label:
                cv2.putText(img, label, (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        
        return img

    def _prompt_class_info(self):
        root = tk.Tk()
        root.withdraw()
        
        # 클래스 선택
        class_str = "\n".join(f"{i+1}. {c}" for i, c in enumerate(self.class_options))
        idx = simpledialog.askinteger(
            "클래스 선택", 
            f"컴포넌트 클래스를 선택하세요:\n\n{class_str}", 
            minvalue=1, maxvalue=len(self.class_options)
        )
        
        if not idx:
            root.destroy()
            return None, None, None
            
        comp_class = self.class_options[idx - 1]
        
        # Junction 노드는 간단하게 처리
        if comp_class == 'Junction':
            default_name = f"J{self.next_node_id}"
            name = simpledialog.askstring("노드 이름", "연결 노드 이름:", initialvalue=default_name)
            root.destroy()
            return name, comp_class, 0.0
            
        # 일반 컴포넌트 처리
        default_name = f"{comp_class[0]}{self.next_node_id}"
        name = simpledialog.askstring("노드 이름", "노드 이름을 입력하세요:", initialvalue=default_name)
        
        val = 0.0
        if comp_class in ['Resistor', 'Capacitor']:
            val = simpledialog.askfloat(f"{comp_class} 값 입력", f"{comp_class} 값 입력:", minvalue=0.0) or 0.0
        elif comp_class == 'VoltageSource':
            val = simpledialog.askfloat("전압 값 입력", "전압 값 입력 (V):", minvalue=0.0) or 0.0
            
        root.destroy()
        return name, comp_class, val

    def _prompt_ic_pin(self, message):
        root = tk.Tk()
        root.withdraw()
        pin = simpledialog.askinteger("핀 선택", message + "\n(1~8):", minvalue=1, maxvalue=8)
        root.destroy()
        return pin or ""

    def _auto_detect_parallel_circuits(self):
        """병렬 회로를 자동으로 감지하고 Junction 노드를 제안"""
        # 간단한 병렬 회로 감지 알고리즘
        voltage_sources = [n for n in self.graph.nodes if self.node_classes.get(n) in ['VoltageSource', 'V-']]
        
        if len(voltage_sources) < 2:
            return []
            
        suggestions = []
        # 전원 노드들 간의 병렬 연결 감지
        for i, vs1 in enumerate(voltage_sources):
            for vs2 in voltage_sources[i+1:]:
                # 두 전원 노드가 직접 연결되지 않았다면 병렬 연결 제안
                if not self.graph.has_edge(vs1, vs2) and not self.graph.has_edge(vs2, vs1):
                    suggestions.append((vs1, vs2))
        
        return suggestions

    def _suggest_junction_placement(self):
        """Junction 노드 배치를 제안"""
        suggestions = self._auto_detect_parallel_circuits()
        
        if not suggestions:
            return
            
        root = tk.Tk()
        root.withdraw()
        
        msg = "병렬 회로가 감지되었습니다. Junction 노드를 자동으로 배치하시겠습니까?\n\n"
        msg += "감지된 병렬 연결:\n"
        for i, (n1, n2) in enumerate(suggestions):
            msg += f"{i+1}. {n1} <-> {n2}\n"
            
        if messagebox.askyesno("병렬 회로 감지", msg):
            self._auto_place_junctions(suggestions)
            
        root.destroy()

    def _auto_place_junctions(self, parallel_pairs):
        """자동으로 Junction 노드를 배치하고 연결"""
        for i, (n1, n2) in enumerate(parallel_pairs):
            # 두 노드 사이 중점에 Junction 노드 생성
            x1, y1 = self.node_positions[n1]
            x2, y2 = self.node_positions[n2]
            
            # 양의 연결점
            jx_pos = (x1 + x2) // 2
            jy_pos = min(y1, y2) - 30
            j_pos_name = f"J_pos_{i+1}"
            
            # 음의 연결점  
            jx_neg = (x1 + x2) // 2
            jy_neg = max(y1, y2) + 30
            j_neg_name = f"J_neg_{i+1}"
            
            # Junction 노드 추가
            self.node_positions[j_pos_name] = (jx_pos, jy_pos)
            self.node_classes[j_pos_name] = 'Junction'
            self.node_values[j_pos_name] = 0.0
            self.graph.add_node(j_pos_name, type='Junction', value=0.0)
            
            self.node_positions[j_neg_name] = (jx_neg, jy_neg)
            self.node_classes[j_neg_name] = 'Junction'
            self.node_values[j_neg_name] = 0.0
            self.graph.add_node(j_neg_name, type='Junction', value=0.0)
            
            # 연결 생성 (전원의 양극과 음극 구분 필요)
            # 간단화: 첫 번째 노드를 양극, 두 번째를 음극으로 가정
            self.graph.add_edge(n1, j_pos_name)
            self.graph.add_edge(j_pos_name, n2)
            self.graph.add_edge(n1, j_neg_name)  
            self.graph.add_edge(j_neg_name, n2)
            
            self.next_node_id += 2

    def _generate_spice_netlist(self):
        """회로 그래프를 SPICE 형태로 변환"""
        spice_lines = []
        
        # 헤더 정보 (참고 파일 형식 따라)
        voltage_count = sum(1 for cls in self.node_classes.values() if cls == 'VoltageSource')
        spice_lines.append("* Multi-Power Circuit Netlist")
        spice_lines.append(f"* Generated with {voltage_count} power sources")
        spice_lines.append("* ")
        
        # 노드 매핑 생성 (Junction 노드를 실제 노드 번호로 변환)
        node_mapping = self._create_node_mapping()
        
        # 컴포넌트 정의
        component_counter = {}
        
        for node_name, node_class in self.node_classes.items():
            if node_class in ['Junction', 'V-']:
                continue  # Junction과 V-는 SPICE에서 제외
                
            # 컴포넌트 번호 생성
            if node_class not in component_counter:
                component_counter[node_class] = 1
            else:
                component_counter[node_class] += 1
                
            comp_id = component_counter[node_class]
            value = self.node_values.get(node_name, 0.0)
            
            # 연결된 노드들 찾기
            connected_nodes = self._get_connected_nodes(node_name, node_mapping)
            
            # SPICE 라인 생성
            if node_class == 'VoltageSource':
                if len(connected_nodes) >= 2:
                    spice_lines.append(f"V{comp_id} {connected_nodes[0]} {connected_nodes[1]} {value}")
                else:
                    spice_lines.append(f"V{comp_id} {connected_nodes[0] if connected_nodes else node_mapping[node_name]} 0 {value}")
                    
            elif node_class == 'Resistor':
                if len(connected_nodes) >= 2:
                    spice_lines.append(f"R{comp_id} {connected_nodes[0]} {connected_nodes[1]} {value}")
                else:
                    spice_lines.append(f"R{comp_id} {connected_nodes[0] if connected_nodes else node_mapping[node_name]} 0 {value}")
                    
            elif node_class == 'Capacitor':
                if len(connected_nodes) >= 2:
                    spice_lines.append(f"C{comp_id} {connected_nodes[0]} {connected_nodes[1]} {value}u")
                else:
                    spice_lines.append(f"C{comp_id} {connected_nodes[0] if connected_nodes else node_mapping[node_name]} 0 {value}u")
                    
            elif node_class == 'Diode':
                if len(connected_nodes) >= 2:
                    spice_lines.append(f"D{comp_id} {connected_nodes[0]} {connected_nodes[1]} DMOD")
                else:
                    spice_lines.append(f"D{comp_id} {connected_nodes[0] if connected_nodes else node_mapping[node_name]} 0 DMOD")
                    
            elif node_class == 'LED':
                if len(connected_nodes) >= 2:
                    spice_lines.append(f"D{comp_id} {connected_nodes[0]} {connected_nodes[1]} LEDMOD")
                else:
                    spice_lines.append(f"D{comp_id} {connected_nodes[0] if connected_nodes else node_mapping[node_name]} 0 LEDMOD")
                    
            elif node_class == 'IC':
                model = self.node_models.get(node_name, 'ua741')
                # IC의 경우 핀 연결 정보 포함
                ic_connections = []
                for u, v in self.graph.edges:
                    if u == node_name:
                        pin_info = self.edge_pins.get((u, v), ('', ''))
                        if pin_info[0]:
                            ic_connections.append(f"{node_mapping.get(v, v)}")
                    elif v == node_name:
                        pin_info = self.edge_pins.get((u, v), ('', ''))
                        if pin_info[1]:
                            ic_connections.append(f"{node_mapping.get(u, u)}")
                
                conn_str = " ".join(ic_connections) if ic_connections else f"{node_mapping[node_name]}"
                spice_lines.append(f"X{comp_id} {conn_str} {model}")
        
        # 구분선
        spice_lines.append("* ")
        
        # 모델 정의 (참고 파일 형식 따라)
        spice_lines.append(".MODEL DMOD D")
        spice_lines.append(".MODEL LEDMOD D(IS=1E-12 N=2)")
        spice_lines.append(".END")
        
        return "\n".join(spice_lines)
    
    def _create_node_mapping(self):
        """노드 이름을 숫자로 매핑"""
        node_mapping = {}
        node_counter = 1
        
        # V- 노드들은 0 (접지)으로 매핑
        for node_name, node_class in self.node_classes.items():
            if node_class == 'V-':
                node_mapping[node_name] = 0
        
        # 나머지 노드들은 순차적으로 번호 할당
        for node_name, node_class in self.node_classes.items():
            if node_name not in node_mapping:
                if node_class == 'Junction':
                    # Junction 노드는 연결된 실제 노드 번호 사용
                    node_mapping[node_name] = node_counter
                    node_counter += 1
                else:
                    node_mapping[node_name] = node_counter
                    node_counter += 1
        
        return node_mapping
    
    def _get_connected_nodes(self, node_name, node_mapping):
        """노드에 연결된 다른 노드들의 매핑된 번호 반환"""
        connected = []
        
        for u, v in self.graph.edges:
            if u == node_name:
                connected.append(node_mapping.get(v, v))
            elif v == node_name:
                connected.append(node_mapping.get(u, u))
        
        # 중복 제거 및 정렬
        return sorted(list(set(connected)))

    def _save_spice_netlist(self, filename):
        """SPICE netlist를 파일로 저장"""
        spice_content = self._generate_spice_netlist()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(spice_content)
        
        print(f"✅ SPICE netlist saved to: {filename}")

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 기존 노드 클릭 확인
            for name, (nx_, ny_) in self.node_positions.items():
                if (x - nx_) ** 2 + (y - ny_) ** 2 < 20 ** 2:
                    if self.edge_start is None:
                        self.edge_start = name
                    else:
                        src, tgt = self.edge_start, name
                        if src != tgt:  # 자기 자신에게는 연결 금지
                            # IC 핀 입력
                            src_pin = self._prompt_ic_pin(f"[{src}] 핀 번호 입력") if self.node_classes[src] == 'IC' else ''
                            tgt_pin = self._prompt_ic_pin(f"[{tgt}] 핀 번호 입력") if self.node_classes[tgt] == 'IC' else ''
                            
                            # 엣지 추가 (중복 방지)
                            if not self.graph.has_edge(src, tgt):
                                self.graph.add_edge(src, tgt)
                                self.edge_pins[(src, tgt)] = (src_pin, tgt_pin)
                                
                        self.edge_start = None
                    return
            
            # 새 노드 생성
            info = self._prompt_class_info()
            if not info[0] or info[0] in self.node_positions:
                return
                
            name, cls, val = info
            self.node_positions[name] = (x, y)
            self.node_classes[name] = cls
            self.node_values[name] = val
            self.graph.add_node(name, type=cls, value=val)
            
            if cls == 'IC':
                self.graph.nodes[name]['model'] = 'ua741'
                self.node_models[name] = 'ua741'
                
            self.next_node_id += 1
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.edge_start = None
            
        elif event == cv2.EVENT_MBUTTONDOWN:  # 마우스 휠 클릭
            # Junction 노드 배치 제안
            self._suggest_junction_placement()

    def draw_and_save(self, base_filename="drawn_circuit"):
        """회로를 그리고 GraphML과 SPICE netlist 형태로 저장"""
        cv2.namedWindow("Draw Circuit")
        cv2.setMouseCallback("Draw Circuit", self._on_mouse)

        print("=" * 60)
        print("🔧 Circuit Drawer Instructions:")
        print("▶ 좌클릭: 노드 생성 또는 엣지 연결")
        print("▶ 우클릭: 엣지 연결 취소") 
        print("▶ 휠클릭: 병렬 회로 자동 감지 및 Junction 배치")
        print("▶ 'q': 저장 및 종료")
        print("▶ 's': 병렬 회로 제안 확인")
        print("▶ 'n': SPICE netlist 미리보기")
        print("=" * 60)
        
        while True:
            img = self._draw_canvas()
            cv2.imshow("Draw Circuit", img)
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._suggest_junction_placement()
            elif key == ord('n'):
                # SPICE netlist 미리보기
                spice_preview = self._generate_spice_netlist()
                print("\n" + "="*50)
                print("🔍 SPICE NETLIST PREVIEW:")
                print("="*50)
                print(spice_preview)
                print("="*50)

        cv2.destroyAllWindows()
        
        # 파일 저장
        graphml_path = f"{base_filename}.graphml"
        spice_path = f"{base_filename}.spice"
        
        # GraphML 저장 시 엣지 핀 정보도 포함
        for (u, v), (p1, p2) in self.edge_pins.items():
            self.graph[u][v]['source_pin'] = str(p1)  
            self.graph[u][v]['target_pin'] = str(p2)
            
        # GraphML 저장
        write_graphml(self.graph, graphml_path)
        print(f"✅ GraphML saved to: {graphml_path}")
        
        # SPICE netlist 저장
        self._save_spice_netlist(spice_path)
        
        # 요약 정보 출력
        print("\n" + "="*50)
        print("📊 CIRCUIT SUMMARY:")
        print("="*50)
        print(f"Total nodes: {len(self.graph.nodes)}")
        print(f"Total edges: {len(self.graph.edges)}")
        print("Component count:")
        for comp_class, count in self._count_components().items():
            print(f"  - {comp_class}: {count}")
        print("="*50)

    def _count_components(self):
        """컴포넌트 개수 계산"""
        count = {}
        for node_class in self.node_classes.values():
            count[node_class] = count.get(node_class, 0) + 1
        return count

# 노드 추가 없이 병렬 회로 처리하는 헬퍼 함수
def detect_parallel_without_junctions(graph):
    """
    Junction 노드 없이 병렬 회로를 감지하는 함수
    같은 시작점과 끝점을 가진 경로들을 찾아 병렬로 간주
    """
    parallel_groups = []
    
    # 모든 노드 쌍에 대해 여러 경로가 있는지 확인
    for source in graph.nodes:
        for target in graph.nodes:
            if source != target:
                try:
                    # NetworkX를 사용해 모든 단순 경로 찾기
                    paths = list(nx.all_simple_paths(graph, source, target, cutoff=5))
                    if len(paths) > 1:
                        # 여러 경로가 있으면 병렬 회로로 간주
                        parallel_groups.append({
                            'start': source,
                            'end': target, 
                            'paths': paths,
                            'branches': len(paths)
                        })
                except nx.NetworkXNoPath:
                    continue
                    
    return parallel_groups

# 사용 예시
if __name__ == "__main__":
    # Junction 노드 포함 버전
    print("=== Junction 노드를 사용한 병렬 회로 그리기 ===")
    saver = CircuitSaver()
    saver.draw_and_save("circuit10_amplifier")
    
    # 병렬 회로 감지 (Junction 없이)
    print("\n=== Junction 없이 병렬 회로 감지 ===")
    parallel_circuits = detect_parallel_without_junctions(saver.graph)
    
    if parallel_circuits:
        print("감지된 병렬 회로:")
        for i, circuit in enumerate(parallel_circuits):
            print(f"{i+1}. {circuit['start']} -> {circuit['end']}: {circuit['branches']}개 경로")
            for j, path in enumerate(circuit['paths']):
                print(f"   경로 {j+1}: {' -> '.join(path)}")
    else:
        print("병렬 회로가 감지되지 않았습니다.")
    
    print("\n🎉 GraphML과 SPICE netlist 파일이 모두 생성되었습니다!")