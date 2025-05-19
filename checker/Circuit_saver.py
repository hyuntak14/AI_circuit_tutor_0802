# checker/Circuit_saver.py (개선된 버전)
import cv2
import numpy as np
import networkx as nx
from networkx.readwrite import write_graphml
import tkinter as tk
from tkinter import simpledialog, messagebox

class CircuitSaver:
    def __init__(self, canvas_size=(800, 600)):
        self.graph = nx.DiGraph()  # 방향성 그래프
        self.canvas_size = canvas_size
        self.node_positions = {}    # {node_name: (x, y)}
        self.node_classes = {}      # {node_name: class}
        self.node_models = {}       # {node_name: model} (for IC)
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
            self.graph.add_node(j_pos_name, type='Junction', value=0.0)
            
            self.node_positions[j_neg_name] = (jx_neg, jy_neg)
            self.node_classes[j_neg_name] = 'Junction'  
            self.graph.add_node(j_neg_name, type='Junction', value=0.0)
            
            # 연결 생성 (전원의 양극과 음극 구분 필요)
            # 간단화: 첫 번째 노드를 양극, 두 번째를 음극으로 가정
            self.graph.add_edge(n1, j_pos_name)
            self.graph.add_edge(j_pos_name, n2)
            self.graph.add_edge(n1, j_neg_name)  
            self.graph.add_edge(j_neg_name, n2)
            
            self.next_node_id += 2

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

    def draw_and_save(self, graphml_path="drawn_circuit.graphml"):
        cv2.namedWindow("Draw Circuit")
        cv2.setMouseCallback("Draw Circuit", self._on_mouse)

        print("=" * 60)
        print("🔧 Circuit Drawer Instructions:")
        print("▶ 좌클릭: 노드 생성 또는 엣지 연결")
        print("▶ 우클릭: 엣지 연결 취소") 
        print("▶ 휠클릭: 병렬 회로 자동 감지 및 Junction 배치")
        print("▶ 'q': 저장 및 종료")
        print("▶ 's': 병렬 회로 제안 확인")
        print("=" * 60)
        
        while True:
            img = self._draw_canvas()
            cv2.imshow("Draw Circuit", img)
            key = cv2.waitKey(50) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self._suggest_junction_placement()

        cv2.destroyAllWindows()
        
        # GraphML 저장 시 엣지 핀 정보도 포함
        for (u, v), (p1, p2) in self.edge_pins.items():
            self.graph[u][v]['source_pin'] = str(p1)  
            self.graph[u][v]['target_pin'] = str(p2)
            
        write_graphml(self.graph, graphml_path)
        print(f"✅ GraphML saved to: {graphml_path}")

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
    saver.draw_and_save("circuit8_rectification.graphml")
    
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