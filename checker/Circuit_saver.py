import cv2
import numpy as np
import networkx as nx
from networkx.readwrite import write_graphml
import tkinter as tk
from tkinter import simpledialog

class CircuitSaver:
    def __init__(self, canvas_size=(800, 600)):
        self.graph = nx.DiGraph()  # 방향성 그래프 (핀 연결 방향)
        self.canvas_size = canvas_size
        self.node_positions = {}    # {node_name: (x, y)}
        self.node_classes = {}      # {node_name: class}
        self.node_models = {}       # {node_name: model} (for IC)
        self.edge_pins = {}         # {(src, tgt): (src_pin, tgt_pin)}
        self.next_node_id = 1
        self.edge_start = None
        self.canvas = 255 * np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
        self.class_options = ['VoltageSource','V-','Resistor', 'Capacitor', 'Diode', 'LED', 'IC']

        self.ua741_pins = {
            1: 'Offset Null', 2: 'Inverting', 3: 'Non-inverting',
            4: 'V-', 5: 'Offset Null', 6: 'Output', 7: 'V+', 8: 'NC'
        }

    def _draw_canvas(self):
        img = self.canvas.copy()
        for name, (x, y) in self.node_positions.items():
            cls = self.node_classes.get(name, '?')
            model = self.node_models.get(name, '')
            cv2.circle(img, (x, y), 15, (0, 200, 255), -1)
            cv2.putText(img, name, (x - 10, y - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(img, cls, (x - 15, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 0, 0), 1)
            if model:
                cv2.putText(img, model, (x - 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        for u, v in self.graph.edges:
            x1, y1 = self.node_positions[u]
            x2, y2 = self.node_positions[v]
            pin_label = self.edge_pins.get((u, v), ('', ''))
            label = f"{pin_label[0]}->{pin_label[1]}"
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
            mx, my = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.putText(img, label, (mx, my - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        return img

    def _prompt_class_info(self):
        root = tk.Tk()
        root.withdraw()
        idx = simpledialog.askinteger("클래스 선택", "\n".join(f"{i+1}. {c}" for i, c in enumerate(self.class_options)), minvalue=1, maxvalue=len(self.class_options))
        if not idx:
            root.destroy()
            return None, None, None
        comp_class = self.class_options[idx - 1]
        default_name = f"{comp_class[0]}{self.next_node_id}"
        name = simpledialog.askstring("노드 이름", "노드 이름을 입력하세요:", initialvalue=default_name)
        val = simpledialog.askfloat("값 입력", f"{comp_class} 값 입력:", minvalue=0.0) if comp_class in ['Resistor', 'Capacitor'] else 0.0
        root.destroy()
        return name, comp_class, val

    def _prompt_ic_pin(self, message):
        root = tk.Tk()
        root.withdraw()
        pin = simpledialog.askinteger("핀 선택", message + "\n(1~8):", minvalue=1, maxvalue=8)
        root.destroy()
        return pin

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for name, (nx_, ny_) in self.node_positions.items():
                if (x - nx_) ** 2 + (y - ny_) ** 2 < 20 ** 2:
                    if self.edge_start is None:
                        self.edge_start = name
                    else:
                        src, tgt = self.edge_start, name
                        src_pin = self._prompt_ic_pin(f"[{src}] 핀 번호 입력") if self.node_classes[src] == 'IC' else ''
                        tgt_pin = self._prompt_ic_pin(f"[{tgt}] 핀 번호 입력") if self.node_classes[tgt] == 'IC' else ''
                        self.graph.add_edge(src, tgt)
                        self.edge_pins[(src, tgt)] = (src_pin, tgt_pin)
                        self.edge_start = None
                    return
            # 새 노드 생성
            name, cls, val = self._prompt_class_info()
            if not name or name in self.node_positions:
                return
            self.node_positions[name] = (x, y)
            self.node_classes[name] = cls
            self.graph.add_node(name, type=cls, value=val)
            if cls == 'IC':
                self.graph.nodes[name]['model'] = 'ua741'
                self.node_models[name] = 'ua741'
            self.next_node_id += 1

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.edge_start = None

    def draw_and_save(self, graphml_path="drawn_circuit.graphml"):
        cv2.namedWindow("Draw Circuit")
        cv2.setMouseCallback("Draw Circuit", self._on_mouse)

        print("▶ 좌클릭: 노드 생성 또는 엣지 연결\n▶ 우클릭: 엣지 연결 취소\n▶ 'q': 저장 및 종료")
        while True:
            img = self._draw_canvas()
            cv2.imshow("Draw Circuit", img)
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        for (u, v), (p1, p2) in self.edge_pins.items():
            self.graph[u][v]['source_pin'] = p1
            self.graph[u][v]['target_pin'] = p2
        write_graphml(self.graph, graphml_path)
        print(f"[Saved] GraphML written to: {graphml_path}")




if __name__ == "__main__":
    saver = CircuitSaver()
    saver.draw_and_save("circuit_test.graphml")

