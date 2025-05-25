# circuit_generator_manager.py
import cv2
import tkinter as tk
from tkinter import simpledialog
from circuit_generator import generate_circuit

class CircuitGeneratorManager:
    def __init__(self, hole_detector):
        self.hole_det = hole_detector
    
    def quick_value_input(self, component_pins):
        """개별 저항값 입력"""
        resistors = [(i, comp) for i, comp in enumerate(component_pins) if comp['class'] == 'Resistor']
        
        if not resistors:
            print("✅ 저항이 없습니다.")
            return
        
        print(f"📝 {len(resistors)}개 저항의 값을 입력하세요")
        
        root = tk.Tk()
        root.withdraw()
        
        for idx, (comp_idx, comp) in enumerate(resistors):
            # 현재 저항 정보 표시
            x1, y1, x2, y2 = comp['box']
            
            value = simpledialog.askfloat(
                f"저항값 입력 ({idx+1}/{len(resistors)})", 
                f"저항 R{idx+1} (위치: {x1},{y1}) 값을 입력하세요 (Ω):",
                initialvalue=100.0,
                minvalue=0.1
            )
            
            if value is not None:
                comp['value'] = value
                print(f"✅ R{idx+1}: {value}Ω")
            else:
                print(f"⚠️ R{idx+1}: 기본값 100Ω 사용")
                comp['value'] = 100.0
        
        root.destroy()
        print(f"✅ 모든 저항값 입력 완료")

    def quick_power_selection(self, warped, component_pins):
        """간단한 전원 선택"""
        print("⚡ 전원 단자를 선택하세요")
        print("- 첫 번째 클릭: 양극(+)")
        print("- 두 번째 클릭: 음극(-)")
        
        # 모든 핀 위치 수집
        all_endpoints = [pt for comp in component_pins for pt in comp['pins']]
        
        # 전원 전압 입력
        root = tk.Tk()
        root.withdraw()
        voltage = simpledialog.askfloat("전원 전압", "전원 전압을 입력하세요 (V):", initialvalue=5.0)
        root.destroy()
        
        if not voltage:
            voltage = 5.0
        
        # 클릭으로 전원 단자 선택
        selected_points = []
        
        def on_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 2:
                selected_points.append((x, y))
                # 가장 가까운 실제 핀 찾기
                closest = min(all_endpoints, key=lambda p: (p[0]-x)**2 + (p[1]-y)**2)
                cv2.circle(warped, closest, 8, (0, 0, 255), -1)
                cv2.putText(warped, f"{'+'if len(selected_points)==1 else '-'}", 
                           (closest[0]+10, closest[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('전원 선택', warped)
        
        # 전원 선택을 위한 시각화 이미지 준비
        power_img = warped.copy()
        
        # 모든 핀을 표시
        for comp in component_pins:
            for px, py in comp['pins']:
                cv2.circle(power_img, (int(px), int(py)), 4, (0, 255, 0), -1)
        
        cv2.imshow('전원 선택', power_img)
        cv2.setMouseCallback('전원 선택', on_click)
        
        while len(selected_points) < 2:
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        
        if len(selected_points) == 2:
            # 가장 가까운 실제 핀들 찾기
            plus_pt = min(all_endpoints, key=lambda p: (p[0]-selected_points[0][0])**2 + (p[1]-selected_points[0][1])**2)
            minus_pt = min(all_endpoints, key=lambda p: (p[0]-selected_points[1][0])**2 + (p[1]-selected_points[1][1])**2)
            
            print(f"✅ 전원 설정: +{plus_pt}, -{minus_pt}, {voltage}V")
            return voltage, plus_pt, minus_pt
        else:
            # 기본값 사용
            print("⚠️ 전원 선택 실패, 기본값 사용")
            return voltage, all_endpoints[0], all_endpoints[-1]

    def generate_final_circuit(self, component_pins, holes, voltage, plus_pt, minus_pt, warped):
        """최종 회로 생성"""
        print("🔄 회로도 생성 중...")
        
        try:
            # hole_to_net 맵 생성
            nets, row_nets = self.hole_det.get_board_nets(holes, base_img=warped, show=False)
            hole_to_net = {}
            for row_idx, clusters in row_nets:
                for entry in clusters:
                    net_id = entry['net_id']
                    for x, y in entry['pts']:
                        hole_to_net[(int(round(x)), int(round(y)))] = net_id
            
            # nearest_net 함수 정의
            def nearest_net(pt):
                closest = min(hole_to_net.keys(), key=lambda h: (h[0]-pt[0])**2 + (h[1]-pt[1])**2)
                return hole_to_net[closest]
            
            # 전원 매핑
            net_plus = nearest_net(plus_pt)
            net_minus = nearest_net(minus_pt)
            
            # 와이어 연결 처리
            wires = []
            for comp in component_pins:
                if comp['class'] == 'Line_area' and len(comp['pins']) == 2:
                    net1 = nearest_net(comp['pins'][0])
                    net2 = nearest_net(comp['pins'][1])
                    if net1 != net2:
                        wires.append((net1, net2))
            
            # schemdraw 그리드 좌표 변환
            img_w = warped.shape[1]
            comp_count = len([c for c in component_pins if c['class'] != 'Line_area'])
            grid_width = comp_count * 2 + 2
            x_plus_grid = plus_pt[0] / img_w * grid_width
            x_minus_grid = minus_pt[0] / img_w * grid_width
            
            power_pairs = [(net_plus, x_plus_grid, net_minus, x_minus_grid)]
            
            # 회로 생성
            mapped, final_hole_to_net = generate_circuit(
                all_comps=component_pins,
                holes=holes,
                wires=wires,
                voltage=voltage,
                output_spice='circuit.spice',
                output_img='circuit.jpg',
                hole_to_net=hole_to_net,
                power_pairs=power_pairs
            )
            
            print("✅ 회로도 생성 완료!")
            print("📁 생성된 파일:")
            print("  - circuit.jpg (회로도)")
            print("  - circuit.spice (SPICE 넷리스트)")
            
            return True
            
        except Exception as e:
            print(f"❌ 회로 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False