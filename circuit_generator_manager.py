# circuit_generator_manager.py (수정된 버전)
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
from circuit_generator import generate_circuit
from checker.error_checker import ErrorChecker

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
        
        if not all_endpoints:
            print("❌ 핀이 없습니다. 기본값을 사용합니다.")
            return 5.0, (100, 100), (200, 200)
        
        # 전원 전압 입력
        root = tk.Tk()
        root.withdraw()
        voltage = simpledialog.askfloat("전원 전압", "전원 전압을 입력하세요 (V):", initialvalue=5.0)
        root.destroy()
        
        if not voltage:
            voltage = 5.0
        
        # 클릭으로 전원 단자 선택
        selected_points = []
        power_img = warped.copy()
        
        def on_click(event, x, y, flags, param):
            nonlocal selected_points, power_img
            if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 2:
                selected_points.append((x, y))
                # 가장 가까운 실제 핀 찾기
                closest = min(all_endpoints, key=lambda p: (p[0]-x)**2 + (p[1]-y)**2)
                cv2.circle(power_img, closest, 8, (0, 0, 255), -1)
                cv2.putText(power_img, f"{'+'if len(selected_points)==1 else '-'}", 
                           (closest[0]+10, closest[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Select Power', power_img)
        
        # 전원 선택을 위한 시각화 이미지 준비
        power_img = warped.copy()
        
        # 모든 핀을 표시
        for comp in component_pins:
            for px, py in comp['pins']:
                cv2.circle(power_img, (int(px), int(py)), 4, (0, 255, 0), -1)
        
        cv2.putText(power_img, "Click '+' terminal first, then '-' terminal", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Select Power', power_img)
        cv2.setMouseCallback('Select Power', on_click)
        
        while len(selected_points) < 2:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
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
        """최종 회로 생성 (오류 검출 포함)"""
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
                if not hole_to_net:
                    print("⚠️ 경고: hole_to_net이 비어있습니다!")
                    return 0
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
            
            # 🔧 오류 검출을 위한 컴포넌트 매핑 생성 (수정된 버전)
            print("🔍 회로 오류 검사 중...")
            
            # Union-Find 구조로 와이어 병합 처리
            parent = {net: net for net in set(hole_to_net.values())}
            
            def find(u):
                if parent[u] != u:
                    parent[u] = find(parent[u])
                return parent[u]
            
            def union(u, v):
                pu, pv = find(u), find(v)
                if pu != pv:
                    parent[pv] = pu
            
            # 와이어 연결로 넷 병합
            for net1, net2 in wires:
                union(net1, net2)
            
            # 병합된 넷으로 전원 재매핑
            final_net_plus = find(net_plus)
            final_net_minus = find(net_minus)
            
            # 오류 검사용 컴포넌트 매핑 생성
            mapped_components = self._create_component_mapping_fixed(
                component_pins, hole_to_net, nearest_net, voltage, 
                final_net_plus, final_net_minus, wires, find
            )
            
            # 오류 검출 수행
            error_result = self._check_circuit_errors(mapped_components, final_net_minus)
            
            # 오류가 있으면 회로도 생성 중단 여부 결정
            if not error_result:
                print("❌ 회로 오류로 인해 회로도 생성을 중단합니다.")
                return False
            
            # 오류가 없거나 사용자가 계속하기로 했으면 실제 회로도 생성 진행
            print("✅ 회로 오류 검사 통과! 회로도 생성을 계속합니다...")
            
            # schemdraw 그리드 좌표 변환
            img_w = warped.shape[1]
            comp_count = len([c for c in component_pins if c['class'] != 'Line_area'])
            grid_width = comp_count * 2 + 2
            x_plus_grid = plus_pt[0] / img_w * grid_width
            x_minus_grid = minus_pt[0] / img_w * grid_width
            
            power_pairs = [(final_net_plus, x_plus_grid, final_net_minus, x_minus_grid)]
            
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
    
    def _create_component_mapping_fixed(self, component_pins, hole_to_net, nearest_net, 
                                      voltage, net_plus, net_minus, wires, find_func):
        """오류 검출을 위한 개선된 컴포넌트 매핑 생성"""
        mapped_components = []
        
        print(f"🔍 컴포넌트 매핑 생성 중...")
        print(f"  - 전원 넷: +{net_plus}, -{net_minus}")
        print(f"  - 와이어 연결: {len(wires)}개")
        
        # 일반 컴포넌트들 매핑
        comp_index = 1
        for i, comp in enumerate(component_pins):
            if comp['class'] == 'Line_area':  # 와이어는 제외
                continue
                
            # 핀 검증
            if len(comp['pins']) < 2:
                print(f"⚠️ 경고: {comp['class']} #{i+1}의 핀이 부족합니다 ({len(comp['pins'])}개)")
                continue
            
            try:
                # 핀을 넷에 매핑 (Union-Find로 병합된 넷 사용)
                raw_net1 = nearest_net(comp['pins'][0])
                raw_net2 = nearest_net(comp['pins'][1])
                node1 = find_func(raw_net1)
                node2 = find_func(raw_net2)
                
                # 클래스에 따른 컴포넌트 이름 생성
                class_prefixes = {
                    'Resistor': 'R', 'Diode': 'D', 'LED': 'L', 
                    'Capacitor': 'C', 'IC': 'U'
                }
                prefix = class_prefixes.get(comp['class'], 'X')
                comp_name = f"{prefix}{comp_index}"
                comp_index += 1
                
                mapped_comp = {
                    'name': comp_name,
                    'class': comp['class'],
                    'value': comp.get('value', 0),
                    'nodes': (node1, node2)
                }
                
                mapped_components.append(mapped_comp)
                
                print(f"  ✓ {comp_name} ({comp['class']}): Net{node1} - Net{node2} (값: {comp.get('value', 0)})")
                
            except Exception as e:
                print(f"⚠️ 경고: {comp['class']} #{i+1} 매핑 실패: {e}")
                continue
        
        # 전압원 추가
        vs_comp = {
            'name': 'V1',
            'class': 'VoltageSource', 
            'value': voltage,
            'nodes': (net_plus, net_minus)
        }
        mapped_components.append(vs_comp)
        print(f"  ✓ V1 (VoltageSource): Net{net_plus} - Net{net_minus} (값: {voltage}V)")
        
        print(f"✅ 총 {len(mapped_components)}개 컴포넌트 매핑 완료")
        return mapped_components
    
    def _check_circuit_errors(self, components, ground_net):
        """회로 오류 검출 및 사용자에게 알림"""
        try:
            print(f"🔍 ErrorChecker 실행 중...")
            print(f"  - 컴포넌트 수: {len(components)}")
            print(f"  - 접지 넷: {ground_net}")
            
            # nets_mapping 생성 (개선된 버전)
            nets_mapping = {}
            all_nets = set()
            
            for comp in components:
                n1, n2 = comp['nodes']
                all_nets.add(n1)
                all_nets.add(n2)
                nets_mapping.setdefault(n1, []).append(comp['name'])
                nets_mapping.setdefault(n2, []).append(comp['name'])
            
            print(f"  - 넷 수: {len(all_nets)}")
            print(f"  - 넷 매핑: {dict(list(nets_mapping.items())[:3])}..." if nets_mapping else "  - 넷 매핑: 없음")
            
            # ErrorChecker 실행
            checker = ErrorChecker(components, nets_mapping, ground_nodes={ground_net})
            errors = checker.run_all_checks()
            
            # 결과 처리
            if errors:
                print(f"⚠️ {len(errors)}개의 회로 오류가 발견되었습니다:")
                for i, error in enumerate(errors, 1):
                    print(f"  {i}. {error}")
                
                # 사용자에게 오류 알림 및 선택권 제공
                root = tk.Tk()
                root.withdraw()
                
                error_msg = f"다음 {len(errors)}개의 회로 오류가 발견되었습니다:\n\n"
                for i, error in enumerate(errors[:5], 1):  # 최대 5개만 표시
                    error_msg += f"{i}. {error}\n"
                
                if len(errors) > 5:
                    error_msg += f"\n... 및 {len(errors) - 5}개 추가 오류\n"
                
                error_msg += "\n그래도 회로도를 생성하시겠습니까?"
                
                # 사용자 선택
                result = messagebox.askyesno("회로 오류 발견", error_msg)
                root.destroy()
                
                if not result:
                    print("❌ 사용자가 회로도 생성을 취소했습니다.")
                    return False
                else:
                    print("⚠️ 사용자가 오류를 무시하고 회로도 생성을 계속하기로 했습니다.")
                    return True
                    
            else:
                print("✅ 회로 오류가 발견되지 않았습니다!")
                return True
                
        except Exception as e:
            print(f"⚠️ 오류 검사 중 문제가 발생했습니다: {e}")
            import traceback
            traceback.print_exc()
            print("회로도 생성을 계속합니다...")
            return True  # 오류 검사 실패 시에도 회로도 생성은 계속