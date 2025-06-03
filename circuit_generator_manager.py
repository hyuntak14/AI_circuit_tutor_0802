# circuit_generator_manager.py (다중 전원 지원 수정된 버전)
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
        """다중 전원 선택 - 여러 개의 전원을 입력받을 수 있도록 수정"""
        print("⚡ 전원 단자들을 선택하세요")
        print("- 각 전원마다 양극(+)과 음극(-)을 순서대로 클릭")
        print("- ESC 키를 누르면 전원 추가 중단")
        
        # 모든 핀 위치 수집
        all_endpoints = [pt for comp in component_pins for pt in comp['pins']]
        
        if not all_endpoints:
            print("❌ 핀이 없습니다. 기본값을 사용합니다.")
            return [(5.0, (100, 100), (200, 200))]
        
        # 전원 리스트를 저장할 변수
        power_sources = []
        
        # 첫 번째 전원은 필수
        print("\n=== 첫 번째 전원 설정 ===")
        first_power = self._select_single_power_source(warped, all_endpoints, 1, component_pins)
        if first_power:
            power_sources.append(first_power)
        else:
            # 기본값 사용
            power_sources.append((5.0, all_endpoints[0], all_endpoints[-1]))
        
        # 추가 전원 입력 여부 확인
        while True:
            root = tk.Tk()
            root.withdraw()
            
            # 현재까지 입력된 전원 개수 표시
            current_count = len(power_sources)
            add_more = messagebox.askyesno(
                "추가 전원 입력", 
                f"현재 {current_count}개의 전원이 설정되었습니다.\n\n추가 전원을 입력하시겠습니까?"
            )
            root.destroy()
            
            if not add_more:
                break
            
            # 추가 전원 입력
            power_num = len(power_sources) + 1
            print(f"\n=== {power_num}번째 전원 설정 ===")
            additional_power = self._select_single_power_source(warped, all_endpoints, power_num, component_pins)
            
            if additional_power:
                power_sources.append(additional_power)
                print(f"✅ {power_num}번째 전원 추가됨")
            else:
                print(f"⚠️ {power_num}번째 전원 입력 취소됨")
                break
        
        print(f"\n✅ 총 {len(power_sources)}개의 전원이 설정되었습니다:")
        for i, (voltage, plus_pt, minus_pt) in enumerate(power_sources, 1):
            print(f"  전원 {i}: {voltage}V, +{plus_pt}, -{minus_pt}")
        
        return power_sources

    def _select_single_power_source(self, warped, all_endpoints, power_num, component_pins):
        """단일 전원 선택 (내부 함수)"""
        # 전원 전압 입력
        root = tk.Tk()
        root.withdraw()
        voltage = simpledialog.askfloat(
            f"전원 {power_num} 전압", 
            f"전원 {power_num}의 전압을 입력하세요 (V):", 
            initialvalue=5.0
        )
        root.destroy()
        
        if voltage is None:
            return None
        
        # 클릭으로 전원 단자 선택
        selected_points = []
        power_img = warped.copy()
        
        def on_click(event, x, y, flags, param):
            nonlocal selected_points, power_img
            if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 2:
                selected_points.append((x, y))
                # 가장 가까운 실제 핀 찾기
                closest = min(all_endpoints, key=lambda p: (p[0]-x)**2 + (p[1]-y)**2)
                
                if len(selected_points) == 1:
                    cv2.circle(power_img, closest, 8, (0, 0, 255), -1)
                    cv2.putText(power_img, f"V{power_num}+", 
                               (closest[0]+10, closest[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.circle(power_img, closest, 8, (255, 0, 0), -1)
                    cv2.putText(power_img, f"V{power_num}-", 
                               (closest[0]+10, closest[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                cv2.imshow(f'Select Power {power_num}', power_img)
        
        # 전원 선택을 위한 시각화 이미지 준비
        power_img = warped.copy()
        
        # 모든 핀을 표시
        for comp in component_pins:
            for px, py in comp['pins']:
                cv2.circle(power_img, (int(px), int(py)), 4, (0, 255, 0), -1)
        
        cv2.putText(power_img, f"Power {power_num}: Click '+' first, then '-' (ESC to cancel)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(f'Select Power {power_num}', power_img)
        cv2.setMouseCallback(f'Select Power {power_num}', on_click)
        
        while len(selected_points) < 2:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                return None
        
        cv2.destroyAllWindows()
        
        if len(selected_points) == 2:
            # 가장 가까운 실제 핀들 찾기
            plus_pt = min(all_endpoints, key=lambda p: (p[0]-selected_points[0][0])**2 + (p[1]-selected_points[0][1])**2)
            minus_pt = min(all_endpoints, key=lambda p: (p[0]-selected_points[1][0])**2 + (p[1]-selected_points[1][1])**2)
            
            return (voltage, plus_pt, minus_pt)
        else:
            return None

    def generate_final_circuit(self, component_pins, holes, power_sources, warped):
        """최종 회로 생성 - 다중 전원 지원"""
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
            
            # Union-Find 초기화
            parent = {net: net for net in set(hole_to_net.values())}
            def find(u):
                if parent[u] != u:
                    parent[u] = find(parent[u])
                return parent[u]
            def union(u, v):
                pu, pv = find(u), find(v)
                if pu != pv:
                    parent[pv] = pu
            
            # nearest_net 함수 정의 (Union-Find 적용)
            def nearest_net(pt):
                if not hole_to_net:
                    print("⚠️ 경고: hole_to_net이 비어있습니다!")
                    return 0
                closest = min(hole_to_net.keys(), key=lambda h: (h[0]-pt[0])**2 + (h[1]-pt[1])**2)
                return find(hole_to_net[closest])
            
            # 와이어 연결 처리
            wires = []
            for comp in component_pins:
                if comp['class'] == 'Line_area' and len(comp['pins']) == 2:
                    net1 = nearest_net(comp['pins'][0])
                    net2 = nearest_net(comp['pins'][1])
                    if net1 != net2:
                        wires.append((net1, net2))
                        union(net1, net2)
            
            # 다중 전원 매핑
            power_pairs = []
            img_w = warped.shape[1]
            comp_count = len([c for c in component_pins if c['class'] != 'Line_area'])
            grid_width = comp_count * 2 + 2
            
            for voltage, plus_pt, minus_pt in power_sources:
                net_plus = nearest_net(plus_pt)
                net_minus = nearest_net(minus_pt)
                
                # schemdraw 그리드 좌표 변환
                x_plus_grid = plus_pt[0] / img_w * grid_width
                x_minus_grid = minus_pt[0] / img_w * grid_width
                
                power_pairs.append((net_plus, x_plus_grid, net_minus, x_minus_grid))
            
            print(f"📊 회로 정보:")
            print(f"  - 컴포넌트: {len([c for c in component_pins if c['class'] != 'Line_area'])}개")
            print(f"  - 전원: {len(power_pairs)}개")
            print(f"  - 와이어: {len(wires)}개")
            
            # 🔧 1단계: generate_circuit 실행
            print("🔄 회로 생성 실행 중...")
            
            # 첫 번째 전원의 전압을 대표 전압으로 사용 (기존 인터페이스 호환성을 위해)
            representative_voltage = power_sources[0][0] if power_sources else 5.0
            
            components, final_hole_to_net = generate_circuit(
                all_comps=component_pins,
                holes=holes,
                wires=wires,
                voltage=representative_voltage,
                output_spice='circuit.spice',
                output_img='circuit.jpg',
                hole_to_net=hole_to_net,
                power_pairs=power_pairs
            )
            
            # 🔧 2단계: 오류 검사 (다중 전원 고려)
            print("🔍 회로 오류 검사 중...")
            error_result = self._check_circuit_errors_multi_power(
                components, power_pairs, power_sources
            )
            
            if not error_result:
                print("❌ 사용자가 회로도 생성을 취소했습니다.")
                return False
            
            print("✅ 회로도 생성 완료!")
            print("📁 생성된 파일:")
            print("  - circuit.jpg (회로도)")
            print("  - circuit.spice (SPICE 넷리스트)")
            
            # 다중 전원 정보 출력
            for i, (voltage, _, _) in enumerate(power_sources, 1):
                if i > 1:
                    print(f"  - circuit_pwr{i}.jpg (전원 {i} 회로도)")
            
            return True
            
        except Exception as e:
            print(f"❌ 회로 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _check_circuit_errors_multi_power(self, components, power_pairs, power_sources):
        """다중 전원을 고려한 오류 검사"""
        try:
            # 🔧 중복 방지: components 복사본으로 작업
            components_for_check = components.copy()
            
            # 1. 기존 전압원 확인
            existing_voltage_sources = [comp for comp in components_for_check if comp['class'] == 'VoltageSource']
            print(f"🔍 기존 전압원: {len(existing_voltage_sources)}개")
            
            # 2. nets_mapping 생성
            nets_mapping = {}
            for comp in components_for_check:
                n1, n2 = comp['nodes']
                nets_mapping.setdefault(n1, []).append(comp['name'])
                nets_mapping.setdefault(n2, []).append(comp['name'])
            
            # 🔧 3. 전압원이 부족한 경우에만 추가 (다중 전원 대응)
            expected_voltage_sources = len(power_sources)
            if len(existing_voltage_sources) < expected_voltage_sources:
                print(f"⚠️ 전압원이 부족합니다. {expected_voltage_sources - len(existing_voltage_sources)}개 추가합니다.")
                
                # 부족한 만큼 전압원 추가
                for i in range(len(existing_voltage_sources), expected_voltage_sources):
                    voltage, _, _ = power_sources[i]
                    net_p, _, net_m, _ = power_pairs[i]
                    
                    vs_name = f"V{i+1}"
                    vs_comp = {
                        'name': vs_name,
                        'class': 'VoltageSource',
                        'value': voltage,
                        'nodes': (net_p, net_m)
                    }
                    components_for_check.append(vs_comp)
                    nets_mapping.setdefault(net_p, []).append(vs_name)
                    nets_mapping.setdefault(net_m, []).append(vs_name)
            else:
                print("✅ 전압원이 충분히 존재합니다.")
            
            # 4. ground_net 설정 (첫 번째 전원의 minus 단자)
            ground_net = power_pairs[0][2] if power_pairs else 0
            
            print(f"🔍 ErrorChecker 데이터:")
            print(f"  - 컴포넌트 수: {len(components_for_check)}")
            print(f"  - 전압원 수: {len([c for c in components_for_check if c['class'] == 'VoltageSource'])}")
            print(f"  - 넷 수: {len(nets_mapping)}")
            print(f"  - Ground 넷: {ground_net}")
            
            # 🔧 중복 컴포넌트 확인 및 제거
            comp_names = [comp['name'] for comp in components_for_check]
            duplicates = [name for name in set(comp_names) if comp_names.count(name) > 1]
            if duplicates:
                print(f"⚠️ 중복된 컴포넌트 이름: {duplicates}")
                seen_names = set()
                unique_components = []
                for comp in components_for_check:
                    if comp['name'] not in seen_names:
                        unique_components.append(comp)
                        seen_names.add(comp['name'])
                    else:
                        print(f"  - 중복 제거: {comp['name']} ({comp['class']})")
                components_for_check = unique_components
                
                # nets_mapping 재생성
                nets_mapping = {}
                for comp in components_for_check:
                    n1, n2 = comp['nodes']
                    nets_mapping.setdefault(n1, []).append(comp['name'])
                    nets_mapping.setdefault(n2, []).append(comp['name'])
            
            # 5. ErrorChecker 실행
            checker = ErrorChecker(components_for_check, nets_mapping, ground_nodes={ground_net})
            errors = checker.run_all_checks()
            
            # 6. 결과 처리
            if errors:
                print(f"⚠️ {len(errors)}개의 회로 오류가 발견되었습니다:")
                for i, error in enumerate(errors, 1):
                    print(f"  {i}. {error}")
                
                # 사용자에게 오류 알림 및 선택권 제공
                root = tk.Tk()
                root.withdraw()
                
                error_msg = f"다음 {len(errors)}개의 회로 오류가 발견되었습니다:\n\n"
                for i, error in enumerate(errors[:5], 1):
                    error_msg += f"{i}. {error}\n"
                
                if len(errors) > 5:
                    error_msg += f"\n... 및 {len(errors) - 5}개 추가 오류\n"
                
                error_msg += "\n그래도 회로도를 생성하시겠습니까?"
                
                result = messagebox.askyesno("회로 오류 발견", error_msg)
                root.destroy()
                
                if not result:
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
            return True