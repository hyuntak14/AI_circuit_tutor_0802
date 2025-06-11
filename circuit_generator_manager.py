# circuit_generator_manager.py (다중 전원 지원 수정된 버전)
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
from circuit_generator import generate_circuit
from checker.error_checker import ErrorChecker


class CircuitGeneratorManager:
    def __init__(self, hole_detector):
        self.hole_det = hole_detector
    
    def quick_value_input(self, warped, component_pins):
        """개별 저항 및 캐패시터값 입력 (이미지에 번호 표시)"""
        # 1) 이미지 복사본 생성
        annotated = warped.copy()

        # 2) 부품 분류: 저항(R)과 캐패시터(C)만 추출
        resistors   = [(i, comp) for i, comp in enumerate(component_pins) if comp['class'] == 'Resistor']
        capacitors  = [(i, comp) for i, comp in enumerate(component_pins) if comp['class'] == 'Capacitor']

        # 3) 번호(label) 그리기: 저항은 파란색, 캐패시터는 초록색(예시)
        for idx, (comp_idx, comp) in enumerate(resistors, start=1):
            x1, y1, x2, y2 = comp['box']
            label = f"R{idx}"
            # 박스 좌상단에 텍스트 표시
            cv2.putText(annotated, label, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        for idx, (comp_idx, comp) in enumerate(capacitors, start=1):
            x1, y1, x2, y2 = comp['box']
            label = f"C{idx}"
            cv2.putText(annotated, label, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 4) 화면에 띄워 사용자가 확인할 수 있도록 함
        cv2.imshow("check components number (ESC : close)", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 5) Tkinter 다이얼로그로 순차 입력
        root = tk.Tk()
        root.withdraw()

        # --- 저항값 입력 ---
        if not resistors and not capacitors:
            print("✅ 입력할 저항/캐패시터가 없습니다.")
            root.destroy()
            return

        # 5-1) 저항 입력
        if resistors:
            print(f"📝 {len(resistors)}개 저항의 값을 입력하세요")
            for idx, (comp_idx, comp) in enumerate(resistors, start=1):
                x1, y1, x2, y2 = comp['box']
                prompt = simpledialog.askfloat(
                    f"저항값 R{idx} 입력 ({idx}/{len(resistors)})",
                    f"R{idx} (위치: {x1},{y1}) 값을 입력하세요 (Ω):",
                    initialvalue=100.0,
                    minvalue=0.1
                )
                if prompt is not None:
                    comp['value'] = prompt
                    print(f"✅ R{idx}: {prompt}Ω")
                else:
                    comp['value'] = 100.0
                    print(f"⚠️ R{idx}: 입력 없음 → 기본값 100Ω 사용")

        # 5-2) 캐패시터 입력
        if capacitors:
            print(f"📝 {len(capacitors)}개 캐패시터의 값을 입력하세요")
            for idx, (comp_idx, comp) in enumerate(capacitors, start=1):
                x1, y1, x2, y2 = comp['box']
                prompt = simpledialog.askfloat(
                    f"캐패시터값 C{idx} 입력 ({idx}/{len(capacitors)})",
                    f"C{idx} (위치: {x1},{y1}) 값을 입력하세요 (μF):",
                    initialvalue=1.0,
                    minvalue=0.0001
                )
                if prompt is not None:
                    comp['value'] = prompt
                    print(f"✅ C{idx}: {prompt}μF")
                else:
                    comp['value'] = 1.0
                    print(f"⚠️ C{idx}: 입력 없음 → 기본값 1μF 사용")

        root.destroy()
        print("✅ 모든 저항/캐패시터 값 입력 완료")


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


    # circuit_generator_manager.py 간단한 수정 - 하나의 회로도에 다중 전원 표시

# circuit_generator_manager.py 핵심 수정 - 넷 병합 완전 해결

    def generate_final_circuit(self, component_pins, holes, power_sources, warped):
        """최종 회로 생성 - 넷 병합 완전 처리"""
        print("🔄 회로도 생성 중 (넷 병합 디버깅)...")
        
        try:
            # 1️⃣ hole_to_net 맵 생성
            nets, row_nets = self.hole_det.get_board_nets(holes, base_img=warped, show=False)
            hole_to_net = {}
            for row_idx, clusters in row_nets:
                for entry in clusters:
                    net_id = entry['net_id']
                    for x, y in entry['pts']:
                        hole_to_net[(int(round(x)), int(round(y)))] = net_id
            
            print(f"📊 초기 홀-넷 매핑: {len(hole_to_net)}개")
            
            # 2️⃣ Union-Find 초기화 및 함수 정의
            all_nets = set(hole_to_net.values())
            parent = {net: net for net in all_nets}
            
            def find(u):
                if parent[u] != u:
                    parent[u] = find(parent[u])
                return parent[u]
            
            def union(u, v):
                pu, pv = find(u), find(v)
                if pu != pv:
                    if pu < pv:
                        parent[pv] = pu
                        print(f"    Union: Net{u}({pu}) ← Net{v}({pv}) → 대표: Net{pu}")
                    else:
                        parent[pu] = pv  
                        print(f"    Union: Net{u}({pu}) → Net{v}({pv}) ← 대표: Net{pv}")
                    return True
                return False
            
            def nearest_net(pt):
                if not hole_to_net:
                    print("⚠️ 경고: hole_to_net이 비어있습니다!")
                    return 0
                closest = min(hole_to_net.keys(), key=lambda h: (h[0]-pt[0])**2 + (h[1]-pt[1])**2)
                original_net = hole_to_net[closest]
                merged_net = find(original_net)
                print(f"    핀 {pt} → 홀 {closest} → 원래넷 {original_net} → 병합넷 {merged_net}")
                return merged_net
            
            # 3️⃣ 와이어 연결 처리 및 병합
            wires = []
            print("\n=== Line_area 와이어 처리 ===")
            
            for i, comp in enumerate(component_pins):
                if comp['class'] == 'Line_area' and len(comp['pins']) == 2:
                    pin1, pin2 = comp['pins']
                    
                    # 원래 넷 찾기 (병합 전)
                    closest1 = min(hole_to_net.keys(), key=lambda h: (h[0]-pin1[0])**2 + (h[1]-pin1[1])**2)
                    closest2 = min(hole_to_net.keys(), key=lambda h: (h[0]-pin2[0])**2 + (h[1]-pin2[1])**2)
                    net1_orig = hole_to_net[closest1]
                    net2_orig = hole_to_net[closest2]
                    
                    print(f"Wire {i+1}: 핀 {pin1} → Net{net1_orig}, 핀 {pin2} → Net{net2_orig}")
                    
                    if net1_orig != net2_orig:
                        wires.append((net1_orig, net2_orig))
                        union_result = union(net1_orig, net2_orig)
                        if union_result:
                            print(f"  ✅ 병합 성공: Net{net1_orig} ↔ Net{net2_orig}")
                        else:
                            print(f"  ⚠️ 이미 같은 그룹: Net{net1_orig}, Net{net2_orig}")
                    else:
                        print(f"  ⚠️ 같은 넷에 연결: Net{net1_orig}")
            
            # 4️⃣ 최종 병합 결과 확인
            print("\n=== 최종 병합 결과 ===")
            final_groups = {}
            for net in sorted(all_nets):
                root = find(net)
                final_groups.setdefault(root, []).append(net)
            
            for root, members in sorted(final_groups.items()):
                if len(members) > 1:
                    print(f"그룹 Net{root}: {sorted(members)} (병합됨)")
                else:
                    print(f"그룹 Net{root}: {members} (단독)")
            
            # 5️⃣ 컴포넌트 넷 매핑 디버깅
            print("\n=== 컴포넌트 넷 매핑 ===")
            for comp in component_pins:
                if comp['class'] != 'Line_area' and len(comp.get('pins', [])) == 2:
                    pin1, pin2 = comp['pins']
                    net1 = nearest_net(pin1)
                    net2 = nearest_net(pin2)
                    print(f"{comp['class']}: 핀 {pin1}, {pin2} → Net{net1}, Net{net2}")
            
            # 6️⃣ 다중 전원 매핑 (병합된 넷 사용)
            power_pairs = []
            img_w = warped.shape[1]
            comp_count = len([c for c in component_pins if c['class'] != 'Line_area'])
            grid_width = comp_count * 2 + 2
            
            print("\n=== 전원 넷 매핑 ===")
            for i, (voltage, plus_pt, minus_pt) in enumerate(power_sources, 1):
                # 🔧 핵심: 병합된 넷 사용
                net_plus = nearest_net(plus_pt)
                net_minus = nearest_net(minus_pt)
                
                x_plus_grid = plus_pt[0] / img_w * grid_width
                x_minus_grid = minus_pt[0] / img_w * grid_width
                power_pairs.append((net_plus, x_plus_grid, net_minus, x_minus_grid))
                
                print(f"전원 {i}: +{plus_pt}→Net{net_plus}, -{minus_pt}→Net{net_minus}")
            
            print(f"\n📊 최종 회로 정보:")
            print(f"  - 컴포넌트: {len([c for c in component_pins if c['class'] != 'Line_area'])}개")
            print(f"  - 전원: {len(power_pairs)}개")
            print(f"  - 와이어: {len(wires)}개")
            print(f"  - 병합 그룹: {len(final_groups)}개")
            
            # 7️⃣ 병합된 hole_to_net 생성 (중요!)
            merged_hole_to_net = {}
            for hole, original_net in hole_to_net.items():
                merged_net = find(original_net)
                merged_hole_to_net[hole] = merged_net
            
            print(f"\n🔧 병합된 hole_to_net 생성: {len(merged_hole_to_net)}개")
            
            # 8️⃣ generate_circuit 실행 (병합 완료된 데이터 전달)
            representative_voltage = power_sources[0][0] if power_sources else 5.0
            
            # 🔧 중요: 이미 병합된 데이터 전달, wires는 빈 리스트로!
            components, final_hole_to_net = generate_circuit(
                all_comps=component_pins,
                holes=holes,
                wires=[],  # 🔧 이미 병합했으므로 빈 리스트
                voltage=representative_voltage,
                output_spice='circuit.spice',
                output_img='circuit.jpg',
                hole_to_net=merged_hole_to_net,  # 🔧 병합된 결과 전달
                power_pairs=power_pairs
            )
            
            # 9️⃣ 오류 검사
            print("🔍 회로 오류 검사 중...")
            error_result = self._check_circuit_errors_lenient(
                components, power_pairs, power_sources
            )
            
            if not error_result:
                print("❌ 사용자가 회로도 생성을 취소했습니다.")
                return False
            
            print("✅ 넷 병합이 완전히 적용된 회로도 생성 완료!")
            print("📁 생성된 파일:")
            print("  - circuit.jpg (병합 적용 회로도)")
            print("  - circuit.spice (병합 적용 SPICE 넷리스트)")
            print("  - circuit_connected.jpg (연결선 포함)")
            print("  - circuit_traditional.jpg (전통적 스타일)")
            
            return True
            
        except Exception as e:
            print(f"❌ 회로 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _check_circuit_errors_lenient(self, components, power_pairs, power_sources):
        """관대한 오류 검사 - 다중 전원 허용"""
        try:
            components_for_check = components.copy()
            
            # 기존 전압원 확인
            existing_voltage_sources = [comp for comp in components_for_check if comp['class'] == 'VoltageSource']
            print(f"🔍 기존 전압원: {len(existing_voltage_sources)}개")
            
            # nets_mapping 생성
            nets_mapping = {}
            for comp in components_for_check:
                n1, n2 = comp['nodes']
                nets_mapping.setdefault(n1, []).append(comp['name'])
                nets_mapping.setdefault(n2, []).append(comp['name'])
            
            # 전압원이 부족한 경우에만 추가
            expected_voltage_sources = len(power_sources)
            if len(existing_voltage_sources) < expected_voltage_sources:
                print(f"⚠️ 전압원이 부족합니다. {expected_voltage_sources - len(existing_voltage_sources)}개 추가합니다.")
                
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
            
            ground_net = power_pairs[0][2] if power_pairs else 0
            
            # 🔧 관대한 ErrorChecker (다중 전원 경고만)
            checker = ErrorChecker(components_for_check, nets_mapping, ground_nodes={ground_net})
            errors = checker.run_all_checks()
            
            # 다중 전원 오류는 경고로만 처리
            filtered_errors = []
            for error in errors:
                if "Multiple voltage sources" in error:
                    print(f"⚠️ 경고 (무시됨): {error}")
                    print("   → 다중 전원이 의도적으로 설정되었습니다.")
                else:
                    filtered_errors.append(error)
            
            if filtered_errors:
                print(f"⚠️ {len(filtered_errors)}개의 심각한 오류가 발견되었습니다:")
                for i, error in enumerate(filtered_errors, 1):
                    print(f"  {i}. {error}")
                
                # 심각한 오류만 사용자에게 확인
                root = tk.Tk()
                root.withdraw()
                
                error_msg = f"다음 {len(filtered_errors)}개의 오류가 발견되었습니다:\n\n"
                for i, error in enumerate(filtered_errors[:5], 1):
                    error_msg += f"{i}. {error}\n"
                
                if len(filtered_errors) > 5:
                    error_msg += f"\n... 및 {len(filtered_errors) - 5}개 추가 오류\n"
                
                error_msg += "\n그래도 회로도를 생성하시겠습니까?"
                
                result = messagebox.askyesno("회로 오류 발견", error_msg)
                root.destroy()
                
                return result
            else:
                print("✅ 심각한 회로 오류가 발견되지 않았습니다!")
                if len(power_sources) > 1:
                    print(f"📋 {len(power_sources)}개의 전원이 하나의 회로에 표시됩니다.")
                return True
                
        except Exception as e:
            print(f"⚠️ 오류 검사 중 문제가 발생했습니다: {e}")
            print("회로도 생성을 계속합니다...")
            return True

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