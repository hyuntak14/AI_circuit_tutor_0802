# pin_manager.py
import cv2
import numpy as np

class PinManager:
    def __init__(self, class_colors, detectors):
        self.class_colors = class_colors
        self.resistor_det = detectors['resistor']
        self.led_det = detectors['led']
        self.diode_det = detectors['diode']
        self.ic_det = detectors['ic']
        self.wire_det = detectors['wire']
        self.hole_det = detectors['hole']
    
    def auto_pin_detection(self, warped, components):
        """자동 핀 검출 (실패 시 기본값 사용)"""
        print("📍 컴포넌트 핀 자동 검출 중...")
        
        # 구멍 검출
        holes = self.hole_det.detect_holes(warped)
        print(f"✅ {len(holes)}개 구멍 검출됨")
        
        component_pins = []
        for i, (cls, _, box) in enumerate(components):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            expected = 8 if cls == 'IC' else 2
            pins = []
            
            try:
                pins = self._detect_pins_by_class(cls, warped, box, holes)
            except Exception as e:
                print(f"⚠️ {cls} 핀 검출 실패: {e}")
            
            # 실패시 기본 위치 사용 (스마트 버전)
            if len(pins) != expected:
                print(f"⚠️ {cls} 핀 자동검출 실패, 스마트 기본값 사용")
                pins = self._get_smart_default_pins(cls, x1, y1, x2, y2, w, h, holes)
            
            component_pins.append({
                'class': cls,
                'box': box,
                'pins': pins,
                'value': 100.0 if cls == 'Resistor' else 0.0,
                'num_idx': i+1
            })
        
        print(f"✅ 모든 컴포넌트 핀 처리 완료")
        return component_pins, holes
    
    def manual_pin_verification_and_correction(self, warped, component_pins, holes):
        """핀 위치 확인 및 수정 단계"""
        print("📍 핀 위치 확인 및 수정 단계")
        print("- 좌클릭: 컴포넌트 선택하여 핀 수정")
        print("- 'v': 전체 핀 위치 확인 모드")
        print("- Enter: 다음 단계로")
        
        # hole_to_net 맵 생성 및 네트워크 설정
        hole_to_net, net_colors, find = self._setup_network_mapping(warped, holes)
        
        # 메인 루프
        window_name = 'Pin Location Validate & Edit'
        verification_mode = False
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and not verification_mode:
                # 클릭한 위치의 컴포넌트 찾기
                for i, comp in enumerate(component_pins):
                    x1, y1, x2, y2 = comp['box']
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        self._manual_pin_selection(warped, comp, i)
                        break
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            img = self._redraw_pins(warped, component_pins, hole_to_net, net_colors, find, verification_mode)
            
            if verification_mode:
                cv2.putText(img, "VERIFICATION MODE - Detailed pin analysis", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img, "Press 'v' to exit verification mode", 
                           (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(img, "EDIT MODE - Click component to edit pins", 
                           (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, "'v': detailed view | Enter: next step", 
                           (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(window_name, img)
            key = cv2.waitKey(30) & 0xFF
            
            if key == 13:  # Enter
                break
            elif key == ord('v'):  # 확인 모드 토글
                verification_mode = not verification_mode
                print(f"{'📋 상세 확인 모드' if verification_mode else '✏️ 편집 모드'}")
            elif key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        print("✅ 핀 위치 확인 및 수정 완료")
        
        return component_pins
    
    def _detect_pins_by_class(self, cls, warped, box, holes):
        """클래스별 핀 검출"""
        if cls == 'Resistor':
            result = self.resistor_det.extract(warped, box)
            if result and result[0] is not None and result[1] is not None:
                return list(result)
        elif cls == 'LED':
            result = self.led_det.extract(warped, box, holes)
            if result and 'endpoints' in result:
                return result['endpoints']
        elif cls == 'Diode':
            result = self.diode_det.extract(warped, box)
            if result and result[0] is not None and result[1] is not None:
                return list(result)
        elif cls == 'IC':
            x1, y1, x2, y2 = box
            roi = warped[y1:y2, x1:x2]
            ics = self.ic_det.detect(roi)
            if ics:
                return [(x1 + px, y1 + py) for px, py in ics[0]['pin_points']]
        elif cls == 'Line_area':
            x1, y1, x2, y2 = box
            roi = warped[y1:y2, x1:x2]
            segs = self.wire_det.detect_wires(roi)
            endpoints, _ = self.wire_det.select_best_endpoints(segs)
            if endpoints:
                return [(x1 + pt[0], y1 + pt[1]) for pt in endpoints]
        return []
    
    def _get_default_pins(self, cls, x1, y1, x2, y2, w, h):
        """기본 핀 위치 생성 (개선된 버전)"""
        if cls == 'IC':
            # IC 핀은 DIP 패키지 기준으로 좌상단부터 시계방향으로 8개
            # 실제 IC 핀 간격을 고려한 배치
            pin_margin_x = w // 6  # 가로 여백
            pin_margin_y = h // 6  # 세로 여백
            
            return [
                # 상단 (좌→우)
                (x1 + pin_margin_x, y1 + pin_margin_y),           # 핀 1
                (x1 + w//2, y1 + pin_margin_y),                  # 핀 2
                (x2 - pin_margin_x, y1 + pin_margin_y),          # 핀 3
                # 우측 (상→하)
                (x2 - pin_margin_x, y1 + h//2),                  # 핀 4
                # 하단 (우→좌)
                (x2 - pin_margin_x, y2 - pin_margin_y),          # 핀 5
                (x1 + w//2, y2 - pin_margin_y),                  # 핀 6
                (x1 + pin_margin_x, y2 - pin_margin_y),          # 핀 7
                # 좌측 (하→상)
                (x1 + pin_margin_x, y1 + h//2)                   # 핀 8
            ]
        else:
            # 2핀 컴포넌트: 컴포넌트 크기에 따른 적응적 여백
            center_x = x1 + w // 2
            center_y = y1 + h // 2
            
            if cls == 'Resistor' or cls == 'Diode':
                # 저항, 다이오드: 더 짧은 변의 중점을 핀 위치로 설정 (5% 안쪽)
                if w > h:  # 가로가 더 긴 경우 (일반적인 경우)
                    # 짧은 변(세로)의 중점들 = 좌우 변의 중점, 가로로 5% 안쪽
                    margin_x = int(w * 0.05)  # 가로의 5%
                    return [(x1 + margin_x, center_y), (x2 - margin_x, center_y)]
                else:  # 세로가 더 긴 경우
                    # 짧은 변(가로)의 중점들 = 상하 변의 중점, 세로로 5% 안쪽
                    margin_y = int(h * 0.05)  # 세로의 5%
                    return [(center_x, y1 + margin_y), (center_x, y2 - margin_y)]
            
            elif cls == 'LED':
                # LED: 긴 다리(양극)와 짧은 다리(음극) 고려
                margin = max(min(w // 4, 15), 5)
                return [(x1 + margin, center_y), (x2 - margin, center_y)]
            
            elif cls == 'Line_area':
                # 와이어: 양 끝점
                if w > h:  # 가로로 긴 경우
                    return [(x1 + 5, center_y), (x2 - 5, center_y)]
                else:  # 세로로 긴 경우
                    return [(center_x, y1 + 5), (center_x, y2 - 5)]
            
            elif cls == 'Capacitor':
                # 커패시터: 극성이 있는 경우 고려
                margin = max(min(w // 4, 15), 5)
                return [(x1 + margin, center_y), (x2 - margin, center_y)]
            
            else:
                # 기타 컴포넌트: 기본값
                margin = max(min(w // 4, 15), 5)
                return [(x1 + margin, center_y), (x2 - margin, center_y)]
    
    def _snap_to_nearest_hole(self, pin_pos, holes, max_distance=20):
        """핀 위치를 가장 가까운 구멍에 스냅"""
        if not holes:
            return pin_pos
            
        px, py = pin_pos
        closest_hole = min(holes, key=lambda h: (h[0]-px)**2 + (h[1]-py)**2)
        distance = ((closest_hole[0]-px)**2 + (closest_hole[1]-py)**2) ** 0.5
        
        # 최대 거리 내에 있으면 구멍 위치로 스냅
        if distance <= max_distance:
            return closest_hole
        else:
            return pin_pos
    
    def _get_smart_default_pins(self, cls, x1, y1, x2, y2, w, h, holes):
        """구멍 위치를 고려한 스마트 기본 핀 생성"""
        # 기본 핀 위치 생성
        default_pins = self._get_default_pins(cls, x1, y1, x2, y2, w, h)
        
        # 각 핀을 가장 가까운 구멍으로 스냅
        snapped_pins = []
        for pin in default_pins:
            snapped_pin = self._snap_to_nearest_hole(pin, holes)
            snapped_pins.append(snapped_pin)
        
        return snapped_pins
    
    def _setup_network_mapping(self, warped, holes):
        """네트워크 매핑 설정"""
        nets, row_nets = self.hole_det.get_board_nets(holes, base_img=warped, show=False)
        hole_to_net = {}
        for row_idx, clusters in row_nets:
            for entry in clusters:
                net_id = entry['net_id']
                for x, y in entry['pts']:
                    hole_to_net[(int(round(x)), int(round(y)))] = net_id
        
        # Union-Find 초기화
        parent = {net_id: net_id for net_id in set(hole_to_net.values())}
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
        # Net 색상 매핑
        rng = np.random.default_rng(1234)
        final_nets = set(find(n) for n in hole_to_net.values())
        net_colors = {
            net_id: tuple(int(c) for c in rng.integers(0, 256, 3))
            for net_id in final_nets
        }
        
        return hole_to_net, net_colors, find
    
    def _redraw_pins(self, warped, component_pins, hole_to_net, net_colors, find, verification_mode=False):
        """핀 위치와 네트워크 매핑을 시각화"""
        img = warped.copy()
        
        if verification_mode:
            # 확인 모드: 상세한 정보 표시
            self._draw_detailed_view(img, component_pins, hole_to_net, net_colors, find)
        else:
            # 일반 모드: 간단한 표시
            self._draw_simple_view(img, component_pins, hole_to_net, net_colors, find)
        
        return img
    
    def _draw_detailed_view(self, img, component_pins, hole_to_net, net_colors, find):
        """상세 확인 모드 그리기"""
        # 모든 구멍을 네트 색상으로 표시
        for (hx, hy), net_id in hole_to_net.items():
            final_net = find(net_id)
            hole_color = net_colors.get(final_net, (128, 128, 128))
            cv2.circle(img, (int(hx), int(hy)), 3, hole_color, -1)
        
        # 컴포넌트와 핀을 자세히 표시
        for i, comp in enumerate(component_pins):
            x1, y1, x2, y2 = comp['box']
            color = self.class_colors.get(comp['class'], (0, 255, 255))
            
            # 컴포넌트 박스 (두껍게)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            # 컴포넌트 상태 정보
            expected = 8 if comp['class'] == 'IC' else 2
            actual = len(comp['pins'])
            status = "✓" if actual == expected else "⚠"
            info_text = f"{i+1}:{comp['class']} {status}({actual}/{expected})"
            cv2.putText(img, info_text, (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 핀을 크게 표시하고 연결선 추가
            for j, (px, py) in enumerate(comp['pins']):
                if hole_to_net:
                    closest = min(hole_to_net.keys(), 
                                key=lambda h: (h[0]-px)**2 + (h[1]-py)**2)
                    raw_net = hole_to_net[closest]
                    net_id = find(raw_net)
                    pin_color = net_colors.get(net_id, (255, 255, 255))
                    
                    # 핀과 구멍 사이 연결선 표시
                    cv2.line(img, (int(px), int(py)), closest, (255, 255, 255), 1)
                else:
                    pin_color = (0, 255, 0)
                
                # 핀을 크게 표시
                cv2.circle(img, (int(px), int(py)), 8, pin_color, -1)
                cv2.circle(img, (int(px), int(py)), 8, (0, 0, 0), 2)
                
                # 핀 번호와 네트 ID 표시
                cv2.putText(img, f"P{j+1}", (int(px)+10, int(py)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if hole_to_net and net_id:
                    cv2.putText(img, f"N{net_id}", (int(px)+10, int(py)+15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, pin_color, 1)
        
        # 범례 추가
        self._draw_legend(img)
    
    def _draw_simple_view(self, img, component_pins, hole_to_net, net_colors, find):
        """간단한 편집 모드 그리기"""
        for i, comp in enumerate(component_pins):
            x1, y1, x2, y2 = comp['box']
            color = self.class_colors.get(comp['class'], (0, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{i+1}:{comp['class']}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 핀을 간단히 표시
            for j, (px, py) in enumerate(comp['pins']):
                if hole_to_net:
                    closest = min(hole_to_net.keys(), 
                                key=lambda h: (h[0]-px)**2 + (h[1]-py)**2)
                    raw_net = hole_to_net[closest]
                    net_id = find(raw_net)
                    pin_color = net_colors.get(net_id, (255, 255, 255))
                else:
                    pin_color = (0, 255, 0)
                
                cv2.circle(img, (int(px), int(py)), 6, pin_color, -1)
                cv2.circle(img, (int(px), int(py)), 6, (0, 0, 0), 2)
                
                # 핀 번호만 간단히 표시
                cv2.putText(img, str(j+1), (int(px)+8, int(py)-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_legend(self, img):
        """범례 그리기"""
        legend_y = 50
        cv2.putText(img, "Legend:", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "Small dots: Holes", (10, legend_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "Large circles: Component pins", (10, legend_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, "Lines: Pin-to-hole connections", (10, legend_y + 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _manual_pin_selection(self, warped, comp, comp_idx):
        """특정 컴포넌트의 핀을 수동으로 재설정"""
        x1, y1, x2, y2 = comp['box']
        expected = 8 if comp['class'] == 'IC' else 2
        
        print(f"📍 {comp['class']} #{comp_idx+1}의 핀 {expected}개를 선택하세요")
        
        # ROI 추출 및 확대
        roi = warped[y1:y2, x1:x2].copy()
        h, w = roi.shape[:2]
        scale = 3
        roi_resized = cv2.resize(roi, (w*scale, h*scale), interpolation=cv2.INTER_LINEAR)
        
        pins = []
        window_name = f'{comp["class"]} Pin Selection'
        
        def mouse_cb(event, x, y, flags, param):
            nonlocal pins, roi_resized
            
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(pins) < expected:
                    # 클릭 좌표를 원본 크기로 변환
                    x_orig = int(x / scale)
                    y_orig = int(y / scale)
                    pins.append((x1 + x_orig, y1 + y_orig))
                    
                    # 시각화
                    cv2.circle(roi_resized, (x, y), 8, (0, 0, 255), -1)
                    cv2.putText(roi_resized, str(len(pins)), (x+10, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow(window_name, roi_resized)
                    
            elif event == cv2.EVENT_RBUTTONDOWN:
                # 마지막 핀 제거
                if pins:
                    pins.pop()
                    # 이미지 다시 그리기
                    roi_resized = cv2.resize(warped[y1:y2, x1:x2].copy(), 
                                           (w*scale, h*scale), interpolation=cv2.INTER_LINEAR)
                    for idx, (px, py) in enumerate(pins):
                        disp_x = (px - x1) * scale
                        disp_y = (py - y1) * scale
                        cv2.circle(roi_resized, (int(disp_x), int(disp_y)), 8, (0, 0, 255), -1)
                        cv2.putText(roi_resized, str(idx+1), (int(disp_x)+10, int(disp_y)-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow(window_name, roi_resized)
        
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_cb)
        cv2.imshow(window_name, roi_resized)
        
        print(f"좌클릭: 핀 추가 ({len(pins)}/{expected}), 우클릭: 마지막 핀 제거, ESC: 완료")
        
        while len(pins) < expected:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        cv2.destroyWindow(window_name)
        
        if len(pins) == expected:
            comp['pins'] = pins
            print(f"✅ {comp['class']} #{comp_idx+1} 핀 {len(pins)}개 업데이트됨")
        else:
            print(f"⚠️ 핀 개수 부족 ({len(pins)}/{expected}), 기존 핀 유지")

    def get_component_pin_info(self, component_pins):
        """컴포넌트 핀 정보 요약"""
        total_components = len(component_pins)
        completed_components = 0
        
        for comp in component_pins:
            expected = 8 if comp['class'] == 'IC' else 2
            actual = len(comp['pins'])
            if actual == expected:
                completed_components += 1
        
        return {
            'total': total_components,
            'completed': completed_components,
            'remaining': total_components - completed_components
        }