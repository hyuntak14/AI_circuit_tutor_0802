# pin_manager.py - 수정된 버전
import cv2
import numpy as np

class PinManager:
    def __init__(self, class_colors, detectors):
        self.class_colors = class_colors
        self.resistor_det = detectors['resistor']
        self.led_det = detectors['led']
        self.diode_det = detectors['diode']
        self.capacitor_det = detectors['capacitor']
        self.ic_det = detectors['ic']
        self.wire_det = detectors['wire']
        self.hole_det = detectors['hole']
    
    def auto_pin_detection(self, warped, components, original_img=None, original_bb=None):
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
                # 원본 이미지가 있으면 원본에서 핀 검출
                pins, updated_box = self._detect_pins_by_class(cls, warped, box, holes, original_img, original_bb)
                if updated_box:
                    box = updated_box
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
    

    def _transform_to_warped_coords_box(self, orig_box, original_bb, warped_img):
        """원본 좌표 box를 warped box로 변환"""
        orig_x1, orig_y1, orig_x2, orig_y2 = orig_box
        orig_bb_x1, orig_bb_y1, orig_bb_x2, orig_bb_y2 = original_bb
        warped_h, warped_w = warped_img.shape[:2]
        orig_w = orig_bb_x2 - orig_bb_x1
        orig_h = orig_bb_y2 - orig_bb_y1

        scale_x = warped_w / orig_w
        scale_y = warped_h / orig_h

        warped_x1 = int((orig_x1 - orig_bb_x1) * scale_x)
        warped_y1 = int((orig_y1 - orig_bb_y1) * scale_y)
        warped_x2 = int((orig_x2 - orig_bb_x1) * scale_x)
        warped_y2 = int((orig_y2 - orig_bb_y1) * scale_y)

        return (warped_x1, warped_y1, warped_x2, warped_y2)

    def _detect_pins_by_class(self, cls, warped, box, holes, original_img=None, original_bb=None):
        """
        clsごとのピン検出。必ず (pins, updated_warped_box or None) のタプルを返す。
        """
        # 기본값
        updated_box = None

        # ① 원본 이미지 기반 검출 시도
        if original_img is not None and original_bb is not None:
            # 원본 좌표계 박스 계산 + 10% 확장
            orig_box = self._transform_to_original_coords(box, warped, original_bb)
            x1o, y1o, x2o, y2o = orig_box
            w0, h0 = x2o-x1o, y2o-y1o
            dx, dy = int(w0*0.05), int(h0*0.05)
            ih, iw = original_img.shape[:2]
            x1e = max(0, x1o-dx); y1e = max(0, y1o-dy)
            x2e = min(iw, x2o+dx); y2e = min(ih, y2o+dy)
            orig_box = (x1e, y1e, x2e, y2e)

            # warped上の拡張box
            updated_box = self._transform_to_warped_coords_box(orig_box, original_bb, warped)

            # 클래스별 원본検出
            if cls == 'Resistor':
                res = self.resistor_det.extract(original_img, orig_box)
                if res and res[0] is not None and res[1] is not None:
                    pins = self._transform_to_warped_coords(list(res), original_bb, warped)
                    return pins, updated_box

            elif cls == 'LED':
                # 1) 원본 이미지 기반 검출 시도
                if original_img is not None and original_bb is not None:
                    # 올바른 인자 순서: box, warped, original_bb
                    orig_box   = self._transform_to_original_coords(box, warped, original_bb)
                    orig_holes = self._transform_holes_to_original(holes, original_bb, warped)


                    # 원본 이미지에서 LED endpoint 추출
                    endpoints = self.led_det.extract(original_img, orig_box, orig_holes)
                    if endpoints:  # 리스트 반환만으로도 검사됨
                        # original 좌표계를 warped 좌표계로 다시 변환
                        endpoints_warped = self._transform_to_warped_coords(
                            endpoints, original_bb, warped
                        )
                        return endpoints_warped, None

            elif cls == 'Capacitor':
                # 1) 원본 이미지 기반 검출 시도
                if original_img is not None and original_bb is not None:
                    # 올바른 인자 순서: box, warped, original_bb
                    orig_box   = self._transform_to_original_coords(box, warped, original_bb)
                    orig_holes = self._transform_holes_to_original(holes, original_bb, warped)


                    # 원본 이미지에서 Capacitor endpoint 추출
                    endpoints = self.capacitor_det.extract(original_img, orig_box, orig_holes)
                    if endpoints:  # 리스트 반환만으로도 검사됨
                        # original 좌표계를 warped 좌표계로 다시 변환
                        endpoints_warped = self._transform_to_warped_coords(
                            endpoints, original_bb, warped
                        )
                        return endpoints_warped, None
                # 2) warped 이미지 기반 폴백 검출
                endpoints = self.led_det.extract(warped, box, holes)
                if endpoints:
                    return endpoints, None

            elif cls == 'Diode':
                res = self.diode_det.extract(original_img, orig_box)
                if res and res[0] is not None and res[1] is not None:
                    pins = self._transform_to_warped_coords(list(res), original_bb, warped)
                    return pins, updated_box

            elif cls == 'IC':
                roi = original_img[y1e:y2e, x1e:x2e]
                dets = self.ic_det.detect(roi)
                if dets:
                    orig_pins = [(x1e+px, y1e+py) for px,py in dets[0]['pin_points']]
                    pins = self._transform_to_warped_coords(orig_pins, original_bb, warped)
                    return pins, updated_box

            elif cls == 'Line_area':
                roi = original_img[y1e:y2e, x1e:x2e]
                segs = self.wire_det.detect_wires(roi)
                ends, _ = self.wire_det.select_best_endpoints(segs)
                if ends:
                    orig_pins = [(x1e+pt[0], y1e+pt[1]) for pt in ends]
                    pins = self._transform_to_warped_coords(orig_pins, original_bb, warped)
                    return pins, updated_box

        # ② warped上에서 fallback 検出
        if cls == 'Resistor':
            res = self.resistor_det.extract(warped, box)
            if res and res[0] is not None and res[1] is not None:
                return list(res), None
        elif cls == 'LED':
            endpoints = self.led_det.extract(warped, box, holes)
            if endpoints:
                # 항상 [(x1,y1), (x2,y2)] 형태의 리스트를 돌려받으므로
                return endpoints, None

        elif cls == 'Capacitor':
            endpoints = self.led_det.extract(warped, box, holes)
            if endpoints:
                # 항상 [(x1,y1), (x2,y2)] 형태의 리스트를 돌려받으므로
                return endpoints, None

        elif cls == 'Diode':
            res = self.diode_det.extract(warped, box)
            if res and res[0] is not None and res[1] is not None:
                return list(res), None

        elif cls == 'IC':
            x1, y1, x2, y2 = box
            roi = warped[y1:y2, x1:x2]
            dets = self.ic_det.detect(roi)
            if dets:
                pins = [(x1+px, y1+py) for px,py in dets[0]['pin_points']]
                return pins, None

        elif cls == 'Line_area':
            x1, y1, x2, y2 = box
            roi = warped[y1:y2, x1:x2]
            segs = self.wire_det.detect_wires(roi)
            ends, _ = self.wire_det.select_best_endpoints(segs)
            if ends:
                pins = [(x1+pt[0], y1+pt[1]) for pt in ends]
                return pins, None

        # ③ 모두 실패 시
        return [], None

    
    def _transform_to_original_coords(self, warped_box, warped_img, original_bb):
        """warped 좌표를 원본 이미지 좌표로 변환"""
        x1, y1, x2, y2 = warped_box
        orig_bb_x1, orig_bb_y1, orig_bb_x2, orig_bb_y2 = original_bb
        
        # warped는 640x640, 원본 bbox는 original_bb 크기
        warped_h, warped_w = warped_img.shape[:2]
        orig_w = orig_bb_x2 - orig_bb_x1
        orig_h = orig_bb_y2 - orig_bb_y1
        
        # 스케일 계산
        scale_x = orig_w / warped_w
        scale_y = orig_h / warped_h
        
        # 좌표 변환
        orig_x1 = int(orig_bb_x1 + x1 * scale_x)
        orig_y1 = int(orig_bb_y1 + y1 * scale_y)
        orig_x2 = int(orig_bb_x1 + x2 * scale_x)
        orig_y2 = int(orig_bb_y1 + y2 * scale_y)
        
        return (orig_x1, orig_y1, orig_x2, orig_y2)
    
    def _transform_to_warped_coords(self, original_points, original_bb, warped_img):
        """원본 좌표를 warped 좌표로 변환"""
        orig_bb_x1, orig_bb_y1, orig_bb_x2, orig_bb_y2 = original_bb
        warped_h, warped_w = warped_img.shape[:2]
        orig_w = orig_bb_x2 - orig_bb_x1
        orig_h = orig_bb_y2 - orig_bb_y1
        
        # 스케일 계산
        scale_x = warped_w / orig_w
        scale_y = warped_h / orig_h
        
        warped_points = []
        for px, py in original_points:
            # 원본 bbox 기준으로 정규화 후 warped 크기로 스케일링
            warped_x = int((px - orig_bb_x1) * scale_x)
            warped_y = int((py - orig_bb_y1) * scale_y)
            warped_points.append((warped_x, warped_y))
        
        return warped_points
    
    def _transform_holes_to_original(self, holes, original_bb, warped_img):
        """구멍 좌표를 원본 좌표로 변환"""
        orig_bb_x1, orig_bb_y1, orig_bb_x2, orig_bb_y2 = original_bb
        warped_h, warped_w = warped_img.shape[:2]
        orig_w = orig_bb_x2 - orig_bb_x1
        orig_h = orig_bb_y2 - orig_bb_y1
        
        # 스케일 계산
        scale_x = orig_w / warped_w
        scale_y = orig_h / warped_h
        
        orig_holes = []
        for hx, hy in holes:
            orig_x = int(orig_bb_x1 + hx * scale_x)
            orig_y = int(orig_bb_y1 + hy * scale_y)
            orig_holes.append((orig_x, orig_y))
        
        return orig_holes
    
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
    
    # 나머지 메서드들은 기존과 동일
    def _get_default_pins(self, cls, x1, y1, x2, y2, w, h):
        """기본 핀 위치 생성 (개선된 버전)"""
        if cls == 'IC':
            # IC 핀은 DIP 패키지 기준으로 좌상단부터 시계방향으로 8개
            pin_margin_x = w // 6
            pin_margin_y = h // 6
            
            return [
                (x1 + pin_margin_x, y1 + pin_margin_y),
                (x1 + w//2, y1 + pin_margin_y),
                (x2 - pin_margin_x, y1 + pin_margin_y),
                (x2 - pin_margin_x, y1 + h//2),
                (x2 - pin_margin_x, y2 - pin_margin_y),
                (x1 + w//2, y2 - pin_margin_y),
                (x1 + pin_margin_x, y2 - pin_margin_y),
                (x1 + pin_margin_x, y1 + h//2)
            ]
        else:
            center_x = x1 + w // 2
            center_y = y1 + h // 2
            
            if cls == 'Resistor' or cls == 'Diode':
                if w > h:
                    margin_x = int(w * 0.05)
                    return [(x1 + margin_x, center_y), (x2 - margin_x, center_y)]
                else:
                    margin_y = int(h * 0.05)
                    return [(center_x, y1 + margin_y), (center_x, y2 - margin_y)]
            else:
                margin = max(min(w // 4, 15), 5)
                return [(x1 + margin, center_y), (x2 - margin, center_y)]
    
    def _snap_to_nearest_hole(self, pin_pos, holes, max_distance=20):
        """핀 위치를 가장 가까운 구멍에 스냅"""
        if not holes:
            return pin_pos
        px, py = pin_pos
        closest_hole = min(holes, key=lambda h: (h[0]-px)**2 + (h[1]-py)**2)
        distance = ((closest_hole[0]-px)**2 + (closest_hole[1]-py)**2) ** 0.5
        
        if distance <= max_distance:
            return closest_hole
        else:
            return pin_pos
    
    def _get_smart_default_pins(self, cls, x1, y1, x2, y2, w, h, holes):
        """구멍 위치를 고려한 스마트 기본 핀 생성"""
        default_pins = self._get_default_pins(cls, x1, y1, x2, y2, w, h)
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
        
        parent = {net_id: net_id for net_id in set(hole_to_net.values())}
        def find(u):
            if parent[u] != u:
                parent[u] = find(parent[u])
            return parent[u]
        
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
            self._draw_detailed_view(img, component_pins, hole_to_net, net_colors, find)
        else:
            self._draw_simple_view(img, component_pins, hole_to_net, net_colors, find)
        
        return img
    
    def _draw_detailed_view(self, img, component_pins, hole_to_net, net_colors, find):
        """상세 확인 모드 그리기"""
        for (hx, hy), net_id in hole_to_net.items():
            final_net = find(net_id)
            hole_color = net_colors.get(final_net, (128, 128, 128))
            cv2.circle(img, (int(hx), int(hy)), 3, hole_color, -1)
        
        for i, comp in enumerate(component_pins):
            x1, y1, x2, y2 = comp['box']
            color = self.class_colors.get(comp['class'], (0, 255, 255))
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            expected = 8 if comp['class'] == 'IC' else 2
            actual = len(comp['pins'])
            status = "✓" if actual == expected else "⚠"
            info_text = f"{i+1}:{comp['class']} {status}({actual}/{expected})"
            cv2.putText(img, info_text, (x1, y1-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            for j, (px, py) in enumerate(comp['pins']):
                if hole_to_net:
                    closest = min(hole_to_net.keys(), 
                                key=lambda h: (h[0]-px)**2 + (h[1]-py)**2)
                    raw_net = hole_to_net[closest]
                    net_id = find(raw_net)
                    pin_color = net_colors.get(net_id, (255, 255, 255))
                    cv2.line(img, (int(px), int(py)), closest, (255, 255, 255), 1)
                else:
                    pin_color = (0, 255, 0)
                
                cv2.circle(img, (int(px), int(py)), 8, pin_color, -1)
                cv2.circle(img, (int(px), int(py)), 8, (0, 0, 0), 2)
                
                cv2.putText(img, f"P{j+1}", (int(px)+10, int(py)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                if hole_to_net and net_id:
                    cv2.putText(img, f"N{net_id}", (int(px)+10, int(py)+15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, pin_color, 1)
        
        self._draw_legend(img)
    
    def _draw_simple_view(self, img, component_pins, hole_to_net, net_colors, find):
        """간단한 편집 모드 그리기"""
        for i, comp in enumerate(component_pins):
            x1, y1, x2, y2 = comp['box']
            color = self.class_colors.get(comp['class'], (0, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{i+1}:{comp['class']}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
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
                    x_orig = int(x / scale)
                    y_orig = int(y / scale)
                    pins.append((x1 + x_orig, y1 + y_orig))
                    
                    cv2.circle(roi_resized, (x, y), 8, (0, 0, 255), -1)
                    cv2.putText(roi_resized, str(len(pins)), (x+10, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imshow(window_name, roi_resized)
                    
            elif event == cv2.EVENT_RBUTTONDOWN:
                if pins:
                    pins.pop()
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