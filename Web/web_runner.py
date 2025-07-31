# web_runner_fixed.py - 핀 검출 로직이 개선된 웹 기반 회로 분석 백엔드
import cv2
import numpy as np
import os
import sys
import base64
import tempfile
import json
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from gemini_test_rag import create_rag_prompt, initialize_gemini  # ➊ 

# 더미 검출기 클래스들 (모델 파일이 없을 때 사용)
class DummyDetector:
    """모델 파일이 없을 때 사용하는 더미 검출기"""
    
    def __init__(self):
        print("🔧 더미 검출기 사용 중 - 실제 검출 대신 샘플 데이터를 반환합니다")
    
    def detect(self, image):
        """더미 컴포넌트 검출 - 샘플 데이터 반환"""
        h, w = image.shape[:2]
        
        # 가상의 컴포넌트들 생성
        components = [
            ('Resistor', 0.85, (int(w*0.2), int(h*0.3), int(w*0.4), int(h*0.4))),
            ('LED', 0.78, (int(w*0.5), int(h*0.2), int(w*0.7), int(h*0.35))),
            ('Resistor', 0.82, (int(w*0.3), int(h*0.6), int(w*0.5), int(h*0.7)))
        ]
        
        print(f"🔧 더미 검출 결과: {len(components)}개 컴포넌트")
        return components

class DummyHoleDetector:
    """구멍 검출기가 없을 때 사용하는 더미"""
    
    def __init__(self):
        print("🔧 더미 구멍 검출기 사용 중")
    
    def detect_holes(self, image):
        """더미 구멍 검출 - 격자 패턴 반환"""
        h, w = image.shape[:2]
        holes = []
        
        # 30x10 격자 패턴으로 가상 구멍 생성
        for row in range(10):
            for col in range(30):
                x = int(w * 0.05 + col * (w * 0.9 / 29))
                y = int(h * 0.1 + row * (h * 0.8 / 9))
                holes.append((x, y))
        
        print(f"🔧 더미 구멍 검출 결과: {len(holes)}개")
        return holes

class DummyComponentDetector:
    """더미 컴포넌트별 검출기"""
    def __init__(self):
        pass
    
    def extract(self, image, box, holes=None):
        """더미 핀 추출"""
        x1, y1, x2, y2 = box
        center_y = (y1 + y2) // 2
        # 기본 2핀 반환
        return [(x1 + 10, center_y), (x2 - 10, center_y)]

# 프로젝트 경로 추가
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

try:
    from simple_main import SimpleCircuitConverter
    from detector.fasterrcnn_detector import FasterRCNNDetector
    from detector.hole_detector import HoleDetector
    from ComponentEditor import ComponentEditor
    from pin_manager import PinManager
    from circuit_generator_manager import CircuitGeneratorManager
    from llm_feedback_manager import LLMFeedbackManager
except ImportError as e:
    print(f"⚠️ Import 오류: {e}")
    print("필요한 모듈들이 없거나 경로가 잘못되었습니다.")
    
    # 더미 클래스들로 대체
    class SimpleCircuitConverter: pass
    class FasterRCNNDetector: pass
    class HoleDetector: pass
    class ComponentEditor: pass
    class PinManager: pass 
    class CircuitGeneratorManager: pass
    class LLMFeedbackManager: pass

class WebRunnerComplete:
    """완전한 웹 기반 회로 분석기"""
    
    def __init__(self):
        self.logs = []
        self.progress = 0
        self.current_step = 0
        self.model = None
        self.generation_config = None

        try:
            self.model, self.generation_config = initialize_gemini()
            self._log("✅ Gemini 모델 초기화 완료")
        except Exception as e:
            self._log(f"⚠️ Gemini 초기화 실패: {e}")
        # 기본 설정
        self.display_size = (1200, 1200)
        
        # 모델 파일 경로 자동 탐지
        model_paths = [
            # 절대 경로 (원본)
            r'D:/Hyuntak/lab/AR_circuit_tutor/breadboard_project/model/fasterrcnn_v2.pt',
            # 상대 경로들 시도
            'model/fasterrcnn_v2.pt',
            '../model/fasterrcnn_v2.pt',
            '../../model/fasterrcnn_v2.pt',
            './breadboard_project/model/fasterrcnn_v2.pt'
        ]
        
        template_paths = [
            ('detector/template_holes_complete.csv', 'detector/breadboard18.jpg'),
            ('../detector/template_holes_complete.csv', '../detector/breadboard18.jpg'),
            ('../../detector/template_holes_complete.csv', '../../detector/breadboard18.jpg'),
            ('./detector/template_holes_complete.csv', './detector/breadboard18.jpg')
        ]
        
        # 검출기 초기화 시도
        self.detector = None
        self.hole_detector = None
        
        # FasterRCNN 모델 로드 시도
        for model_path in model_paths:
            if os.path.exists(model_path):
                try:
                    self.detector = FasterRCNNDetector(model_path)
                    print(f"✅ 모델 로드 성공: {model_path}")
                    break
                except Exception as e:
                    print(f"⚠️ 모델 로드 실패 ({model_path}): {e}")
                    continue
        
        if self.detector is None:
            print("⚠️ FasterRCNN 모델을 찾을 수 없어 더미 검출기를 사용합니다")
            self.detector = DummyDetector()
        
        # HoleDetector 초기화 시도
        for csv_path, img_path in template_paths:
            if os.path.exists(csv_path) and os.path.exists(img_path):
                try:
                    self.hole_detector = HoleDetector(
                        template_csv_path=csv_path,
                        template_image_path=img_path,
                        max_nn_dist=20.0
                    )
                    print(f"✅ 구멍 검출기 로드 성공: {csv_path}")
                    break
                except Exception as e:
                    print(f"⚠️ 구멍 검출기 로드 실패: {e}")
                    continue
        
        if self.hole_detector is None:
            print("⚠️ 구멍 검출기를 찾을 수 없어 더미 검출기를 사용합니다")
            self.hole_detector = DummyHoleDetector()
        
        # 컴포넌트 색상 매핑
        self.class_colors = {
            'Breadboard': (0, 128, 255),
            'Capacitor': (255, 0, 255),
            'Diode': (0, 255, 0),
            'IC': (204, 102, 255),
            'LED': (102, 0, 102),
            'Line_area': (255, 0, 0),
            'Resistor': (200, 170, 0)
        }
        
        # 매니저 클래스들 (지연 초기화)
        self.component_editor = None
        self.pin_manager = None
        self.circuit_generator = None
        self.llm_manager = None
        
        # 분석 상태
        self.warped_image = None
        self.original_bb = None
        self.analysis_data = {}
        self.original_image = None  # 원본 이미지 저장
    
    def _log(self, message, progress=None):
        """로그 기록 및 진행률 업데이트"""
        self.logs.append(message)
        if progress is not None:
            self.progress = progress
        print(f"[{self.progress}%] {message}")
    
    def _image_to_base64(self, image):
        """OpenCV 이미지를 base64 문자열로 변환"""
        try:
            # OpenCV BGR을 RGB로 변환
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            # PIL Image로 변환
            pil_image = Image.fromarray(image_rgb)
            
            # base64로 인코딩
            buffer = BytesIO()
            pil_image.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str
        except Exception as e:
            self._log(f"이미지 변환 오류: {e}")
            return None
    
    def _base64_to_image(self, base64_str):
        """base64 문자열을 OpenCV 이미지로 변환"""
        try:
            # base64 디코딩
            image_data = base64.b64decode(base64_str)
            
            # PIL Image로 로드
            pil_image = Image.open(BytesIO(image_data))
            
            # numpy 배열로 변환
            image_array = np.array(pil_image)
            
            # RGB를 BGR로 변환 (OpenCV 형식)
            if len(image_array.shape) == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_array
            
            return image_bgr
        except Exception as e:
            self._log(f"이미지 디코딩 오류: {e}")
            return None
    
    def _draw_components_on_image(self, image, components):
        """이미지에 컴포넌트 박스 그리기"""
        try:
            # OpenCV 이미지를 PIL로 변환
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            pil_image = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # 폰트 설정 (기본 폰트 사용)
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            for i, (cls, conf, box) in enumerate(components):
                x1, y1, x2, y2 = box
                
                # 컴포넌트별 색상
                color = self.class_colors.get(cls, (255, 0, 0))  # 기본 빨간색
                
                # 박스 그리기
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # 라벨 그리기
                label = f"{i+1}. {cls} ({conf:.1%})"
                
                # 라벨 배경
                bbox = draw.textbbox((x1, y1-25), label, font=font)
                draw.rectangle(bbox, fill=color)
                
                # 라벨 텍스트
                draw.text((x1, y1-25), label, fill=(255, 255, 255), font=font)
            
            # PIL을 다시 OpenCV로 변환
            result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return result_image
            
        except Exception as e:
            self._log(f"컴포넌트 시각화 오류: {e}")
            return image
    
    def _draw_pins_on_image(self, image, component_pins):
        """이미지에 핀 위치 그리기"""
        try:
            # OpenCV 이미지를 PIL로 변환
            if len(image.shape) == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            pil_image = Image.fromarray(image_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # 폰트 설정
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            for i, comp in enumerate(component_pins):
                cls = comp['class']
                pins = comp.get('pins', [])
                
                # 컴포넌트 박스도 그리기
                if 'box' in comp:
                    x1, y1, x2, y2 = comp['box']
                    color = self.class_colors.get(cls, (255, 0, 0))
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    
                    # 컴포넌트 라벨
                    label = f"{i+1}. {cls}"
                    bbox = draw.textbbox((x1, y1-20), label, font=font)
                    draw.rectangle(bbox, fill=color)
                    draw.text((x1, y1-20), label, fill=(255, 255, 255), font=font)
                
                # 핀 그리기
                for j, (px, py) in enumerate(pins):
                    # 핀 점 그리기 (파란색 원)
                    pin_radius = 8
                    draw.ellipse([px-pin_radius, py-pin_radius, px+pin_radius, py+pin_radius], 
                               fill=(0, 100, 255), outline=(0, 0, 255), width=2)
                    
                    # 핀 번호 표시
                    pin_label = str(j+1)
                    draw.text((px+10, py-10), pin_label, fill=(0, 0, 255), font=font)
            
            # PIL을 다시 OpenCV로 변환
            result_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return result_image
            
        except Exception as e:
            self._log(f"핀 시각화 오류: {e}")
            return image
    
    def detect_components(self, image_path):
        """
        1-3단계: 이미지 로드, 브레드보드 검출, 컴포넌트 검출
        
        Args:
            image_path: 업로드된 이미지 경로
            
        Returns:
            dict: {
                'success': bool,
                'components': list,  # [(class, confidence, box), ...]
                'warped_image_b64': str,
                'component_image_b64': str,  # 컴포넌트가 표시된 이미지
                'logs': list,
                'progress': int
            }
        """
        try:
            self._log("1단계: 이미지 로드 중...", 10)
            
            # 이미지 로드
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
            
            # 원본 이미지 저장
            self.original_image = img.copy()
            
            self._log("2단계: 브레드보드 자동 검출 중...", 20)
            
            # 브레드보드 검출 및 변환
            if self.detector is None:
                self._log("⚠️ 검출기가 없어 전체 이미지를 브레드보드로 사용", 25)
                result = self._use_full_image_as_breadboard(img)
            else:
                result = self._detect_and_transform_breadboard(img)
                
            if result is None:
                raise RuntimeError("브레드보드 검출 실패")
            
            self.warped_image, self.original_bb = result
            
            self._log("3단계: 컴포넌트 자동 검출 중...", 30)
            
            # 컴포넌트 검출
            if hasattr(self.detector, 'detect'):
                detections = self.detector.detect(self.warped_image)
                components = [(cls, conf, box) for cls, conf, box in detections 
                             if cls.lower() != 'breadboard']
            else:
                self._log("⚠️ 실제 검출기 대신 더미 데이터 사용", 32)
                components = []
            
            if not components:
                self._log("⚠️ 컴포넌트가 검출되지 않았습니다", 35)
            
            self._log(f"✅ {len(components)}개 컴포넌트 검출 완료", 40)
            
            # 컴포넌트가 표시된 이미지 생성
            component_image = self._draw_components_on_image(self.warped_image.copy(), components)
            
            # 결과 반환
            return {
                'success': True,
                'components': components,
                'warped_image_b64': self._image_to_base64(self.warped_image),
                'component_image_b64': self._image_to_base64(component_image),
                'logs': self.logs.copy(),
                'progress': self.progress
            }
            
        except Exception as e:
            self._log(f"❌ 컴포넌트 검출 오류: {e}")
            return {
                'success': False,
                'components': [],
                'warped_image_b64': None,
                'component_image_b64': None,
                'logs': self.logs.copy(),
                'progress': self.progress
            }
    
    def _use_full_image_as_breadboard(self, img):
        """검출기가 없을 때 전체 이미지를 브레드보드로 사용"""
        try:
            h, w = img.shape[:2]
            
            # 전체 이미지를 표준 크기로 리사이즈
            warped = cv2.resize(img, self.display_size, interpolation=cv2.INTER_LINEAR)
            bb = (0, 0, w, h)
            
            self._log("✅ 전체 이미지를 브레드보드로 설정")
            return warped, bb
            
        except Exception as e:
            self._log(f"이미지 처리 오류: {e}")
            return None
    
    def _detect_and_transform_breadboard(self, img):
        """브레드보드 검출 및 perspective transform"""
        try:
            # FasterRCNN으로 브레드보드 검출
            detections = self.detector.detect(img)
            breadboard_boxes = [box for cls, conf, box in detections 
                              if cls.lower() == 'breadboard' and conf > 0.5]
            
            if not breadboard_boxes:
                # 전체 이미지를 브레드보드로 간주
                h, w = img.shape[:2]
                bb = (0, 0, w, h)
                self._log("⚠️ 브레드보드 자동 검출 실패, 전체 이미지 사용")
            else:
                # 가장 큰 브레드보드 선택
                bb = max(breadboard_boxes, key=lambda x: (x[2]-x[0])*(x[3]-x[1]))
                self._log("✅ 브레드보드 자동 검출 성공")
            
            # 간단한 perspective transform (크롭 + 리사이즈)
            x1, y1, x2, y2 = bb
            cropped = img[y1:y2, x1:x2]
            
            # 표준 크기로 리사이즈
            warped = cv2.resize(cropped, self.display_size, interpolation=cv2.INTER_LINEAR)
            
            return warped, bb
            
        except Exception as e:
            self._log(f"브레드보드 변환 오류: {e}")
            return None
    
    def detect_pins_advanced(self, components, image_path, warped_image_b64):
        """
        4단계: 개선된 핀 위치 자동 검출 (실제 pin_manager 사용)
        
        Args:
            components: 검출된 컴포넌트 리스트
            image_path: 원본 이미지 경로
            warped_image_b64: warped 이미지 base64
            
        Returns:
            dict: {
                'success': bool,
                'component_pins': list,
                'pin_image_b64': str,  # 핀이 표시된 이미지
                'holes': list,  # 구멍 데이터
                'logs': list,
                'progress': int
            }
        """
        try:
            self._log("4단계: 고급 핀 위치 자동 검출 중...", 50)
            
            # PinManager 초기화 (지연 로딩)
            if self.pin_manager is None:
                self._log("⚠️ PinManager가 없어 초기화 시도", 52)
                self._initialize_managers()

            # warped 이미지 복원
            if isinstance(warped_image_b64, str):
                warped_img = self._base64_to_image(warped_image_b64)
            else:
                warped_img = self.warped_image

            # 실제 PinManager 사용
            if self.pin_manager and hasattr(self.pin_manager, 'auto_pin_detection'):
                self._log("✅ 실제 PinManager 사용하여 핀 검출", 55)
                component_pins, holes = self.pin_manager.auto_pin_detection(
                    warped_img, components, self.original_image, self.original_bb
                )
            else:
                self._log("⚠️ PinManager가 없어 fallback 핀 검출 사용", 55)
                component_pins, holes = self._fallback_pin_detection(components, warped_img)
            
            # warped 이미지가 base64인 경우 복원
            if isinstance(warped_image_b64, str):
                warped_img = self._base64_to_image(warped_image_b64)
            else:
                warped_img = self.warped_image
            
            # PinManager가 있고 auto_pin_detection 메서드가 있는 경우 사용
            if self.pin_manager and hasattr(self.pin_manager, 'auto_pin_detection'):
                try:
                    self._log("✅ 실제 PinManager 사용하여 핀 검출", 55)
                    component_pins, holes = self.pin_manager.auto_pin_detection(
                        warped_img, components, self.original_image, self.original_bb
                    )
                    
                    self._log(f"✅ PinManager로 {len(holes)}개 구멍 및 모든 컴포넌트 핀 검출 완료", 60)
                    
                except Exception as e:
                    self._log(f"⚠️ PinManager 오류: {e}, fallback 사용")
                    component_pins, holes = self._fallback_pin_detection(components, warped_img)
            else:
                self._log("⚠️ PinManager가 없어 fallback 핀 검출 사용", 55)
                component_pins, holes = self._fallback_pin_detection(components, warped_img)
            
            # 핀이 표시된 이미지 생성
            pin_image = self._draw_pins_on_image(warped_img.copy(), component_pins)
            
            return {
                'success': True,
                'component_pins': component_pins,
                'pin_image_b64': self._image_to_base64(pin_image),
                'holes': holes,
                'logs': self.logs.copy(),
                'progress': self.progress
            }
            
        except Exception as e:
            self._log(f"❌ 핀 검출 오류: {e}")
            return {
                'success': False,
                'component_pins': [],
                'pin_image_b64': None,
                'holes': [],
                'logs': self.logs.copy(),
                'progress': self.progress
            }
    
    def _fallback_pin_detection(self, components, warped_img):
        """fallback 핀 검출 (기본 방법)"""
        try:
            # 구멍 검출
            if self.hole_detector and hasattr(self.hole_detector, 'detect_holes'):
                holes = self.hole_detector.detect_holes(warped_img)
                self._log(f"✅ {len(holes)}개 구멍 검출됨", 55)
            else:
                holes = self._generate_virtual_holes()
                self._log("⚠️ 가상 구멍 데이터 사용", 55)
            
            # 컴포넌트별 핀 검출
            component_pins = []
            for i, (cls, conf, box) in enumerate(components):
                x1, y1, x2, y2 = box
                expected_pins = 8 if cls == 'IC' else 2
                
                try:
                    # 실제 검출기가 있으면 사용
                    pins = self._detect_component_pins_advanced(cls, box, holes, warped_img)
                    
                    # 실패시 기본 위치 사용
                    if len(pins) != expected_pins:
                        pins = self._get_smart_default_pins(cls, x1, y1, x2, y2, holes)
                    
                    component_pins.append({
                        'class': cls,
                        'box': box,
                        'pins': pins,
                        'value': 100.0 if cls == 'Resistor' else 0.001 if cls == 'Capacitor' else 0.0,
                        'num_idx': i + 1
                    })
                    
                    self._log(f"  {cls} #{i+1}: {len(pins)}개 핀 설정", 50 + (i+1)*5)
                    
                except Exception as e:
                    self._log(f"⚠️ {cls} 핀 검출 오류: {e}")
                    # 기본값으로 fallback
                    pins = self._get_smart_default_pins(cls, x1, y1, x2, y2, holes)
                    component_pins.append({
                        'class': cls,
                        'box': box,
                        'pins': pins,
                        'value': 100.0 if cls == 'Resistor' else 0.0,
                        'num_idx': i + 1
                    })
            
            return component_pins, holes
            
        except Exception as e:
            self._log(f"fallback 핀 검출 오류: {e}")
            return [], []
    
    def _detect_component_pins_advanced(self, component_class, box, holes, warped_img):
        """개선된 컴포넌트별 핀 검출"""
        x1, y1, x2, y2 = box
        
        # 실제 검출기 시도
        if self.pin_manager:
            try:
                # 개별 검출기들 사용 시도
                detectors = getattr(self.pin_manager, '__dict__', {})
                
                if component_class == 'Resistor' and 'resistor_det' in detectors:
                    result = detectors['resistor_det'].extract(warped_img, box)
                    if result and len(result) == 2:
                        return list(result)
                        
                elif component_class == 'LED' and 'led_det' in detectors:
                    result = detectors['led_det'].extract(warped_img, box, holes)
                    if result and len(result) == 2:
                        return list(result)
                        
                elif component_class == 'IC' and 'ic_det' in detectors:
                    roi = warped_img[y1:y2, x1:x2]
                    dets = detectors['ic_det'].detect(roi)
                    if dets and 'pin_points' in dets[0]:
                        return [(x1+px, y1+py) for px, py in dets[0]['pin_points']]
                        
            except Exception as e:
                self._log(f"개별 검출기 오류: {e}")
        
        # 기본 검출 로직으로 fallback
        return self._detect_component_pins_basic(component_class, box, holes)
    
    def _detect_component_pins_basic(self, component_class, box, holes):
        """기본 컴포넌트별 핀 검출"""
        x1, y1, x2, y2 = box
        
        if component_class == 'IC':
            # IC는 8개 핀 (양쪽 4개씩)
            w, h = x2 - x1, y2 - y1
            pins = []
            # 왼쪽 4개
            for i in range(4):
                y = y1 + h * (i + 1) / 5
                pins.append((x1 + 5, int(y)))
            # 오른쪽 4개
            for i in range(4):
                y = y1 + h * (4 - i) / 5
                pins.append((x2 - 5, int(y)))
            return pins
        else:
            # 저항, LED, 다이오드, 캐패시터는 2개 핀
            center_y = (y1 + y2) // 2
            return [(x1 + 10, center_y), (x2 - 10, center_y)]
    
    def _get_smart_default_pins(self, component_class, x1, y1, x2, y2, holes):
        """구멍 위치를 고려한 스마트 기본 핀 생성"""
        # 기본 핀 위치 생성
        basic_pins = self._detect_component_pins_basic(component_class, (x1, y1, x2, y2), holes)
        
        # 구멍에 스냅
        snapped_pins = []
        for pin in basic_pins:
            snapped_pin = self._snap_to_nearest_hole(pin, holes)
            snapped_pins.append(snapped_pin)
        
        return snapped_pins
    
    def _snap_to_nearest_hole(self, pin_pos, holes, max_distance=25):
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
    
    def _initialize_managers(self):
        """PinManager 및 관련 매니저 초기화 (simple_main.py 스타일)"""
        from pin_manager import PinManager
        from ComponentEditor import ComponentEditor
        from circuit_generator_manager import CircuitGeneratorManager
        from detector.resistor_detector import ResistorEndpointDetector
        from detector.led_detector import LedEndpointDetector
        from detector.wire_detector import WireDetector
        from detector.cap_detector import CapEndpointDetector
        from detector.diode_detector import ResistorEndpointDetector as DiodeEndpointDetector
        from detector.ic_chip_detector import ICChipPinDetector

        # 1) 개별 detector 초기화
        detectors = {
            'resistor': ResistorEndpointDetector(),
            'led': LedEndpointDetector(),
            'diode': DiodeEndpointDetector(),
            'ic': ICChipPinDetector(),
            'capacitor': CapEndpointDetector(),
            'wire': WireDetector(kernel_size=4),
            'hole': self.hole_detector  # HoleDetector는 이미 초기화됨
        }

        # 2) PinManager, ComponentEditor, CircuitGenerator 초기화
        self.pin_manager = PinManager(self.class_colors, detectors)
        self.component_editor = ComponentEditor(self.class_colors)
        self.circuit_generator = CircuitGeneratorManager(self.hole_detector)

        self._log("✅ PinManager 및 매니저 초기화 완료")


    
    def detect_pins(self, components, image_path):
        """
        기존 detect_pins 메서드 (하위 호환성)
        """
        return self.detect_pins_advanced(components, image_path, self.warped_image)
    
    def generate_circuit_and_analyze(self, component_pins, voltage, reference_circuit):
        """
        5-7단계: 회로 생성, SPICE 분석, AI 피드백
        
        Args:
            component_pins: 핀이 설정된 컴포넌트 리스트
            voltage: 전원 전압
            reference_circuit: 기준 회로 ID
            
        Returns:
            dict: {
                'success': bool,
                'analysis_text': str,
                'output_files': list,
                'circuit_data': dict,
                'logs': list,
                'progress': int
            }
        """
        try:
            self._log("5단계: 회로 생성 및 SPICE 분석 중...", 70)
            
            # CircuitGenerator 초기화
            if self.circuit_generator is None:
                self._initialize_managers()
            
            # 기준 회로 설정
            if reference_circuit != 'skip':
                if self.circuit_generator and hasattr(self.circuit_generator, 'select_reference_circuit'):
                    self.circuit_generator.select_reference_circuit(reference_circuit)
                self._log(f"기준 회로 선택: {reference_circuit}", 75)
            
            self._log("6단계: 전원 및 연결 분석 중...", 80)
            
            # 가상의 구멍 데이터 생성 (실제로는 hole_detector 결과 사용)
            holes = self._generate_virtual_holes()
            
            # 전원 설정
            power_sources = [{'voltage': voltage, 'pos_pin': (100, 100), 'neg_pin': (100, 200)}]
            
            # 회로 생성 시도
            try:
                if self.circuit_generator and hasattr(self.circuit_generator, 'generate_final_circuit'):
                    success, feedback_data = self.circuit_generator.generate_final_circuit(
                        component_pins, holes, power_sources, self.warped_image
                    )
                else:
                    # fallback: 간단한 회로 생성
                    success, feedback_data = self._generate_simple_circuit(
                        component_pins, power_sources, voltage
                    )
                
                if not success:
                    raise RuntimeError("회로 생성 실패")
                
            except Exception as e:
                self._log(f"회로 생성 오류: {e}, 간단한 분석으로 대체")
                success, feedback_data = self._generate_simple_circuit(
                    component_pins, power_sources, voltage
                )
            
            self._log("7단계: AI 분석 생성 중...", 90)
            
            # AI 분석 텍스트 생성
            analysis_text = self._generate_analysis_text(
                component_pins, feedback_data, reference_circuit
            )
            
            # 출력 파일 목록
            output_files = ['circuit.spice', 'circuit_graph.png']
            
            self._log("✅ 모든 분석 완료!", 100)
            
            return {
                'success': True,
                'analysis_text': analysis_text,
                'output_files': output_files,
                'circuit_data': feedback_data,
                'logs': self.logs.copy(),
                'progress': self.progress
            }
            
        except Exception as e:
            self._log(f"❌ 회로 분석 오류: {e}")
            return {
                'success': False,
                'analysis_text': f'분석 중 오류가 발생했습니다: {str(e)}',
                'output_files': [],
                'circuit_data': {},
                'logs': self.logs.copy(),
                'progress': self.progress
            }
    
    def _generate_virtual_holes(self):
        """가상의 브레드보드 구멍 데이터 생성"""
        holes = []
        # 표준 브레드보드 구멍 패턴 (30x10)
        for row in range(10):
            for col in range(30):
                x = 40 + col * 40  # 40픽셀 간격
                y = 40 + row * 40
                holes.append((x, y))
        return holes
    
    def _generate_simple_circuit(self, component_pins, power_sources, voltage):
        """간단한 회로 생성 (fallback)"""
        try:
            # 간단한 피드백 데이터 생성
            feedback_data = {
                'component_count': len(component_pins),
                'power_count': len(power_sources),
                'voltage': voltage,
                'errors': [],
                'warnings': [],
                'similarity_score': 0.8,
                'reference_circuit': 'auto_generated'
            }
            
            # 간단한 SPICE 파일 생성
            spice_content = f"""* Auto-generated circuit
.title Simple Circuit Analysis
V1 1 0 {voltage}V
"""
            
            # 컴포넌트 추가
            for i, comp in enumerate(component_pins):
                if comp['class'] == 'Resistor':
                    spice_content += f"R{i+1} 1 2 {comp.get('value', 100)}\n"
                elif comp['class'] == 'LED':
                    spice_content += f"D{i+1} 1 2 LED_MODEL\n"
                elif comp['class'] == 'Capacitor':
                    spice_content += f"C{i+1} 1 2 {comp.get('value', 0.001)}\n"
            
            spice_content += """
.model LED_MODEL D(IS=1e-12 N=2)
.end
"""
            
            # 파일 저장
            with open('circuit.spice', 'w') as f:
                f.write(spice_content)
            
            return True, feedback_data
            
        except Exception as e:
            self._log(f"간단한 회로 생성 오류: {e}")
            return False, {}
    
    def _generate_analysis_text(self, component_pins, feedback_data, reference_circuit):  
        """  
        Gemini LLM에 create_rag_prompt 형식의 프롬프트를 보내서  
        AI 분석 텍스트를 그대로 받아 반환합니다.  
        """  
        try:  
            # 1) feedback_data를 사람이 읽기 좋은 문자열로 포맷  
            from llm_feedback_manager import LLMFeedbackManager  
            manager = LLMFeedbackManager(practice_circuit_topic="")  
            context = manager._format_analysis_context(feedback_data)  # :contentReference[oaicite:4]{index=4}  
  
            # 2) RAG 프롬프트 생성 (첫 턴 고정)  
            prompt = create_rag_prompt(  
                user_query="",  
                context=context,  
                is_first_turn=True,  
                practice_circuit_topic=""  
            )  # :contentReference[oaicite:5]{index=5}  
  
            # 3) 모델이 미초기화 상태면 다시 초기화  
            if not self.model or not self.generation_config:  
                self.model, self.generation_config = initialize_gemini()  
  
            # 4) Gemini 호출  
            response = self.model.generate_content(  
                prompt,  
                generation_config=self.generation_config  
            )  
            return response.text  
        except Exception as e:  
            self._log(f"❌ 분석 텍스트 생성 오류: {e}")  
            return f"분석 텍스트 생성 중 오류 발생: {str(e)}" 

    
    def get_session_info(self):
        """현재 세션 정보 반환"""
        return {
            'logs': self.logs.copy(),
            'progress': self.progress,
            'current_step': self.current_step,
            'has_warped_image': self.warped_image is not None,
            'analysis_data': self.analysis_data.copy()
        }
    
    def cleanup(self):
        """리소스 정리"""
        try:
            # 임시 파일들 정리
            temp_files = ['circuit.spice', 'circuit_graph.png', 'circuit.jpg']
            for file_path in temp_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except:
                        pass
            
            # 메모리 정리
            self.warped_image = None
            self.original_image = None
            self.analysis_data.clear()
            self.logs.clear()
            
        except Exception as e:
            print(f"정리 중 오류: {e}")

# 테스트 코드
if __name__ == "__main__":
    print("🧪 WebRunnerComplete 테스트")
    
    runner = WebRunnerComplete()
    print(f"✅ 초기화 완료")
    print(f"검출기 상태: {runner.detector is not None}")
    print(f"구멍 검출기 상태: {runner.hole_detector is not None}")
    print(f"PinManager 상태: {runner.pin_manager is not None}")
    
    session_info = runner.get_session_info()
    print(f"세션 정보: {session_info}")
    
    # 정리
    runner.cleanup()
    print("🧹 정리 완료")