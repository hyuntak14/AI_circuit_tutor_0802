# main.py (간소화된 버전) - 다중 전원 지원 + LLM 피드백 추가
import os
import matplotlib
matplotlib.use('Qt5Agg')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import tkinter as tk
from tkinter import filedialog

from detector.fasterrcnn_detector import FasterRCNNDetector
from detector.hole_detector import HoleDetector
from detector.resistor_detector import ResistorEndpointDetector
from detector.led_detector import LedEndpointDetector
from detector.wire_detector import WireDetector
from detector.cap_detector import CapEndpointDetector
from detector.diode_detector import ResistorEndpointDetector as DiodeEndpointDetector
from detector.ic_chip_detector import ICChipPinDetector
from ui.perspective_editor import select_and_transform

# 새로 분리된 클래스들 import
from ComponentEditor import ComponentEditor
from pin_manager import PinManager
from circuit_generator_manager import CircuitGeneratorManager
from llm_feedback_manager import LLMFeedbackManager  # 새로 추가

class SimpleCircuitConverter:
    def __init__(self):
        # 디스플레이 크기 설정
        self.display_size = (1200, 1200)
        
        # 기본 검출기들 초기화
        self.detector = FasterRCNNDetector(r'D:/Hyuntak/lab/AR_circuit_tutor/breadboard_project/model/fasterrcnn_v2.pt')
        self.hole_det = HoleDetector(
            template_csv_path='detector/template_holes_complete.csv',
            template_image_path='detector/breadboard18.jpg',
            max_nn_dist=20.0
        )
        
        # 컴포넌트별 검출기들
        detectors = {
            'resistor': ResistorEndpointDetector(),
            'led': LedEndpointDetector(),
            'diode': DiodeEndpointDetector(),
            'ic': ICChipPinDetector(),
            'capacitor': CapEndpointDetector(),
            'wire': WireDetector(kernel_size=4),
            'hole': self.hole_det
        }
        
        # 컴포넌트 색상
        self.class_colors = {
            'Breadboard': (0, 128, 255),
            'Capacitor': (255, 0, 255),
            'Diode': (0, 255, 0),
            'IC': (204, 102, 255),
            'LED': (102, 0, 102),
            'Line_area': (255, 0, 0),
            'Resistor': (200, 170, 0)
        }
        
        # 기능별 매니저 클래스들 초기화
        self.component_editor = ComponentEditor(self.class_colors)
        self.pin_manager = PinManager(self.class_colors, detectors)
        self.circuit_generator = CircuitGeneratorManager(self.hole_det)
        
        # LLM 피드백 매니저 초기화 (새로 추가)
        self.llm_manager = None

    def _resize_for_display(self, image):
        """이미지를 1200x1200 크기로 리사이즈"""
        h, w = image.shape[:2]
        scale = min(self.display_size[0] / w, self.display_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        # 중앙 배치를 위한 패딩
        pad_w = (self.display_size[0] - new_w) // 2
        pad_h = (self.display_size[1] - new_h) // 2
        padded = cv2.copyMakeBorder(resized, pad_h, self.display_size[1] - new_h - pad_h,
                                   pad_w, self.display_size[0] - new_w - pad_w,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return padded

    def load_image(self):
        """이미지 파일 선택 및 로드"""
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.askopenfilename(
            title="브레드보드 이미지 선택",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        root.destroy()
        
        if not file_path:
            return None
            
        return cv2.imread(file_path)

    def auto_detect_and_transform(self, img):
        """자동 브레드보드 검출 및 변환"""
        print("🔍 브레드보드 자동 검출 중...")
        comps = self.detector.detect(img)
        bb = next((b for c, _, b in comps if c.lower() == 'breadboard'), None)
        
        if bb is None:
            print("❌ 브레드보드를 찾을 수 없습니다.")
            return None
            
        print("✅ 브레드보드 검출 완료")
        warped, _ = select_and_transform(img.copy(), bb)
        return warped, bb  # 원본 bounding box도 반환

    def _initialize_llm_manager(self):
        """LLM 매니저 초기화 (지연 로딩)"""
        if self.llm_manager is None:
            try:
                self.llm_manager = LLMFeedbackManager()
                return True
            except Exception as e:
                print(f"⚠️  LLM 시스템 초기화 실패: {e}")
                print("   회로 변환은 계속 진행되지만 AI 피드백은 제공되지 않습니다.")
                return False
        return True

    def _provide_llm_feedback(self, component_pins):
        """LLM 피드백 제공"""
        if not self._initialize_llm_manager():
            return
        
        spice_file = "circuit.spice"
        if not os.path.exists(spice_file):
            print("⚠️  SPICE 파일을 찾을 수 없어 AI 피드백을 제공할 수 없습니다.")
            return
        
        print("\n" + "="*60)
        print("🤖 AI 회로 분석 및 피드백")
        print("="*60)
        
        # 초기 회로 분석 피드백
        feedback_success = self.llm_manager.provide_initial_feedback(
            spice_file, 
            component_pins,
            "제가 구성한 브레드보드 회로를 분석해주세요."
        )
        
        if not feedback_success:
            print("❌ AI 피드백 제공에 실패했습니다.")
            return
        
        # 사용자가 추가 질문을 원하는지 확인
        print("\n" + "-"*60)
        while True:
            try:
                user_choice = input("\n💬 AI와 더 대화하시겠습니까? (y/n): ").strip().lower()
                
                if user_choice in ['y', 'yes', '예', 'ㅇ']:
                    self.llm_manager.start_interactive_chat()
                    break
                elif user_choice in ['n', 'no', '아니오', 'ㄴ']:
                    print("👍 AI 피드백을 완료합니다.")
                    break
                else:
                    print("y 또는 n을 입력해주세요.")
                    
            except KeyboardInterrupt:
                print("\n👍 AI 피드백을 완료합니다.")
                break

    def run(self):
        """전체 프로세스 실행 - 다중 전원 지원 + LLM 피드백"""
        print("=" * 50)
        print("🔌 간소화된 브레드보드 → 회로도 변환기 (AI 피드백 포함)")
        print("=" * 50)
        
        # 1. 이미지 로드
        img = self.load_image()
        if img is None:
            print("❌ 이미지를 선택하지 않았습니다.")
            return
        
        # 2. 브레드보드 자동 검출 및 변환
        result = self.auto_detect_and_transform(img)
        if result is None:
            return
        warped, original_bb = result  # warped와 원본 bbox 둘 다 받기
        
        # 3. 컴포넌트 검출 및 편집 (ComponentEditor 사용)
        components = self.component_editor.quick_component_detection(warped, self.detector)
        if not components:
            print("❌ 컴포넌트가 검출되지 않았습니다.")
            return
        
        # 4. 핀 검출 (PinManager 사용) - 원본 이미지와 bbox 전달
        component_pins, holes = self.pin_manager.auto_pin_detection(warped, components, img, original_bb)
        
        # 5. 핀 위치 확인 및 수정 단계 (PinManager 사용)
        component_pins = self.pin_manager.manual_pin_verification_and_correction(warped, component_pins, holes)
        
        # 6. 값 입력 (CircuitGeneratorManager 사용)
        self.circuit_generator.quick_value_input(warped,component_pins)
        
        # 7. 다중 전원 선택 (수정된 CircuitGeneratorManager 사용)
        print("\n🔋 전원 설정 단계")
        power_sources = self.circuit_generator.quick_power_selection(warped, component_pins)
        
        if not power_sources:
            print("❌ 전원이 설정되지 않았습니다.")
            return
        
        # 전원 정보 출력
        print(f"\n📊 설정된 전원 정보:")
        for i, (voltage, plus_pt, minus_pt) in enumerate(power_sources, 1):
            print(f"  전원 {i}: {voltage}V, 양극 {plus_pt}, 음극 {minus_pt}")
        
        # 8. 회로 생성 (수정된 CircuitGeneratorManager 사용)
        print("\n🔧 회로도 생성 단계")
        success = self.circuit_generator.generate_final_circuit(
            component_pins, holes, power_sources, warped
        )
        
        if success:
            print("\n🎉 다중 전원 회로 변환 완료!")
            print("📁 생성된 파일:")
            print("  - circuit.jpg (메인 회로도)")
            print("  - circuit.spice (SPICE 넷리스트)")
            
            # 다중 전원에 따른 추가 파일들 안내
            if len(power_sources) > 1:
                print("  - 추가 전원별 회로도:")
                for i in range(2, len(power_sources) + 1):
                    print(f"    - circuit_pwr{i}.jpg")
            
            # 연결성 그래프 파일도 안내
            print("  - circuit_graph.png (연결성 그래프)")
            print("  - circuit.graphml (그래프 데이터)")
            
            print(f"\n✨ 총 {len(power_sources)}개의 전원을 가진 회로가 성공적으로 생성되었습니다!")
            
            # 9. LLM 피드백 제공 (새로 추가)
            print("\n🤖 AI 회로 분석을 시작합니다...")
            self._provide_llm_feedback(component_pins)
            
        else:
            print("\n❌ 회로 생성에 실패했습니다.")

if __name__ == "__main__":
    converter = SimpleCircuitConverter()
    converter.run()