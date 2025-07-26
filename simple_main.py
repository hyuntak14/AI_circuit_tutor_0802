# main.py (LLM 피드백 통합 버전) - 다중 전원 지원 + 종합 AI 분석
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
from llm_feedback_manager import LLMFeedbackManager


class SimpleCircuitConverter:
    def __init__(self):
        # 디스플레이 크기 설정
        self.display_size = (1200, 1200)
        self.root = tk.Tk() # 메인 Tkinter 인스턴스 생성
        self.root.withdraw() # 메인 창은 숨김
        
        # 기본 검출기들 초기화
        self.detector = FasterRCNNDetector(r'D:/Hyuntak/lab/AR_circuit_tutor/breadboard_project/model/fasterrcnn_v2.pt')
        self.hole_det = HoleDetector(
            template_csv_path='detector/template_holes_complete.csv',
            template_image_path='detector/breadboard18.jpg',
            max_nn_dist=20.0
        )
        self.llm_manager = None
        # ▶ 선택된 회로 번호·토픽명 저장용
        self.selected_circuit = None
        self.practice_circuit_topic = ""
        
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
        
        # LLM 피드백 매니저 초기화 (지연 로딩)
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

    def _select_reference_circuit_gui(self):
        """GUI를 통해 기준 회로를 선택하는 함수"""
        import tkinter as tk
        from tkinter import ttk, messagebox
        
        class CircuitSelector:
            def __init__(self):
                self.selected_circuit = None
                self.root = tk.Tk()
                self.root.title("기준 회로 선택")
                self.root.geometry("500x600")
                self.root.resizable(False, False)
                
                # 창을 화면 중앙에 배치
                self.root.update_idletasks()
                x = (self.root.winfo_screenwidth() // 2) - (500 // 2)
                y = (self.root.winfo_screenheight() // 2) - (600 // 2)
                self.root.geometry(f"500x600+{x}+{y}")
                
                # 스타일 설정
                style = ttk.Style()
                style.theme_use('clam')
                
                self.setup_ui()
                
            def setup_ui(self):
                # 메인 프레임
                main_frame = ttk.Frame(self.root, padding="20")
                main_frame.pack(fill=tk.BOTH, expand=True)
                
                # 제목
                title_label = ttk.Label(main_frame, text="🎯 기준 회로 선택", 
                                      font=("Arial", 16, "bold"))
                title_label.pack(pady=(0, 10))
                
                # 설명
                desc_label = ttk.Label(main_frame, 
                                     text="생성된 회로와 비교할 기준 회로를 선택하세요.\n"
                                          "선택한 회로와 유사도를 분석합니다.",
                                     font=("Arial", 10))
                desc_label.pack(pady=(0, 20))
                
                # 회로 목록 프레임
                list_frame = ttk.LabelFrame(main_frame, text="실습 주제 목록", padding="10")
                list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
                
                # 스크롤바와 리스트박스
                scrollbar = ttk.Scrollbar(list_frame)
                scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                
                self.listbox = tk.Listbox(list_frame, 
                                        yscrollcommand=scrollbar.set,
                                        font=("Arial", 11),
                                        height=12,
                                        selectmode=tk.SINGLE,
                                        activestyle='dotbox')
                self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                scrollbar.config(command=self.listbox.yview)
                
                # 회로 목록 추가
                self.circuit_topics = {
                    1: "병렬회로", 2: "직렬회로", 3: "키르히호프 1법칙", 4: "키르히호프 2법칙",
                    5: "중첩의 원리-a",6: "중첩의 원리-b",7: "중첩의 원리-c",8: "교류 전원", 9: "오실로스코프1",
                    10: "반파정류회로", 11: "반파정류회로2", 12: "비반전 증폭기"
                }
                
                for i in range(1, len(self.circuit_topics)):
                    topic = self.circuit_topics.get(i, f"회로 {i}")
                    self.listbox.insert(tk.END, f"{i:2d}. {topic}")
                
                # 기본 선택 (첫 번째 항목)
                self.listbox.selection_set(0)
                
                # 더블클릭 이벤트
                self.listbox.bind('<Double-1>', self.on_double_click)
                
                # 버튼 프레임
                button_frame = ttk.Frame(main_frame)
                button_frame.pack(fill=tk.X, pady=(10, 0))
                
                # 건너뛰기 버튼
                skip_button = ttk.Button(button_frame, text="비교 안함", 
                                       command=self.on_skip)
                skip_button.pack(side=tk.LEFT)
                
                # 취소 버튼
                cancel_button = ttk.Button(button_frame, text="취소", 
                                         command=self.on_cancel)
                cancel_button.pack(side=tk.RIGHT)
                
                # 선택 버튼
                select_button = ttk.Button(button_frame, text="선택", 
                                         command=self.on_select)
                select_button.pack(side=tk.RIGHT, padx=(0, 10))
                
                # 엔터 키 바인딩
                self.root.bind('<Return>', lambda e: self.on_select())
                self.root.bind('<Escape>', lambda e: self.on_cancel())
                
                # 포커스 설정
                self.listbox.focus_set()
                
            def on_double_click(self, event):
                """더블클릭 시 바로 선택"""
                self.on_select()
                
            def on_select(self):
                """선택 버튼 클릭"""
                selection = self.listbox.curselection()
                if selection:
                    circuit_num = selection[0] + 1  # 1부터 시작
                    self.selected_circuit = circuit_num
                    self.root.quit()
                    self.root.destroy()
                else:
                    messagebox.showwarning("선택 오류", "회로를 선택해주세요.")
                    
            def on_cancel(self):
                """취소 버튼 클릭"""
                self.selected_circuit = None
                self.root.quit()
                self.root.destroy()
                
            def on_skip(self):
                """건너뛰기 버튼 클릭"""
                self.selected_circuit = "skip"
                self.root.quit()
                self.root.destroy()
                
            def show(self):
                """다이얼로그 표시 및 결과 반환"""
                self.root.mainloop()
                return self.selected_circuit
        
        # 다이얼로그 실행
        try:
            selector = CircuitSelector()
            return selector.show()
        except Exception as e:
            print(f"⚠️ 기준 회로 선택 GUI 오류: {e}")
            return None

    def _initialize_llm_manager(self):
        """LLM 매니저 초기화 (지연 로딩)"""
        if self.llm_manager is None:
            try:
                topic_str = f"circuit{self.selected_circuit}_{self.practice_circuit_topic}"
                self.llm_manager = LLMFeedbackManager(practice_circuit_topic=topic_str)
                return True
            except Exception as e:
                print(f"⚠️ LLM 시스템 초기화 실패: {e}")
                print("   회로 변환은 계속 진행되지만 AI 피드백은 제공되지 않습니다.")
                return False
        return True

    def _provide_comprehensive_llm_feedback(self, component_pins, feedback_data):
            """종합적인 LLM 피드백 UI를 제공합니다."""
            if not self._initialize_llm_manager():
                return
            
            spice_file = "circuit.spice"
            if not os.path.exists(spice_file):
                print("⚠️ SPICE 파일을 찾을 수 없어 AI 피드백을 제공할 수 없습니다.")
                return

            print("\n" + "🤖" + "="*59)
            print("AI 기반 종합 회로 분석을 위한 데이터 생성 중...")
            print("="*60)

            # 분석 상황에 따른 맞춤형 질문 생성
            similarity = feedback_data.get('similarity_score', 0)
            errors = feedback_data.get('errors', [])
            reference = feedback_data.get('reference_circuit', 'Unknown')
            
            if len(errors) > 0:
                analysis_query = f"이 회로에 {len(errors)}개의 오류가 감지되었습니다. 각 오류의 원인과 해결책을 분석해주세요."
            elif similarity < 0.7:
                analysis_query = f"기준 회로({reference})와의 유사도가 {similarity:.1%}로 낮습니다. 어떤 부분을 개선해야 할지 분석해주세요."
            else:
                analysis_query = f"이 회로는 기준 회로({reference})와 {similarity:.1%}의 높은 유사도를 보입니다. 회로의 동작 원리와 특성을 설명해주세요."
            
            # 초기 분석 텍스트 생성
            initial_analysis_text = self.llm_manager.get_initial_analysis_text(
                spice_file, component_pins, feedback_data, analysis_query
            )
            
            if "❌" in initial_analysis_text:
                print(f"AI 분석 중 오류가 발생했습니다: {initial_analysis_text}")
                return

            print("✅ AI 분석 생성 완료! UI를 실행합니다.")
            
            # AI 채팅 UI 시작
            self.llm_manager.start_chat_ui(self.root, initial_analysis_text)
            # UI 창이 닫힐 때까지 기다리게 하려면 mainloop를 호출해야 합니다.
            # 이 컨텍스트에서는 Toplevel이므로, 메인 프로그램이 계속 실행되도록 둡니다.

    def run(self):
        """전체 프로세스 실행 - UI 기반 AI 피드백"""
        try:
            print("=" * 60)
            print("🔌 간소화된 브레드보드 → 회로도 변환기")
            print("   (UI 기반 AI 분석 기능 포함)")
            print("=" * 60)
            
            # ... run 메서드의 기존 로직 (기준 회로 선택, 이미지 로드, 검출 등)은 그대로 유지 ...
            
            # 🎯 1. 기준 회로 선택 GUI
            print("\n🎯 기준 회로 선택 단계")
            selected_circuit = self._select_reference_circuit_gui() # self.root를 사용하도록 수정 가능
            
            if selected_circuit is None:
                print("❌ 프로그램이 취소되었습니다.")
                return
            elif selected_circuit == "skip":
                print("📋 회로 비교 기능을 사용하지 않습니다.")
                use_reference = False
            else:
                print(f"✅ 선택된 기준 회로: {selected_circuit}")
                use_reference = True
                reference_selected = self.circuit_generator.select_reference_circuit(selected_circuit)
                if reference_selected:
                    self.selected_circuit = selected_circuit
                    self.practice_circuit_topic = self.circuit_generator.reference_circuit_topic
                    print(f"✅ 실습 주제 설정: {self.practice_circuit_topic}")
                else:
                    use_reference = False

            # ... [이미지 로드]부터 [회로 생성 및 분석]까지의 로직은 그대로 유지 ...
            print("\n📷 이미지 로드 단계")
            img = self.load_image()
            if img is None:
                print("❌ 이미지를 선택하지 않았습니다.")
                return
            
            print("\n🔍 브레드보드 검출 단계")
            result = self.auto_detect_and_transform(img)
            if result is None: return
            warped, original_bb = result
            
            print("\n🔧 컴포넌트 검출 단계")
            components = self.component_editor.quick_component_detection(warped, self.detector)
            if not components: return
            
            print("\n📍 핀 검출 단계")
            component_pins, holes = self.pin_manager.auto_pin_detection(warped, components, img, original_bb)
            
            print("\n✏️ 핀 위치 확인 단계")
            component_pins = self.pin_manager.manual_pin_verification_and_correction(warped, component_pins, holes)
            
            print("\n📝 컴포넌트 값 입력 단계")
            self.circuit_generator.quick_value_input(warped, component_pins)
            
            print("\n🔋 전원 설정 단계")
            power_sources = self.circuit_generator.quick_power_selection(warped, component_pins)
            if not power_sources: return
            
            print("\n🔧 회로도 생성 및 분석 단계")
            success, feedback_data = self.circuit_generator.generate_final_circuit(
                component_pins, holes, power_sources, warped
            )
            # ...

            if success:
                print("\n🎉 회로 변환 완료!")
                # ... (생성 파일 안내 등 기존 출력 유지) ...

                # 🤖 10. 종합 AI 분석 UI 제공
                if feedback_data:
                    self._provide_comprehensive_llm_feedback(component_pins, feedback_data)
                    print("\nAI 분석 창이 열렸습니다. 프로그램은 백그라운드에서 계속 실행됩니다.")
                    # UI가 이벤트를 처리할 수 있도록 mainloop를 호출합니다.
                    self.root.mainloop()

                else:
                    print("\n⚠️ 회로 분석 데이터가 없어 AI 분석을 건너뜁니다.")
            else:
                print("\n❌ 회로 생성에 실패했습니다.")
        
        finally:
            # 프로그램 종료 시 Tkinter 루트 창 확실히 닫기
            if self.root:
                self.root.destroy()


if __name__ == "__main__":
    converter = SimpleCircuitConverter()
    converter.run()