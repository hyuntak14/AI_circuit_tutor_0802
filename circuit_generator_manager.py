# circuit_generator_manager.py (LLM 피드백 통합 버전)
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
from circuit_generator import generate_circuit
from checker.error_checker import ErrorChecker
import os
import glob
import networkx as nx
from checker.Circuit_comparer import CircuitComparer
import unicodedata
import re

class CircuitGeneratorManager:
    def __init__(self, hole_detector):
        self.hole_det = hole_detector
        self.reference_circuit_path = None  # 선택된 기준 회로 경로
        self.reference_circuit_topic = "Unknown"  # 선택된 기준 회로 주제
        
        # 회로 주제 맵핑 (기존 topic_map 확장)
        self.topic_map = {
                    1: "병렬회로", 2: "직렬회로", 3: "키르히호프 1법칙", 4: "키르히호프 2법칙",
                    5: "중첩의 원리-a",6: "중첩의 원리-b",7: "중첩의 원리-c",8: "교류 전원", 9: "오실로스코프1",
                    10: "반파정류회로", 11: "반파정류회로2", 12: "비반전 증폭기"
        }

    def provide_comprehensive_feedback(self, errors: list, comparison_result: dict) -> str:
        """종합적인 피드백 메시지 생성"""
        feedback_lines = []
        
        # 🔍 1. 기준 회로 정보
        if comparison_result:
            feedback_lines.append(f"📊 기준 회로: {comparison_result.get('reference_topic', 'Unknown')}")
        
        # 🔍 2. 오류 분석 결과
        if errors:
            feedback_lines.append(f"\n❌ 회로 오류 ({len(errors)}개):")
            for i, error in enumerate(errors[:5], 1):  # 최대 5개만 표시
                feedback_lines.append(f"  {i}. {error}")
            if len(errors) > 5:
                feedback_lines.append(f"  ... 및 {len(errors) - 5}개 추가 오류")
        else:
            feedback_lines.append("\n✅ 회로 오류: 없음")
        
        # 🔍 3. 유사도 분석 결과
        if comparison_result:
            similarity = comparison_result.get('similarity', 0)
            level = comparison_result.get('level', 'UNKNOWN')
            
            # 유사도 아이콘 및 메시지
            if level == 'EXCELLENT':
                icon = "🎉"
                msg = "매우 높은 유사도 - 거의 동일한 회로입니다!"
            elif level == 'GOOD':
                icon = "✅"
                msg = "높은 유사도 - 기준 회로와 유사합니다."
            elif level == 'MODERATE':
                icon = "⚠️"
                msg = "중간 유사도 - 일부 차이가 있습니다."
            else:
                icon = "❌"
                msg = "낮은 유사도 - 기준 회로와 많이 다릅니다."
            
            feedback_lines.append(f"\n📈 유사도: {similarity:.3f} ({similarity*100:.1f}%)")
            feedback_lines.append(f"{icon} {msg}")
        else:
            feedback_lines.append("\n⚠️ 유사도 분석: 비교 실패")
        
        # 🔍 4. 개선 제안사항
        feedback_lines.append("\n💡 개선 제안:")
        
        if errors:
            feedback_lines.append("  🔧 오류 개선:")
            for error in errors[:3]:  # 상위 3개 오류만
                if "missing wire" in error.lower():
                    feedback_lines.append("    - 누락된 연결선을 추가하세요.")
                elif "voltage source" in error.lower():
                    feedback_lines.append("    - 전압원 연결 방식을 확인하세요.")
                elif "short circuit" in error.lower():
                    feedback_lines.append("    - 단락 회로를 확인하세요.")
                elif "open circuit" in error.lower():
                    feedback_lines.append("    - 개방된 회로를 확인하세요.")
                else:
                    feedback_lines.append(f"    - {error[:50]}... 문제를 검토하세요.")
        
        if comparison_result:
            similarity = comparison_result.get('similarity', 0)
            if similarity < 0.5:
                feedback_lines.append("  📐 회로 구조:")
                feedback_lines.append("    - 기준 회로의 부품 배치를 참고하세요.")
                feedback_lines.append("    - 연결 방식을 다시 확인하세요.")
            elif similarity < 0.8:
                feedback_lines.append("  🔍 세부 조정:")
                feedback_lines.append("    - 주요 분기점(노드)을 재검토하세요.")
                feedback_lines.append("    - 부품 값을 확인하세요.")
            else:
                feedback_lines.append("  👍 우수한 회로:")
                feedback_lines.append("    - 기준 회로와 매우 유사합니다!")
                feedback_lines.append("    - 세부 파라미터만 확인하세요.")
        
        return "\n".join(feedback_lines)

    def create_detailed_feedback_data(self, errors: list, comparison_result: dict, component_count: int, power_count: int) -> dict:
        """LLM에 전달할 상세한 피드백 데이터 생성"""
        feedback_data = {
            'reference_circuit': comparison_result.get('reference_topic', 'Unknown') if comparison_result else None,
            'similarity_score': comparison_result.get('similarity', 0) if comparison_result else 0,
            'similarity_level': comparison_result.get('level', 'UNKNOWN') if comparison_result else 'UNKNOWN',
            'errors': errors,
            'error_count': len(errors),
            'component_count': component_count,
            'power_source_count': power_count,
            'analysis_summary': self.provide_comprehensive_feedback(errors, comparison_result)
        }
        
        # 추가적인 분석 정보
        if comparison_result:
            if feedback_data['similarity_score'] >= 0.9:
                feedback_data['performance_grade'] = 'A'
                feedback_data['performance_description'] = '우수'
            elif feedback_data['similarity_score'] >= 0.7:
                feedback_data['performance_grade'] = 'B'
                feedback_data['performance_description'] = '양호'
            elif feedback_data['similarity_score'] >= 0.5:
                feedback_data['performance_grade'] = 'C'
                feedback_data['performance_description'] = '보통'
            else:
                feedback_data['performance_grade'] = 'D'
                feedback_data['performance_description'] = '개선 필요'
        else:
            feedback_data['performance_grade'] = 'N/A'
            feedback_data['performance_description'] = '비교 불가'
        
        return feedback_data

    def show_comprehensive_feedback(self, errors: list, comparison_result: dict):
        """종합 피드백을 콘솔과 messagebox로 표시"""
        # 피드백 메시지 생성
        feedback_message = self.provide_comprehensive_feedback(errors, comparison_result)
        
        # 🖥️ 콘솔에 출력
        print("\n" + "="*80)
        print("🔍 종합 회로 분석 결과")
        print("="*80)
        print(feedback_message)
        print("="*80)
        
        # 📋 MessageBox에 표시
        try:
            root = tk.Tk()
            root.withdraw()
            
            # 창 제목 설정
            if comparison_result:
                similarity = comparison_result.get('similarity', 0)
                level = comparison_result.get('level', 'UNKNOWN')
                
                if level == 'EXCELLENT':
                    title = "🎉 회로 분석 결과 - 우수"
                elif level == 'GOOD':
                    title = "✅ 회로 분석 결과 - 양호"
                elif level == 'MODERATE':
                    title = "⚠️ 회로 분석 결과 - 보통"
                else:
                    title = "❌ 회로 분석 결과 - 개선 필요"
            else:
                title = "🔍 회로 분석 결과"
            
            # 메시지 표시 (긴 메시지는 스크롤 가능한 형태로)
            # if len(feedback_message) > 500:
            #     # 긴 메시지의 경우 요약본 표시
            #     summary_lines = []
            #     lines = feedback_message.split('\n')
                
            #     for line in lines:
            #         if any(keyword in line for keyword in ['📊 기준 회로:', '✅ 회로 오류:', '❌ 회로 오류:', '📈 유사도:', '🎉', '✅', '⚠️', '❌']):
            #             summary_lines.append(line)
                
            #     summary_message = '\n'.join(summary_lines[:15])  # 최대 15줄
            #     if len(lines) > 15:
            #         summary_message += "\n\n📝 상세한 분석 결과는 콘솔을 확인하세요."
            #         summary_message += "\n🤖 AI 분석이 곧 시작됩니다."
                
            #     messagebox.showinfo(title, summary_message)
            # else:
            #     messagebox.showinfo(title, feedback_message + "\n\n🤖 AI 분석이 곧 시작됩니다.")
            
            root.destroy()
            
        except Exception as e:
            print(f"⚠️ MessageBox 표시 실패: {e}")
            print("콘솔에서 결과를 확인하세요.")

    def select_reference_circuit(self, selection=None):
        """사용자가 기준 회로를 선택하는 UI"""
        print("\n🎯 기준 회로 선택")
        print("="*50)
        
        # circuits 폴더 확인
        circuits_dir = r"D:/Hyuntak/lab/AR_circuit_tutor/breadboard_project/circuits"
        if not os.path.exists(circuits_dir):
            print(f"❌ '{circuits_dir}' 폴더를 찾을 수 없습니다.")
            return False
        
        # 사용 가능한 회로 파일 검색
        available_circuits = []

        # circuits_dir: 그래프ML 파일들이 있는 폴더 경로
        for fname in os.listdir(circuits_dir):
            # 전각 숫자 등을 아스키 숫자로 변환
            normalized = unicodedata.normalize('NFKC', fname)
            
            for i in range(1, 13):
                #  예) '4.graphml', 'circuit4.graphml', 'topic4.graphml', 'circuit_4.graphml'
                #  그리고 뒤에 '_...' 접미사까지 허용
                pattern = rf'^(?:{i}|circuit{i}|topic{i}|circuit_{i})(?:_.*)?\.graphml$'
                
                if re.match(pattern, normalized):
                    path = os.path.join(circuits_dir, fname)
                    topic = self.topic_map.get(i, f"회로 {i}")
                    available_circuits.append((i, path, topic))
                    break  # 이 파일은 i에 매칭되었으므로, 다음 파일로

        if not available_circuits:
            print("❌ circuits 폴더에서 기준 회로를 찾을 수 없습니다.")
            return False
        
        # 사용자에게 선택 옵션 표시
        print("📋 사용 가능한 기준 회로:")
        for circuit_num, path, topic in available_circuits:
            print(f"  {circuit_num}. {topic}")
        
        # 직접 선택이 주어진 경우
        if selection is not None:
            selected_circuit = next((item for item in available_circuits if item[0] == selection), None)
            if selected_circuit:
                self.reference_circuit_path = selected_circuit[1]
                self.reference_circuit_topic = selected_circuit[2]
                print(f"✅ 선택된 기준 회로: {selection}. {self.reference_circuit_topic}")
                print(f"   파일 경로: {self.reference_circuit_path}")
                return True
            else:
                print(f"❌ 선택된 회로 {selection}를 찾을 수 없습니다.")
                return False
        
        # Tkinter로 사용자 선택 받기
        root = tk.Tk()
        root.withdraw()
        
        try:
            choice = simpledialog.askinteger(
                "기준 회로 선택",
                f"기준 회로를 선택하세요 (1-12):\n\n" + 
                "\n".join([f"{num}. {topic}" for num, _, topic in available_circuits]),
                minvalue=1,
                maxvalue=12
            )
            root.destroy()
            
            if choice is None:
                print("❌ 기준 회로 선택이 취소되었습니다.")
                return False
            
            # 선택된 회로 정보 저장
            selected_circuit = next((item for item in available_circuits if item[0] == choice), None)
            if selected_circuit:
                self.reference_circuit_path = selected_circuit[1]
                self.reference_circuit_topic = selected_circuit[2]
                print(f"✅ 선택된 기준 회로: {choice}. {self.reference_circuit_topic}")
                print(f"   파일 경로: {self.reference_circuit_path}")
                return True
            else:
                print(f"❌ 선택된 회로 {choice}를 찾을 수 없습니다.")
                return False
                
        except Exception as e:
            print(f"❌ 회로 선택 중 오류 발생: {e}")
            root.destroy()
            return False
    
    def compare_with_reference_circuit(self, generated_circuit_path):
        """생성된 회로와 기준 회로를 비교"""
        if not self.reference_circuit_path:
            print("❌ 기준 회로가 선택되지 않았습니다.")
            return None
        
        if not os.path.exists(generated_circuit_path):
            print(f"❌ 생성된 회로 파일을 찾을 수 없습니다: {generated_circuit_path}")
            return None
        
        print(f"\n🔍 회로 유사도 분석")
        print("="*50)
        print(f"기준 회로: {self.reference_circuit_topic}")
        print(f"생성된 회로: {generated_circuit_path}")
        
        try:
            # 그래프 로드
            reference_graph = nx.read_graphml(self.reference_circuit_path)
            generated_graph = nx.read_graphml(generated_circuit_path)
            
            # 회로 비교
            comparer = CircuitComparer(generated_graph, reference_graph, debug=True)
            similarity = comparer.compute_similarity()
            
            # 결과 출력
            print(f"\n📊 유사도 분석 결과:")
            print(f"  전체 유사도: {similarity:.3f} ({similarity*100:.1f}%)")
            
            # 유사도 해석
            if similarity >= 0.9:
                result_msg = "🎉 매우 높은 유사도 - 거의 동일한 회로입니다!"
                result_level = "EXCELLENT"
            elif similarity >= 0.7:
                result_msg = "✅ 높은 유사도 - 기준 회로와 유사합니다."
                result_level = "GOOD"
            elif similarity >= 0.5:
                result_msg = "⚠️ 중간 유사도 - 일부 차이가 있습니다."
                result_level = "MODERATE"
            else:
                result_msg = "❌ 낮은 유사도 - 기준 회로와 많이 다릅니다."
                result_level = "LOW"
            
            print(f"  평가: {result_msg}")
            
            # 시각화 비교 (선택사항)
            try:
                comparison_img_path = generated_circuit_path.replace('.graphml', '_comparison.png')
                comparer.visualize_comparison(save_path=comparison_img_path, show=False)
                print(f"  비교 시각화 저장: {comparison_img_path}")
            except Exception as e:
                print(f"  ⚠️ 시각화 저장 실패: {e}")
            
            return {
                'similarity': similarity,
                'level': result_level,
                'message': result_msg,
                'reference_topic': self.reference_circuit_topic
            }
            
        except Exception as e:
            print(f"❌ 회로 비교 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return None

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

    def generate_final_circuit(self, component_pins, holes, power_sources, warped):
        """최종 회로 생성 - 넷 병합 완전 처리 + 회로 비교 + LLM용 피드백 데이터 생성"""
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
            error_result, detected_errors = self._check_circuit_errors_lenient_with_details(
                components, power_pairs, power_sources
            )
            
            if not error_result:
                print("❌ 사용자가 회로도 생성을 취소했습니다.")
                return False, None  # 피드백 데이터도 None 반환
            
            # 🔟 기준 회로와 비교
            comparison_result = None
            if self.reference_circuit_path:
                print("\n🔍 기준 회로와 유사도 비교 중...")
                generated_graphml = "circuit.graphml"
                comparison_result = self.compare_with_reference_circuit(generated_graphml)
                
                if comparison_result:
                    print(f"✅ 회로 비교 완료: {comparison_result['level']} ({comparison_result['similarity']:.3f})")
                else:
                    print("⚠️ 회로 비교 실패")
            
            # 1️⃣1️⃣ 종합 피드백 표시
            print("\n🎯 종합 피드백 생성 중...")
            self.show_comprehensive_feedback(detected_errors, comparison_result)
            
            # 1️⃣2️⃣ LLM용 상세 피드백 데이터 생성 ⭐
            component_count = len([c for c in component_pins if c['class'] != 'Line_area'])
            power_count = len(power_sources)
            
            feedback_data = self.create_detailed_feedback_data(
                detected_errors, comparison_result, component_count, power_count
            )
            
            print("✅ 넷 병합이 완전히 적용된 회로도 생성 완료!")
            print("📁 생성된 파일:")
            print("  - circuit.jpg (병합 적용 회로도)")
            print("  - circuit.spice (병합 적용 SPICE 넷리스트)")
            print("  - circuit.graphml (회로 그래프)")
            
            return True, feedback_data  # 성공 여부와 피드백 데이터 반환
            
        except Exception as e:
            print(f"❌ 회로 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False, None

    def _check_circuit_errors_lenient_with_details(self, components, power_pairs, power_sources):
        """관대한 오류 검사 - 다중 전원 허용 + 오류 목록 반환"""
        try:
            components_for_check = components.copy()
            detected_errors = []
            
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
            
            detected_errors = filtered_errors  # 필터링된 오류들을 저장
            
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
                
                #result = messagebox.askyesno("회로 오류 발견", error_msg)
                #root.destroy()
                
                #return result, detected_errors
                root.destroy()
                # 강제 생성 진행 (True 반환)
                return True, detected_errors
            else:
                print("✅ 심각한 회로 오류가 발견되지 않았습니다!")
                if len(power_sources) > 1:
                    print(f"📋 {len(power_sources)}개의 전원이 하나의 회로에 표시됩니다.")
                return True, detected_errors
                
        except Exception as e:
            print(f"⚠️ 오류 검사 중 문제가 발생했습니다: {e}")
            print("회로도 생성을 계속합니다...")
            return True, []