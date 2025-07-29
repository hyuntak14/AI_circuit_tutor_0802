# llm_feedback_manager.py 파일의 코드를 아래 내용으로 수정/대체하세요.

import os
# gemini_test_rag 모듈에서 필요한 함수들을 직접 임포트합니다.
from gemini_test_rag import RAGSystem, create_rag_prompt as rag_create_prompt, initialize_gemini
from llm_ui import LLMChatWindow # 새로 만든 UI 클래스 import
import tkinter as tk

MAX_CONTEXT_LENGTH = 7500 # 컨텍스트 길이 살짝 줄임

class LLMFeedbackManager:
    """
    LLM 피드백 매니저: SPICE 넷리스트, 컴포넌트 핀 정보, 회로 분석 결과를 기반으로 AI 분석 및 대화형 채팅을 제공합니다.
    """
    def __init__(self, practice_circuit_topic: str = ""):
        try:
            self.rag_system = RAGSystem()
        except Exception as e:
            raise RuntimeError(f"RAG 시스템 초기화 실패: {e}")

        try:
            self.model, self.generation_config = initialize_gemini()
        except Exception as e:
            raise RuntimeError(f"Gemini 모델 초기화 실패: {e}")

        self.practice_circuit_topic = practice_circuit_topic
        # create_rag_prompt 함수를 클래스 메서드로 참조
        self.create_rag_prompt = rag_create_prompt

    def get_initial_analysis_text(self, spice_file: str, component_pins, feedback_data: dict, user_query: str = "") -> str:
        """초기 AI 분석 결과를 텍스트로 반환합니다."""
        try:
            with open(spice_file, 'r', encoding='utf-8') as f:
                spice_text = f.read()
        except Exception as e:
            return f"❌ SPICE 파일 '{spice_file}'을 읽는 데 실패: {e}"

        # ... (provide_initial_feedback_with_analysis의 컨텍스트 생성 로직과 동일) ...
        # component_pins를 문자열로 변환하는 로직 (기존 코드 유지)
        items = []
        if isinstance(component_pins, dict):
            items = list(component_pins.items())
        elif isinstance(component_pins, list):
            for entry in component_pins:
                if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                    comp, pins = entry[0], entry[1]
                elif isinstance(entry, dict) and 'pins' in entry:
                    comp = entry.get('component', entry.get('class', 'Unknown'))
                    pins = entry['pins']
                elif hasattr(entry, 'component') and hasattr(entry, 'pins'):
                    comp, pins = entry.component, entry.pins
                else:
                    print(f"❌ 처리할 수 없는 entry 형식: {entry!r} (type: {type(entry)})")
                    return "컴포넌트 핀 정보 처리 중 오류가 발생했습니다."
                items.append((comp, pins))
        else:
            return "지원되지 않는 component_pins 형식입니다."

        pins_info = [f"{comp}: {pins}" for comp, pins in items]
        pins_context = "\n".join(pins_info)

        analysis_context = self._format_analysis_context(feedback_data)

        full_context = f"""=== SPICE 넷리스트 ===
{spice_text}

=== 컴포넌트 핀 정보 ===
{pins_context}

=== 회로 분석 결과 ===
{analysis_context}"""

        if len(full_context) > MAX_CONTEXT_LENGTH:
            truncated_message = "\n\n[...내용이 너무 길어 일부가 생략되었습니다...]"
            full_context = full_context[:MAX_CONTEXT_LENGTH - len(truncated_message)] + truncated_message

        if not user_query:
            user_query = "제가 구성한 브레드보드 회로를 종합적으로 분석해주세요. 특히 오류와 개선점을 중심으로 설명해주세요."

        # RAG 프롬프트 생성 (첫 턴)
        prompt = self.create_rag_prompt(user_query, full_context, True, self.practice_circuit_topic)

        # AI 응답 생성
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            return response.text
        except Exception as e:
            return f"❌ AI 피드백 생성 오류: {e}"

    def start_chat_ui(self, parent_tk_root, initial_analysis: str):
        """대화형 UI 창을 시작합니다."""
        LLMChatWindow(parent_tk_root, self, initial_analysis)

    # 기존 _format_analysis_context 메서드는 그대로 유지
    def _format_analysis_context(self, feedback_data: dict) -> str:
        """회로 분석 결과를 LLM이 이해하기 쉬운 형태로 포맷팅"""
        if not feedback_data:
            return "회로 분석 결과가 없습니다."
        
        context_lines = []
        
        context_lines.append(f"기준 회로: {feedback_data.get('reference_circuit', 'N/A')}")
        context_lines.append(f"유사도 점수: {feedback_data.get('similarity_score', 0):.3f} ({feedback_data.get('similarity_score', 0)*100:.1f}%)")
        context_lines.append(f"성능 평가: {feedback_data.get('performance_grade', 'N/A')}등급")
        
        errors = feedback_data.get('errors', [])
        error_count = feedback_data.get('error_count', 0)
        
        if error_count > 0:
            context_lines.append(f"\n감지된 오류 ({error_count}개):")
            for i, error in enumerate(errors[:5], 1):
                context_lines.append(f"  {i}. {error}")
            if error_count > 5:
                context_lines.append(f"  ... 및 {error_count - 5}개 추가 오류")
        else:
            context_lines.append("\n감지된 오류: 없음")
        
        summary = feedback_data.get('analysis_summary', '')
        if summary:
            context_lines.append(f"\n종합 분석 요약:\n{summary}")
        
        return "\n".join(context_lines)

    # 이 클래스에 있던 provide_comprehensive_analysis와 같은 래퍼 함수들은 
    # simple_main.py로 로직을 옮기거나 단순화합니다.
    # 여기서는 핵심 기능인 get_initial_analysis_text와 start_chat_ui에 집중합니다.