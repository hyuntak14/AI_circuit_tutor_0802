#!/usr/bin/env python3
# llm_feedback_manager.py (피드백 통합 버전)
MAX_CONTEXT_LENGTH = 8000
import os
from gemini_test_rag import RAGSystem, create_rag_prompt, initialize_gemini

class LLMFeedbackManager:
    """
    LLM 피드백 매니저: SPICE 넷리스트, 컴포넌트 핀 정보, 회로 분석 결과를 기반으로 AI 분석 및 대화형 채팅을 제공합니다.
    """
    def __init__(self, practice_circuit_topic: str = ""):
        """
        practice_circuit_topic: 실습 회로 주제 (기본값은 빈 문자열).
        """
        # RAG 시스템 초기화
        try:
            self.rag_system = RAGSystem()
        except Exception as e:
            raise RuntimeError(f"RAG 시스템 초기화 실패: {e}")

        # Gemini 모델 초기화
        try:
            self.model, self.generation_config = initialize_gemini()
        except Exception as e:
            raise RuntimeError(f"Gemini 모델 초기화 실패: {e}")

        self.practice_circuit_topic = practice_circuit_topic
        self.first_turn = True

    def provide_initial_feedback_with_analysis(self, spice_file: str, component_pins, feedback_data: dict, user_query: str = "") -> bool:
        """
        초기 AI 피드백을 제공합니다 (회로 분석 결과 포함).
        - spice_file: SPICE 넷리스트 파일 경로
        - component_pins: 전달받은 컴포넌트 핀 정보
        - feedback_data: 회로 분석 결과 데이터 (오류, 유사도 등)
        - user_query: AI에게 던질 첫 질문 텍스트 (선택사항)
        반환값: 피드백 제공 성공 여부
        """
        # SPICE 파일 읽기
        try:
            with open(spice_file, 'r', encoding='utf-8') as f:
                spice_text = f.read()
        except Exception as e:
            print(f"❌ SPICE 파일 '{spice_file}'을 읽는 데 실패: {e}")
            return False

        # component_pins 형식 확인 및 변환
        items = []
        if isinstance(component_pins, dict):
            items = list(component_pins.items())
        elif isinstance(component_pins, list):
            for entry in component_pins:
                # 튜플 또는 리스트 형태 with at least 2 elements
                if isinstance(entry, (tuple, list)) and len(entry) >= 2:
                    comp, pins = entry[0], entry[1]
                # dict 형태: 'component' 또는 'class' 키 사용
                elif isinstance(entry, dict) and 'pins' in entry:
                    if 'component' in entry:
                        comp = entry['component']
                    elif 'class' in entry:
                        comp = entry['class']
                    else:
                        print("❌ component_pins 리스트 dict 항목에 'component' 또는 'class' 키가 없습니다.")
                        return False
                    pins = entry['pins']
                # object 형태: 속성 확인
                elif hasattr(entry, 'component') and hasattr(entry, 'pins'):
                    comp, pins = entry.component, entry.pins
                else:
                    print(f"❌ 처리할 수 없는 entry 형식: {entry!r} (type: {type(entry)})")
                    return False
                items.append((comp, pins))
        else:
            print("❌ 지원되지 않는 component_pins 형식입니다.")
            return False

        # component_pins를 문자열로 변환
        pins_info = []
        for comp, pins in items:
            pins_info.append(f"{comp}: {pins}")
        pins_context = "\n".join(pins_info)

        # 회로 분석 결과를 컨텍스트에 추가
        analysis_context = self._format_analysis_context(feedback_data)

        # 전체 컨텍스트 결합
        full_context = f"""=== SPICE 넷리스트 ===
{spice_text}

=== 컴포넌트 핀 정보 ===
{pins_context}

=== 회로 분석 결과 ===
{analysis_context}"""

        # 기본 질문 설정
        if not user_query:
            user_query = "제가 구성한 브레드보드 회로를 종합적으로 분석해주세요. 특히 오류와 개선점을 중심으로 설명해주세요."

                # Construct full_context, ensuring it does not exceed MAX_CONTEXT_LENGTH
        # analysis_context가 가장 중요하다고 가정하고, spice_text와 pins_context를 먼저 자릅니다.
        # 또는 전체 full_context를 자를 수도 있습니다. 여기서는 전체를 자르는 예시를 보여줍니다.
        full_context_parts = [
            f"SPICE 넷리스트:\n{spice_text}",
            f"컴포넌트 핀 정보:\n{pins_context}",
            f"회로 분석 결과:\n{analysis_context}"
        ]
        full_context = "\n\n".join(full_context_parts)

        # Truncate full_context if it exceeds the maximum allowed length
        if len(full_context) > MAX_CONTEXT_LENGTH:
            # 텍스트가 잘렸음을 나타내는 메시지 추가
            truncated_message = "\n\n[CONTEXT TRUNCATED DUE TO LENGTH]"
            # 중요한 정보(예: 분석 결과)가 최대한 포함되도록 끝부분에서 자르기
            # 또는 필요한 경우 앞부분에서 자르기
            full_context = full_context[:MAX_CONTEXT_LENGTH - len(truncated_message)] + truncated_message


        # RAG 프롬프트 생성
        prompt = create_rag_prompt(user_query, full_context, self.first_turn, self.practice_circuit_topic)

        print("🔍 DEBUG: full prompt:\n", prompt[:1000] + "..." if len(prompt) > 1000 else prompt)

        self.first_turn = False

        # AI 응답 생성
        try:
            print("\n🤖 AI 회로 분석 결과:")
            print("="*60)
            
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            print(response.text)
            print("="*60)
            return True
        except Exception as e:
            print(f"❌ AI 피드백 생성 오류: {e}")
            return False

    def _format_analysis_context(self, feedback_data: dict) -> str:
        """회로 분석 결과를 LLM이 이해하기 쉬운 형태로 포맷팅"""
        if not feedback_data:
            return "회로 분석 결과가 없습니다."
        
        context_lines = []
        
        # 기본 정보
        context_lines.append(f"기준 회로: {feedback_data.get('reference_circuit', 'N/A')}")
        context_lines.append(f"컴포넌트 수: {feedback_data.get('component_count', 0)}개")
        context_lines.append(f"전원 수: {feedback_data.get('power_source_count', 0)}개")
        
        # 유사도 분석
        similarity = feedback_data.get('similarity_score', 0)
        level = feedback_data.get('similarity_level', 'UNKNOWN')
        grade = feedback_data.get('performance_grade', 'N/A')
        
        context_lines.append(f"유사도 점수: {similarity:.3f} ({similarity*100:.1f}%)")
        context_lines.append(f"유사도 등급: {level}")
        context_lines.append(f"성능 평가: {grade}등급 ({feedback_data.get('performance_description', 'N/A')})")
        
        # 오류 분석
        errors = feedback_data.get('errors', [])
        error_count = feedback_data.get('error_count', 0)
        
        if error_count > 0:
            context_lines.append(f"\n감지된 오류 ({error_count}개):")
            for i, error in enumerate(errors[:5], 1):  # 최대 5개만
                context_lines.append(f"  {i}. {error}")
            if error_count > 5:
                context_lines.append(f"  ... 및 {error_count - 5}개 추가 오류")
        else:
            context_lines.append("\n감지된 오류: 없음")
        
        # 종합 분석 요약
        summary = feedback_data.get('analysis_summary', '')
        if summary:
            context_lines.append(f"\n종합 분석 요약:\n{summary}")
        
        return "\n".join(context_lines)

    def provide_initial_feedback(self, spice_file: str, component_pins, user_query: str) -> bool:
        """
        기존 초기 AI 피드백 메서드 (하위 호환성 유지).
        """
        # 빈 피드백 데이터로 새 메서드 호출
        empty_feedback = {}
        return self.provide_initial_feedback_with_analysis(spice_file, component_pins, empty_feedback, user_query)

    def start_interactive_chat(self):
        """
        초기 피드백 이후 추가 대화를 위한 인터랙티브 채팅 세션을 실행합니다.
        종료하려면 'exit' 또는 'quit'을 입력하세요.
        """
        print("\n💬 추가 질문이 있으시면 언제든 물어보세요!")
        print("   (종료하려면 'exit', 'quit', 'q' 또는 '종료'를 입력하세요)")
        
        while True:
            user_input = input("\n💭 질문: ").strip()
            if user_input.lower() in ['exit', 'quit', '종료', 'q']:
                print("👍 AI 채팅을 종료합니다.")
                break
            if not user_input:
                print("❌ 메시지를 입력해주세요.")
                continue

            # RAG 검색
            print("🔍 관련 문서 검색 중...", end=" ")
            results = self.rag_system.search_similar_documents(user_input, top_k=3)
            print(f"({len(results)}개 문서 발견)")

            # 컨텍스트 생성
            context = self.rag_system.create_context(results)

            # 프롬프트 생성
            prompt = create_rag_prompt(user_input, context, self.first_turn, self.practice_circuit_topic)
            self.first_turn = False

            # AI 응답 생성
            print("🤖 AI: ", end="")
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                print(response.text)
            except Exception as e:
                print(f"❌ 응답 생성 오류: {e}")

    def provide_comprehensive_analysis(self, spice_file: str, component_pins, feedback_data: dict) -> bool:
        """
        종합적인 회로 분석을 제공하는 새로운 메서드
        """
        print("\n" + "🤖" + "="*59)
        print("🧠 AI 기반 종합 회로 분석")
        print("="*60)
        
        # 분석 성공 여부 확인
        if not feedback_data:
            print("⚠️ 회로 분석 데이터가 없어 AI 분석을 건너뜁니다.")
            return False
        
        # 유사도와 오류 상황에 따른 맞춤형 질문 생성
        similarity = feedback_data.get('similarity_score', 0)
        errors = feedback_data.get('errors', [])
        reference = feedback_data.get('reference_circuit', 'Unknown')
        
        if len(errors) > 0 and similarity < 0.5:
            analysis_query = f"이 회로에는 {len(errors)}개의 오류가 있고 기준 회로({reference})와의 유사도가 {similarity:.1%}로 낮습니다. 주요 문제점과 해결 방안을 상세히 분석해주세요."
        elif len(errors) > 0:
            analysis_query = f"이 회로에는 {len(errors)}개의 오류가 감지되었습니다. 각 오류의 원인과 수정 방법을 구체적으로 설명해주세요."
        elif similarity < 0.7:
            analysis_query = f"기준 회로({reference})와의 유사도가 {similarity:.1%}입니다. 어떤 부분을 개선하면 더 정확한 회로가 될지 분석해주세요."
        else:
            analysis_query = f"이 회로는 기준 회로({reference})와 {similarity:.1%} 유사도를 보입니다. 회로의 동작 원리와 특성을 분석해주세요."

        # AI 분석 수행
        success = self.provide_initial_feedback_with_analysis(
            spice_file, component_pins, feedback_data, analysis_query
        )
        
        if success:
            print("\n✅ AI 종합 분석이 완료되었습니다!")
        else:
            print("\n❌ AI 분석 중 오류가 발생했습니다.")
        
        return success