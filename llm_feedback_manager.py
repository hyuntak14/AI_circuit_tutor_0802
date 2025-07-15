#!/usr/bin/env python3
# llm_feedback_manager.py

import os
from gemini_test_rag import RAGSystem, create_rag_prompt, initialize_gemini

class LLMFeedbackManager:
    """
    LLM 피드백 매니저: SPICE 넷리스트와 컴포넌트 핀 정보를 기반으로 AI 분석 및 대화형 채팅을 제공합니다.
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

    def provide_initial_feedback(self, spice_file: str, component_pins, user_query: str) -> bool:
        """
        초기 AI 피드백을 제공합니다.
        - spice_file: SPICE 넷리스트 파일 경로
        - component_pins: 전달받은 컴포넌트 핀 정보 (dict, list of tuples, list of dicts, or list of objects)
        - user_query: AI에게 던질 첫 질문 텍스트
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

        # 디버깅 출력: 변환된 items
        print(f"🔧 component_pins 변환 결과: {items}")

        # 컨텍스트 결합
        context = spice_text + "\n\n" + pins_context

        # RAG 프롬프트 생성
        prompt = create_rag_prompt(user_query, context, self.first_turn, self.practice_circuit_topic)
        self.first_turn = False

        # AI 응답 생성
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            print(response.text)
            return True
        except Exception as e:
            print(f"❌ AI 피드백 생성 오류: {e}")
            return False

    def start_interactive_chat(self):
        """
        초기 피드백 이후 추가 대화를 위한 인터랙티브 채팅 세션을 실행합니다.
        종료하려면 'exit' 또는 'quit'을 입력하세요.
        """
        while True:
            user_input = input("\n💬 질문 입력 (종료: 'exit'/'quit'): ").strip()
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
