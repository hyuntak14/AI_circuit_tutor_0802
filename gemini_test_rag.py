#!/usr/bin/env python3
# rag_gemini_chat.py

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

class RAGSystem:
    def __init__(self, index_path="faiss_index.index", metadata_path="embedding_metadata.parquet"):
        """RAG 시스템을 초기화합니다."""
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.metadata = None
        self.embedding_model = None
        
        # 임베딩 모델 로드 (FAISS 인덱스 생성 시 사용한 것과 동일해야 함)
        print("임베딩 모델 로딩 중...")
        try:
            self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
            # GPU 사용 여부 확인 (선택 사항)
            # import torch
            # if torch.cuda.is_available():
            #     print(f"✅ SentenceTransformer가 GPU를 사용합니다: {torch.cuda.get_device_name(0)}")
            # else:
            #     print("⚠️ SentenceTransformer가 CPU를 사용합니다.")
        except Exception as e:
            print(f"❌ 임베딩 모델 로딩 실패: {e}")
            print("pip install sentence-transformers 로 설치되었는지 확인해주세요.")
            raise

        # FAISS 인덱스와 메타데이터 로드
        self.load_database()
    
    def load_database(self):
        """FAISS 인덱스와 메타데이터를 로드합니다."""
        try:
            # FAISS 인덱스 로드
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                print(f"✅ FAISS 인덱스 로드 완료: {self.index.ntotal}개 벡터")
            else:
                raise FileNotFoundError(f"FAISS 인덱스 파일을 찾을 수 없습니다: {self.index_path}")
            
            # 메타데이터 로드
            if os.path.exists(self.metadata_path):
                self.metadata = pd.read_parquet(self.metadata_path)
                print(f"✅ 메타데이터 로드 완료: {len(self.metadata)}개 문서")
            else:
                raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {self.metadata_path}")
            
            # 데이터 일관성 확인
            if self.index.ntotal != len(self.metadata):
                print(f"⚠️ 경고: 인덱스 벡터 수({self.index.ntotal})와 메타데이터 행 수({len(self.metadata)})가 다릅니다.")
                
        except Exception as e:
            print(f"❌ 데이터베이스 로드 실패: {e}")
            print("FAISS 인덱스 파일과 메타데이터 파일이 올바른 경로에 있는지 확인해주세요.")
            raise
    
    def search_similar_documents(self, query, top_k=5):
        """쿼리와 유사한 문서들을 검색합니다."""
        try:
            # 쿼리를 임베딩으로 변환
            # 1) GPU tensor로 임베딩 생성 (convert_to_tensor=True 시)
            query_embedding = self.embedding_model.encode(
                [query],
                convert_to_tensor=True,
                normalize_embeddings=True
            )
            # 2) CPU로 가져와서 numpy 변환
            query_embedding = query_embedding.cpu().detach().numpy().astype('float32')
            
            # FAISS 검색
            distances, indices = self.index.search(query_embedding, top_k)
            
            # 결과 정리
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.metadata):  # 유효한 인덱스인지 확인
                    doc_info = self.metadata.iloc[idx].to_dict()
                    results.append({
                        'rank': i + 1,
                        'similarity_score': float(1 / (1 + distance)),  # 거리를 유사도로 변환 (간단한 방법)
                        'distance': float(distance),
                        'content': doc_info.get('chunk_text', ''),
                        'metadata': doc_info # 원본 메타데이터 전체를 포함
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ 문서 검색 실패: {e}")
            return []
    
    def create_context(self, search_results, max_length=3000): # max_length 약간 증가
        """검색 결과를 컨텍스트로 변환합니다."""
        if not search_results:
            return ""
        
        context_parts = []
        current_length = 0
        
        for result in search_results:
            content = result['content']
            
            # 길이 제한 확인
            if current_length + len(content) > max_length:
                # 남은 공간만큼만 추가
                remaining_space = max_length - current_length
                if remaining_space > 200:  # 최소 200자는 있어야 의미가 있음
                    content = content[:remaining_space] + "..." # 잘린 부분을 표시
                    context_parts.append(f"[문서 {result['rank']}] {content}")
                break # 더 이상 추가하지 않음
            
            context_parts.append(f"[문서 {result['rank']}] {content}")
            current_length += len(content)
        
        return "\n\n".join(context_parts)

def initialize_gemini():
    """Gemini API를 초기화하고 사용 가능한 모델을 찾습니다."""
    # 중요: 실제 API 키를 여기에 입력하세요.
    # 안전을 위해 환경 변수나 설정 파일에서 로드하는 것을 권장합니다.
    api_key = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY_HERE") # 환경 변수 사용 권장

    if api_key == "YOUR_GEMINI_API_KEY_HERE" or not api_key:
        print("경고: GOOGLE_API_KEY 환경 변수가 설정되지 않았습니다. 코드 내 하드코딩된 키를 사용합니다.")
        print("또는 'export GOOGLE_API_KEY='your_api_key''로 설정해주세요.")
        # 안전상의 이유로 여기에 직접 API 키를 넣는 것은 권장하지 않습니다.
        # 테스트를 위해 아래 라인의 주석을 해제하고 실제 키로 대체하세요.
        api_key = "AIzaSyCxgQUQFLhTi-Y6nzRpdpgpgHO9xVJ-CAo" 
    
    genai.configure(api_key=api_key)

    generation_config = genai.GenerationConfig(
        temperature=0.4, # 온도값을 조금 더 낮춰서 안정적인 응답 유도 (기존 0.7에서 변경)
        max_output_tokens=1500 # 출력 토큰 수 증가 (필요시 조절)
    )

    model_names = [
        "models/gemini-2.5-flash", # 더 빠른 응답을 위해 flash 모델 우선 고려
        "models/gemini-2.5-pro",   # 더 강력한 성능을 위해 pro 모델 고려
    ]

    for model_name in model_names:
        try:
            print(f"모델 테스트 중: {model_name}")
            model = genai.GenerativeModel(model_name)
            
            # 간단한 테스트 호출
            test_response = model.generate_content(
                "안녕하세요! 당신은 누구인가요? 간단히 답해주세요.",
                generation_config=generation_config
            )
            if test_response.text:
                print(f"✅ {model_name} 사용 가능!")
                return model, generation_config
            else:
                print(f"❌ {model_name} 응답 없음 (text 필터링됨).")
        except Exception as e:
            print(f"❌ {model_name} 사용 불가: {e}")
            continue
    
    raise Exception("사용 가능한 Gemini 모델을 찾을 수 없습니다. API 키와 네트워크 연결을 확인해주세요.")

def create_rag_prompt(user_query, context, is_first_turn, practice_circuit_topic=""):
    """
    RAG용 프롬프트를 생성합니다.
    Args:
        user_query (str): 사용자의 현재 질문.
        context (str): 검색된 관련 문서 내용.
        is_first_turn (bool): 현재 턴이 첫 번째 턴인지 여부.
        practice_circuit_topic (str): 실습 회로의 주제 (첫 턴에만 해당).
    """
    if is_first_turn:
        # 첫 번째 턴: 넷리스트 분석 및 실습 주제 일치 여부 확인에 집중
        if context.strip():
            prompt = f"""당신은 전자 회로 설계 및 분석을 돕는 전문 AI 어시스턴트입니다.
사용자가 제공한 넷리스트와 실습 보고서 관련 참고 자료를 기반으로 질문에 답변해야 합니다.
특히, 주어진 넷리스트에 잠재적인 오류가 있는지, 그리고 실습 주제인 '{practice_circuit_topic}' 회로와 일치하는지 면밀히 분석해주세요.
**분석 결과 오류나 불일치(차이점)가 발견되면, 다음 형식에 따라 구체적이고 기술적인 용어를 사용하여 설명해주세요.**
답변은 항상 객관적이고 사실에 기반하며, 어떤 종류의 유해하거나 부적절한 내용은 포함하지 않습니다.

--- 참고 자료 (넷리스트 및 관련 회로 정보) ---
{context}

--- 사용자의 질문 ---
{user_query}

--- 답변 형식 (첫 번째 답변) ---
1.  **개요:** 넷리스트 분석 및 실습 회로 일치 여부에 대한 핵심 요약. (예: "넷리스트는 정상적이며 실습 회로와 일치합니다." 또는 "넷리스트에 오류가 발견되었으며 실습 회로와 일부 다릅니다.")
2.  **분석 결과:**
    * **넷리스트 오류 여부:** 넷리스트에서 발견된 구체적인 문법적/논리적 오류 (예: "Component X의 연결이 잘못됨", "Net Y의 정의 누락" 등). 오류가 없다면 "넷리스트에서 특정 오류는 발견되지 않았습니다."로 기재.
    * **실습 회로 일치 여부:** '{practice_circuit_topic}' 회로의 예상되는 구조/특성과 비교하여, 넷리스트가 어떤 부분이 일치하고 어떤 부분이 다른지 명확하게 설명. (예: "저항 R1의 값이 실습 가이드와 다릅니다.", "Op-Amp의 피드백 루프 구성이 변경되었습니다." 등). 완전히 일치한다면 "넷리스트는 실습 주제인 '{practice_circuit_topic}' 회로와 완전히 일치합니다."로 기재.
3.  **다음 단계 제안:** 사용자가 넷리스트를 수정하거나, 회로를 개선하거나, 추가 분석을 진행할 수 있도록 도움이 될 만한 구체적인 조치 또는 권장 사항.

--- 답변 ---
"""
        else:
            prompt = f"""당신은 전자 회로 설계 및 분석 전문 AI 어시스턴트입니다.
사용자님, 현재 넷리스트 정보가 충분히 제공되지 않아 요청하신 분석을 진행하기 어렵습니다.

--- 사용자의 질문 ---
{user_query}

--- 답변 ---
넷리스트 또는 관련 실습 자료가 부족하여 요청하신 분석을 수행할 수 없습니다. 넷리스트 내용을 제공해주시거나, 어떤 종류의 정보를 분석하고 싶으신지 더 자세히 알려주시면 도움을 드릴 수 있습니다.
답변은 항상 객관적이고 사실에 기반하며, 어떤 종류의 유해하거나 부적절한 내용은 포함하지 않습니다.
"""
    else:
        # 이후 턴: 일반적인 Q&A (RAG 활용)
        if context.strip():
            prompt = f"""당신은 전자 회로 설계 및 분석을 돕는 전문 AI 어시스턴트입니다.
제공된 참고 문서를 바탕으로 사용자의 질문에 정확하고 상세하게 답변해주세요.
**답변은 항상 기술적이고 객관적인 사실에 기반하며, 어떤 종류의 유해하거나 부적절한 내용을 포함하지 않습니다.**
만약 참고 문서에 질문에 대한 직접적인 정보가 없다면, 일반적인 전자 회로 지식을 활용하여 답변을 구성해주세요.

--- 참고 문서 ---
{context}

--- 사용자의 질문 ---
{user_query}

--- 답변 형식 (이후 답변) ---
* **요약:** 질문에 대한 핵심 내용을 간략하게 요약합니다.
* **상세 설명:** 질문에 대한 자세하고 포괄적인 설명입니다. 필요시 예시나 관련 개념을 포함합니다.
* **결론 및 추가 정보:** 답변의 마무리 또는 추가적으로 도움이 될 만한 관련 정보나 제안입니다.

--- 답변 ---
"""
        else:
            prompt = f"""당신은 전자 회로 설계 및 분석을 돕는 전문 AI 어시스턴트입니다.
현재 질문과 관련된 참고 문서를 찾을 수 없습니다. 일반적인 전자 회로 지식에 기반하여 질문에 답변하겠습니다.
**답변은 항상 기술적이고 객관적인 사실에 기반하며, 어떤 종류의 유해하거나 부적절한 내용을 포함하지 않습니다.**

--- 사용자의 질문 ---
{user_query}

--- 답변 ---
"""

    return prompt

def chat_with_rag():
    """RAG 시스템과 Gemini를 결합한 채팅 함수 (첫 응답/이후 응답 형식 분기 포함)"""
    # 1) RAG 시스템 초기화
    print("RAG 시스템 초기화 중...")
    try:
        rag_system = RAGSystem()
    except Exception as e:
        print(f"RAG 시스템 초기화 실패로 종료합니다: {e}")
        return

    # 2) Gemini 모델 초기화
    print("Gemini 모델 초기화 중...")
    try:
        model, generation_config = initialize_gemini()
    except Exception as e:
        print(f"Gemini 모델 초기화 실패로 종료합니다: {e}")
        return
    
    # === 사용자 정의 설정 ===
    # 실습 회로의 주제를 여기에 명시적으로 정의합니다.
    # 실제 사용 시에는 이 값을 동적으로 입력받거나 다른 시스템에서 가져와야 할 수 있습니다.
    practice_circuit_topic = "Op-Amp Inverting Amplifier (반전 증폭기)" 
    # ======================

    first_turn = True
    
    print("\n" + "="*60)
    print("🤖 RAG + Gemini 채팅 시작! (종료: quit/exit/종료)")
    print("============================================================")
    print(f"💡 현재 실습 주제: {practice_circuit_topic}")
    print("============================================================")
    
    while True:
        user_input = input("\n당신: ").strip()
        if user_input.lower() in ['quit', 'exit', '종료', 'q']:
            print("👋 채팅을 종료합니다. 안녕히 가세요!")
            break
        if not user_input:
            print("❌ 메시지를 입력해주세요.")
            continue
        
        # 3) 관련 문서 검색
        print("🔍 관련 문서 검색 중...", end=" ", flush=True)
        search_results = rag_system.search_similar_documents(user_input, top_k=3)
        print(f"({len(search_results)}개 문서 발견)")
        
        # 4) 컨텍스트 생성
        context = rag_system.create_context(search_results)
        
        # 5) RAG 프롬프트 생성 (modified call)
        rag_prompt = create_rag_prompt(user_input, context, first_turn, practice_circuit_topic)

        # 첫 턴 이후에는 first_turn을 False로 설정
        if first_turn:
            first_turn = False

        
        # 6) Gemini로 답변 생성
        print("🤖 Gemini: ", end="", flush=True)
        try:
            response = model.generate_content(
                rag_prompt,
                generation_config=generation_config
            )
            print(response.text)
        except genai.types.BlockedPromptException as e:
            # BlockedPromptException은 안전 필터에 의한 차단일 가능성이 높음
            print(f"❌ 생성 오류: 프롬프트가 안전 정책에 의해 차단되었습니다. (finish_reason: {e.response.prompt_feedback.block_reason})")
            print("이전 질문, 검색된 문서(컨텍스트), 또는 현재 질문에 부적절하거나 유해하다고 판단될 수 있는 내용이 있는지 확인해주세요.")
            print(f"자세한 응답: {e.response}") # 더 자세한 정보 출력
        except Exception as e:
            print(f"❌ 생성 오류: {e}")
            # 일반적인 오류 메시지 (e.g., 네트워크, API 키 문제 등)
            if "finish_reason" in str(e) and "2" in str(e):
                print("이는 주로 안전 필터(Safety Filter)에 의해 응답이 차단될 때 발생합니다.")
                print("프롬프트에 포함된 내용(질문, 컨텍스트)에 문제가 없는지 다시 확인해보세요.")
            print(f"발생한 오류 유형: {type(e)}")

        # 7) 참고 문서 정보 표시 (옵션)
        if search_results:
            print(f"\n📚 참고 문서: {len(search_results)}개 (최고 유사도 {search_results[0]['similarity_score']:.3f})")
            # 디버깅을 위해 참고 문서의 content도 출력해볼 수 있습니다.
            # for res in search_results:
            #     print(f"  - Rank {res['rank']} (Score: {res['similarity_score']:.2f}): {res['content'][:100]}...")

def main():
    # 필요한 라이브러리 확인
    required_packages = ['faiss-cpu', 'sentence-transformers', 'pandas', 'numpy', 'google-generativeai']
    missing_packages = []
    
    try:
        import faiss
        import sentence_transformers
        import pandas
        import numpy
        import google.generativeai
    except ImportError as e:
        missing_module = str(e).split("'")[1]
        if missing_module == 'faiss':
            missing_packages.append('faiss-cpu')
        elif missing_module == 'google': # google.generativeai
            missing_packages.append('google-generativeai')
        else:
            missing_packages.append(missing_module)
    
    if missing_packages:
        print("❌ 다음 패키지들을 설치해주세요:")
        for pkg in missing_packages:
            print(f"   pip install {pkg}")
        return
    
    chat_with_rag()

if __name__ == "__main__":
    main()