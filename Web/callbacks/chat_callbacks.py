# callbacks/chat_callbacks.py - RAG 시스템 통합 채팅 콜백
import dash
from dash import Input, Output, State, html
from datetime import datetime
import sys
import os

# 상위 디렉토리의 모듈 임포트를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from state_manager import get_current_session, get_session_data
from components.chat_interface import create_chat_interface

# RAG 시스템 초기화 (전역 변수로 한 번만 초기화)
rag_system = None
gemini_model = None
generation_config = None

def initialize_rag_system():
    """RAG 시스템을 초기화합니다."""
    global rag_system, gemini_model, generation_config
    
    if rag_system is None:
        try:
            from gemini_test_rag import RAGSystem, initialize_gemini, create_rag_prompt
            
            print("RAG 시스템 초기화 중...")
            rag_system = RAGSystem()
            
            print("Gemini 모델 초기화 중...")
            gemini_model, generation_config = initialize_gemini()
            
            # create_rag_prompt 함수도 전역으로 저장
            global create_prompt
            create_prompt = create_rag_prompt
            
            print("✅ RAG 시스템 초기화 완료")
            
        except Exception as e:
            print(f"❌ RAG 시스템 초기화 실패: {e}")
            raise

def register_chat_callbacks(app):
    """채팅 기능 관련 콜백 등록 (RAG 통합)"""
    
    @app.callback(
        [Output('chat-messages', 'children'),
         Output('chat-input-field', 'value'),
         Output('chat-messages-store', 'data',allow_duplicate=True)],
        Input('chat-send-button', 'n_clicks'),
        [State('chat-input-field', 'value'),
         State('chat-messages-store', 'data'),
         State('rag-context-store', 'data'),
         State('current-step', 'data')],
        prevent_initial_call=True
    )
    def handle_chat_with_rag(n_clicks, message, messages, rag_context, current_step):
        """RAG 시스템을 활용한 채팅 메시지 처리"""
        if not n_clicks or not message or not message.strip():
            return dash.no_update, dash.no_update, dash.no_update
        
        # 7단계가 아니면 처리하지 않음
        if current_step != 7:
            return dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return dash.no_update, '', messages
        
        # RAG 시스템 초기화 확인
        if rag_system is None:
            initialize_rag_system()
        
        # 사용자 메시지 추가
        messages.append({
            'type': 'user',
            'content': message,
            'time': datetime.now().strftime('%H:%M')
        })
        
        try:
            # 1) RAG 검색
            search_results = rag_system.search_similar_documents(message, top_k=3)
            context = rag_system.create_context(search_results)
            
            # 2) 회로 분석 컨텍스트 추가
            circuit_context = _build_circuit_context(rag_context)
            if circuit_context:
                context = f"{circuit_context}\n\n{context}" if context else circuit_context
            
            # 3) 프롬프트 생성 (첫 번째 질문인지 확인)
            is_first = len([m for m in messages if m['type'] == 'user']) == 1
            practice_topic = rag_context.get('reference_circuit', '회로 분석')
            
            prompt = create_prompt(
                user_query=message,
                context=context,
                is_first_turn=is_first,
                practice_circuit_topic=practice_topic
            )
            
            # 4) Gemini로 응답 생성
            response = gemini_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            ai_response = response.text
            
        except Exception as e:
            ai_response = f'응답 생성 중 오류가 발생했습니다: {str(e)}'
        
        # AI 메시지 추가
        messages.append({
            'type': 'ai',
            'content': ai_response,
            'time': datetime.now().strftime('%H:%M')
        })
        
        # 메시지 컴포넌트 생성
        message_components = _create_message_components(messages)
        
        return message_components, '', messages

    # Enter 키로 채팅 메시지 전송
    app.clientside_callback(
        """
        function(n_clicks) {
            document.addEventListener("keypress", function(event) {
                if (event.key === "Enter") {
                    const input = document.getElementById("chat-input-field");
                    const button = document.getElementById("chat-send-button");
                    if (document.activeElement === input && input.value.trim() !== "") {
                        event.preventDefault();
                        button.click();
                    }
                }
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output('chat-input-field', 'id'),
        Input('chat-send-button', 'n_clicks')
    )

def _build_circuit_context(rag_context):
    """회로 분석 컨텍스트 구성"""
    if not rag_context:
        return ""
    
    context_parts = []
    
    # 회로 데이터
    if 'circuit_data' in rag_context:
        data = rag_context['circuit_data']
        context_parts.append(f"=== 회로 분석 결과 ===")
        context_parts.append(f"기준 회로: {data.get('reference_circuit', 'N/A')}")
        context_parts.append(f"유사도 점수: {data.get('similarity_score', 0):.1%}")
        context_parts.append(f"전압: {rag_context.get('voltage', 5.0)}V")
        
        errors = data.get('errors', [])
        if errors:
            context_parts.append(f"감지된 오류: {len(errors)}개")
            for i, error in enumerate(errors[:3], 1):
                context_parts.append(f"  {i}. {error}")
    
    # 컴포넌트 정보
    if 'component_pins' in rag_context:
        comp_summary = {}
        for comp in rag_context['component_pins']:
            cls = comp.get('class', 'Unknown')
            comp_summary[cls] = comp_summary.get(cls, 0) + 1
        
        context_parts.append("\n=== 컴포넌트 구성 ===")
        for cls, count in comp_summary.items():
            context_parts.append(f"- {cls}: {count}개")
    
    return "\n".join(context_parts)

def _create_message_components(messages):
    """메시지 리스트를 화면 컴포넌트로 변환"""
    import dash_bootstrap_components as dbc
    
    components = []
    for msg in messages:
        if msg['type'] == 'user':
            components.append(
                dbc.Row([
                    dbc.Col(width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.P(msg['content'], className='mb-1'),
                                html.Small(msg['time'], className='text-muted')
                            ])
                        ], color='primary', outline=True, className='mb-2')
                    ], width=9)
                ])
            )
        else:  # AI 메시지
            components.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className='fas fa-robot me-2'),
                                    html.Strong('AI', className='me-2')
                                ]),
                                html.P(msg['content'], className='mb-1', style={'whiteSpace': 'pre-wrap'}),
                                html.Small(msg['time'], className='text-muted')
                            ])
                        ], color='light', className='mb-2')
                    ], width=9),
                    dbc.Col(width=3)
                ])
            )
    
    return components