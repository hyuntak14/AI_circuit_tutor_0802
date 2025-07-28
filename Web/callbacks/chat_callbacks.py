# callbacks/chat_callbacks.py - 채팅 기능 콜백
import dash
from dash import Input, Output, State, html

from state_manager import get_current_session, get_session_data

def register_chat_callbacks(app):
    """채팅 기능 관련 콜백 등록"""
    
    @app.callback(
        [Output('chat-window', 'children'),
         Output('chat-input', 'value')],
        Input('send-message', 'n_clicks'),
        [State('chat-input', 'value'),
         State('chat-window', 'children')],
        prevent_initial_call=True
    )
    def handle_chat(n_clicks, message, current_messages):
        """채팅 메시지 처리"""
        if not n_clicks or not message or not message.strip():
            return dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return current_messages, ''
        
        # 사용자 메시지 추가
        new_messages = current_messages or []
        new_messages.append(
            html.Div([
                html.Strong('You: ', className='text-primary'),
                html.Span(message)
            ], className='mb-2')
        )
        
        try:
            # AI 응답 생성
            runner = session['runner']
            ai_response = runner.get_ai_response(
                message, 
                get_session_data('final_result', {})
            )
            
            new_messages.append(
                html.Div([
                    html.Strong('AI: ', className='text-success'),
                    html.Span(ai_response)
                ], className='mb-2')
            )
            
        except Exception as e:
            new_messages.append(
                html.Div([
                    html.Strong('AI: ', className='text-danger'),
                    html.Span(f'응답 생성 중 오류가 발생했습니다: {str(e)}')
                ], className='mb-2')
            )
        
        return new_messages, ''

    # Enter 키로 채팅 메시지 전송
    app.clientside_callback(
        """
        function(id) {
            document.addEventListener("keypress", function(event) {
                if (event.key === "Enter") {
                    let input = document.getElementById("chat-input");
                    let button = document.getElementById("send-message");
                    if (document.activeElement === input && input.value.trim() !== "") {
                        button.click();
                    }
                }
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output('chat-input', 'id'),
        Input('chat-input', 'id')
    )