# components/results.py - 결과 및 채팅 컴포넌트
from dash import html
import dash_bootstrap_components as dbc
from config import CARD_STYLE
from utils import create_step_header

def create_result_section(result):
    """7단계: 결과 및 채팅 섹션"""
    success = result.get('success')
    analysis_text = result.get('analysis_text', '분석을 시작합니다...')
    
    return dbc.Card([
        dbc.CardBody([
            create_step_header(7, '분석 완료', 'fas fa-check-circle', 'prev-to-power-step7'),
            
            dbc.Alert(
                '✅ 회로 분석이 성공적으로 완료되었습니다!' if success else '❌ 분석 중 오류가 발생했습니다',
                color='success' if success else 'danger'
            ),
            
            # 채팅 인터페이스
            dbc.Card([
                dbc.CardHeader([
                    html.I(className='fas fa-robot me-2'),
                    'AI 분석 결과 및 대화'
                ]),
                dbc.CardBody([
                    html.Div(id='chat-window', 
                            style={'height': '300px', 'overflowY': 'auto', 
                                   'backgroundColor': '#f8f9fa', 'padding': '1rem',
                                   'borderRadius': '0.375rem'},
                            children=[
                                html.Div([
                                    html.Strong('AI: ', className='text-success'),
                                    html.Span(analysis_text)
                                ], className='mb-2')
                            ]),
                    
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(id='chat-input', placeholder='질문을 입력하세요...', 
                                     type='text')
                        ], width=10),
                        dbc.Col([
                            dbc.Button('전송', id='send-message', color='primary', 
                                     className='w-100')
                        ], width=2)
                    ], className='mt-3')
                ])
            ]),
            
            dbc.Button('새로운 분석 시작', id='restart-analysis', 
                      color='outline-secondary', className='w-100 mt-3')
        ])
    ], style=CARD_STYLE)