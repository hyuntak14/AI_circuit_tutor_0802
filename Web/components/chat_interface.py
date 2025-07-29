# components/chat_interface.py - 채팅 인터페이스 컴포넌트
from dash import html, dcc
import dash_bootstrap_components as dbc
from config import CARD_STYLE
from datetime import datetime

def create_chat_interface(messages=None, show_upload=True):
    """
    채팅 인터페이스 생성
    
    Args:
        messages: 채팅 메시지 리스트 [{'type': 'user'/'ai', 'content': '...', 'time': '...'}, ...]
        show_upload: 이미지 업로드 표시 여부 (1단계에서만 True)
    """
    if messages is None:
        messages = []
    
    # 초기 메시지 추가
    if not messages and show_upload:
        messages = [{
            'type': 'ai',
            'content': '안녕하세요! 회로 분석 AI입니다. 브레드보드 이미지를 업로드해주세요.',
            'time': datetime.now().strftime('%H:%M')
        }]
    
    # 메시지 표시 컴포넌트 생성
    message_components = []
    for msg in messages:
        if msg['type'] == 'user':
            message_components.append(
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
            message_components.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.Div([
                                    html.I(className='fas fa-robot me-2'),
                                    html.Strong('AI', className='me-2')
                                ]),
                                html.P(msg['content'], className='mb-1'),
                                html.Small(msg['time'], className='text-muted')
                            ])
                        ], color='light', className='mb-2')
                    ], width=9),
                    dbc.Col(width=3)
                ])
            )
    
    # 이미지 업로드 컴포넌트 (1단계에서만 표시)
    upload_component = None
    if show_upload:
        upload_component = dbc.Card([
            dbc.CardBody([
                dcc.Upload(
                    id='chat-image-upload',
                    children=html.Div([
                        html.I(className='fas fa-cloud-upload-alt fa-3x text-muted mb-2'),
                        html.P('이미지를 드래그하거나 클릭하여 업로드', className='mb-0'),
                        html.Small('JPG, PNG 파일 지원', className='text-muted')
                    ], className='text-center'),
                    style={
                        'width': '100%',
                        'height': '120px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '8px',
                        'cursor': 'pointer',
                        'backgroundColor': '#f8f9fa'
                    }
                )
            ])
        ], className='mb-3')
    
    # 전체 레이아웃
    return dbc.Card([
        dbc.CardBody([
            # 헤더
            dbc.Row([
                dbc.Col([
                    html.H5([
                        html.I(className='fas fa-comments me-2'),
                        '회로 분석 대화'
                    ], className='mb-0')
                ]),
                dbc.Col([
                    dbc.Button([
                        html.I(className='fas fa-history me-2'),
                        '대화 기록'
                    ], id='show-chat-history', color='outline-secondary', size='sm')
                ], width='auto')
            ], className='mb-3'),
            
            # 채팅 메시지 영역
            html.Div(
                message_components,
                id='chat-messages',
                style={
                    'height': '400px',
                    'overflowY': 'auto',
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '8px',
                    'padding': '1rem'
                }
            ),
            
            # 이미지 업로드 영역 (조건부)
            upload_component,
            
            # 입력 영역 (7단계에서만 활성화)
            dbc.Row([
                dbc.Col([
                    dbc.InputGroup([
                        dbc.Input(
                            id='chat-input-field',
                            placeholder='메시지를 입력하세요...',
                            disabled=show_upload  # 1단계에서는 비활성화
                        ),
                        dbc.Button([
                            html.I(className='fas fa-paper-plane')
                        ], id='chat-send-button', color='primary', disabled=show_upload)
                    ])
                ])
            ], className='mt-3') if not show_upload else html.Div()
        ])
    ], style=CARD_STYLE)

def create_initial_chat_layout(username):
    """로그인 후 초기 채팅 레이아웃"""
    return dbc.Container([
        # 헤더
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className='fas fa-microchip me-2'),
                    '회로 분석 AI'
                ], className='text-primary mb-2'),
                html.P(f'{username}님, 환영합니다! 회로 분석을 시작해보세요.', 
                      className='text-muted')
            ], width=True),
            dbc.Col([
                dbc.Button('로그아웃', id='logout-btn', color='outline-secondary', size='sm')
            ], width='auto')
        ], justify='between', align='center', className='mb-4'),
        
        # 진행 상황 바
        dbc.Card([
            dbc.CardBody([
                html.H6('분석 진행 상황', className='card-title mb-3'),
                dbc.Progress(id='progress-bar', value=0, striped=True, 
                           animated=True, className='mb-2'),
                html.Div(id='step-info', children='1단계: 이미지 업로드', 
                        className='text-center small text-muted')
            ])
        ], style=CARD_STYLE),
        
        # 채팅 인터페이스
        html.Div(id='main-content', children=[create_chat_interface(show_upload=True)]),
        
        # 숨겨진 스토어들
        dcc.Store(id='chat-messages-store', data=[]),
        dcc.Store(id='rag-context-store', data={})
    ], fluid=True)

def create_final_chat_interface(analysis_result):
    """7단계: 최종 분석 결과와 함께 채팅 인터페이스 표시"""
    initial_messages = [{
        'type': 'ai',
        'content': analysis_result,
        'time': datetime.now().strftime('%H:%M')
    }, {
        'type': 'ai',
        'content': '분석이 완료되었습니다. 궁금한 점이 있으시면 질문해주세요!',
        'time': datetime.now().strftime('%H:%M')
    }]
    
    return create_chat_interface(messages=initial_messages, show_upload=False)