# dash_app_fixed.py - 콜백 에러 및 핀 검출 문제가 해결된 회로 분석기
import dash
from dash import html, dcc, Input, Output, State, callback_context, ALL, MATCH
import dash_bootstrap_components as dbc
from flask import Flask
import base64
import cv2
import numpy as np  
import os
import json
import tempfile
import uuid
from web_runner import WebRunnerComplete

# Flask 서버
server = Flask(__name__)
server.secret_key = 'circuit-analyzer-2025'

# Dash 앱
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}]
)

# 글로벌 상태
app_state = {'sessions': {}, 'current_session': None}

# 스타일 상수
CARD_STYLE = {'marginBottom': '1rem', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}
BUTTON_STYLE = {'marginBottom': '0.5rem'}

# 로딩 컴포넌트
def create_loading_component(message="처리 중..."):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                dbc.Spinner(color="primary", size="lg"),
                html.H5(message, className="mt-3 text-center"),
                html.P("잠시만 기다려주세요.", className="text-muted text-center")
            ], className="text-center p-4")
        ])
    ], style=CARD_STYLE)

# 이전 단계 버튼 생성
def create_prev_button(button_id, is_disabled=False):
    return dbc.Button([
        html.I(className='fas fa-arrow-left me-2'),
        '이전 단계'
    ], id=button_id, color='outline-secondary', size='sm', disabled=is_disabled, className='me-2')

# 로그인 페이지
login_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className='fas fa-microchip fa-3x text-primary mb-3'),
                        html.H2('회로 분석 AI', className='text-center mb-4'),
                        dbc.Input(id='username', placeholder='사용자 ID', className='mb-3', size='lg'),
                        dbc.Input(id='password', type='password', placeholder='비밀번호', className='mb-3', size='lg'),
                        dbc.Button('로그인', id='login-btn', color='primary', size='lg', className='w-100 mb-3'),
                        html.Div(id='login-msg', className='text-center'),
                        html.Hr(),
                        dbc.Alert([
                            html.Strong('테스트 계정: '),
                            'user1/pass1 또는 user2/pass2'
                        ], color='info', className='text-center')
                    ], className='text-center')
                ])
            ], style=CARD_STYLE)
        ], width=12, lg=4)
    ], justify='center', className='min-vh-100 align-items-center')
], fluid=True)

# 메인 레이아웃
def create_main_layout():
    return dbc.Container([
        # 헤더
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className='fas fa-microchip me-2'),
                    '회로 분석 AI'
                ], className='text-primary mb-2'),
                html.P('브레드보드 이미지를 업로드하여 회로를 자동 분석하세요', className='text-muted')
            ], width=True),
            dbc.Col([
                dbc.Button('로그아웃', id='logout-btn', color='outline-secondary', size='sm')
            ], width='auto')
        ], justify='between', align='center', className='mb-4'),
        
        # 진행 상황
        dbc.Card([
            dbc.CardBody([
                html.H6('분석 진행 상황', className='card-title mb-3'),
                dbc.Progress(id='progress-bar', value=0, striped=True, animated=True, className='mb-2'),
                html.Div(id='step-info', children='이미지를 업로드하여 시작하세요', className='text-center small text-muted')
            ])
        ], style=CARD_STYLE),
        
        # 메인 컨텐츠
        html.Div(id='main-content', children=[
            create_upload_section()
        ]),
        
        # 숨겨진 스토어들
        dcc.Store(id='processing-state', data={'is_processing': False, 'operation': ''}),
        dcc.Store(id='upload-result-store', data={}),
        dcc.Store(id='component-result-store', data={}),
        dcc.Store(id='pin-result-store', data={}),
        dcc.Store(id='current-step', data=0),
    ], fluid=True)

# 1단계: 이미지 업로드
def create_upload_section():
    return dbc.Card([
        dbc.CardBody([
            html.H5([
                html.I(className='fas fa-upload me-2'),
                '1단계: 브레드보드 이미지 업로드'
            ], className='card-title'),
            dcc.Upload(
                id='image-upload',
                children=html.Div([
                    html.I(className='fas fa-cloud-upload-alt fa-4x text-muted mb-3'),
                    html.H5('이미지를 드래그하거나 클릭하여 업로드', className='text-muted'),
                    html.P('JPG, PNG 파일만 지원됩니다', className='small text-muted')
                ], className='text-center p-4'),
                style={
                    'width': '100%', 'height': '200px', 'borderWidth': '2px',
                    'borderStyle': 'dashed', 'borderRadius': '10px', 'textAlign': 'center',
                    'cursor': 'pointer', 'backgroundColor': '#f8f9fa'
                }
            ),
            html.Div(id='upload-result', className='mt-3')
        ])
    ], style=CARD_STYLE)

# 1단계 결과 표시
def create_upload_result_section(filename):
    return dbc.Card([
        dbc.CardBody([
            dbc.Alert([
                html.I(className='fas fa-check-circle me-2'),
                f'이미지 업로드 완료: {filename}'
            ], color='success'),
            html.Div([
                html.H6('업로드된 이미지:', className='mb-2'),
                html.P('이미지가 성공적으로 업로드되었습니다. 다음 단계로 진행하세요.', className='text-muted')
            ]),
            dbc.Button('다음 단계: 기준 회로 선택', id='proceed-to-reference', 
                      color='primary', size='lg', className='w-100 mt-3')
        ])
    ], style=CARD_STYLE)

# 2단계: 기준 회로 선택 (12개 회로)
def create_reference_section():
    circuit_topics = {
        1: "병렬회로", 2: "직렬회로", 3: "키르히호프 1법칙", 4: "키르히호프 2법칙",
        5: "중첩의 원리-a", 6: "중첩의 원리-b", 7: "중첩의 원리-c", 8: "교류 전원", 
        9: "오실로스코프1", 10: "반파정류회로", 11: "반파정류회로2", 12: "비반전 증폭기"
    }
    
    circuit_buttons = []
    for circuit_id, name in circuit_topics.items():
        circuit_buttons.append(
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6(f"{circuit_id}. {name}", className='card-title'),
                        dbc.Button('선택', id={'type': 'ref-btn', 'circuit': circuit_id}, 
                                 color='outline-primary', size='sm', className='w-100')
                    ])
                ], className='h-100')
            ], width=12, md=6, lg=4, className='mb-3')
        )
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                create_prev_button('prev-to-upload-step2'),
                html.H5([
                    html.I(className='fas fa-sitemap me-2'),
                    '2단계: 기준 회로 선택'
                ], className='card-title d-inline')
            ], className='d-flex align-items-center mb-3'),
            html.P('비교할 기준 회로를 선택하세요 (선택사항)', className='text-muted'),
            dbc.Row(circuit_buttons),
            dbc.Button('비교 건너뛰기', id='skip-reference', color='secondary', className='w-100 mt-2'),
            html.Div(id='reference-result', className='mt-3')
        ])
    ], style=CARD_STYLE)

# 2단계 결과 표시
def create_reference_result_section(selected_circuit):
    circuit_topics = {
        1: "병렬회로", 2: "직렬회로", 3: "키르히호프 1법칙", 4: "키르히호프 2법칙",
        5: "중첩의 원리-a", 6: "중첩의 원리-b", 7: "중첩의 원리-c", 8: "교류 전원", 
        9: "오실로스코프1", 10: "반파정류회로", 11: "반파정류회로2", 12: "비반전 증폭기"
    }
    
    if selected_circuit == 'skip':
        message = '기준 회로 비교를 건너뛰기로 선택하였습니다.'
        icon = 'fas fa-forward'
    else:
        circuit_name = circuit_topics.get(selected_circuit, f"회로 {selected_circuit}")
        message = f'기준 회로로 "{circuit_name}"를 선택하였습니다.'
        icon = 'fas fa-check-circle'
    
    return dbc.Card([
        dbc.CardBody([
            dbc.Alert([
                html.I(className=f'{icon} me-2'),
                message
            ], color='info'),
            dbc.Button('다음 단계: 컴포넌트 검출 시작', id='start-component-detection', 
                      color='primary', size='lg', className='w-100 mt-3')
        ])
    ], style=CARD_STYLE)

# 3단계: 컴포넌트 확인
def create_component_section(components, warped_img_b64, component_img_b64=None):
    component_cards = []
    for i, (cls, conf, box) in enumerate(components):
        component_cards.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6(f'{cls}', className='mb-1'),
                            dbc.Badge(f'{conf:.1%}', color='info', className='me-2'),
                            html.Small(f'위치: {box}', className='text-muted')
                        ], width=8),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button('수정', id={'type': 'edit-comp', 'idx': i}, 
                                         size='sm', color='outline-warning'),
                                dbc.Button('삭제', id={'type': 'del-comp', 'idx': i}, 
                                         size='sm', color='outline-danger')
                            ], size='sm')
                        ], width=4, className='text-end')
                    ])
                ])
            ], className='mb-2')
        )
    
    # 컴포넌트가 표시된 이미지 사용 (있으면)
    display_img = component_img_b64 if component_img_b64 else warped_img_b64
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                create_prev_button('prev-to-reference-step3'),
                html.H5([
                    html.I(className='fas fa-microchip me-2'),
                    f'3단계: 컴포넌트 확인 ({len(components)}개 발견)'
                ], className='card-title d-inline')
            ], className='d-flex align-items-center mb-3'),
            
            dbc.Alert([
                html.I(className='fas fa-info-circle me-2'),
                '검출된 컴포넌트를 확인하고 필요시 수정 또는 삭제하세요. 빨간 박스로 표시된 영역이 검출된 컴포넌트입니다.'
            ], color='info'),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6('검출 결과 이미지', className='text-center mb-2'),
                        html.Img(src=f'data:image/png;base64,{display_img}', 
                                className='img-fluid', style={'maxHeight': '400px'})
                    ], className='text-center')
                ], width=12, lg=6),
                dbc.Col([
                    html.H6('검출된 컴포넌트 목록', className='mb-3'),
                    html.Div(component_cards, style={'maxHeight': '400px', 'overflowY': 'auto'})
                ], width=12, lg=6)
            ]),
            
            dbc.Button('다음 단계: 핀 위치 검출', id='confirm-components', 
                      color='success', className='w-100 mt-3')
        ])
    ], style=CARD_STYLE)

# 4단계: 핀 위치 설정
def create_pin_section(component_pins, warped_img_b64, pin_img_b64=None):
    pin_status = []
    for i, comp in enumerate(component_pins):
        expected = 8 if comp['class'] == 'IC' else 2
        current = len(comp.get('pins', []))
        status = 'success' if current == expected else 'warning'
        
        pin_status.append(
            dbc.ListGroupItem([
                dbc.Row([
                    dbc.Col([
                        html.Strong(f"{i+1}. {comp['class']}"),
                        html.Br(),
                        html.Small(f'핀 {current}/{expected}개', className='text-muted')
                    ], width=8),
                    dbc.Col([
                        dbc.Badge('완료' if current == expected else '설정필요', 
                                color=status, className='me-2'),
                        dbc.Button('설정', id={'type': 'pin-btn', 'idx': i}, 
                                 size='sm', color='outline-primary')
                    ], width=4, className='text-end')
                ])
            ])
        )
    
    # 핀이 표시된 이미지 사용 (있으면)
    display_img = pin_img_b64 if pin_img_b64 else warped_img_b64
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                create_prev_button('prev-to-components-step4'),
                html.H5([
                    html.I(className='fas fa-crosshairs me-2'),
                    '4단계: 핀 위치 설정'
                ], className='card-title d-inline')
            ], className='d-flex align-items-center mb-3'),
            
            dbc.Alert([
                html.I(className='fas fa-info-circle me-2'),
                '각 컴포넌트의 핀 위치가 자동으로 설정되었습니다. 파란 점이 검출된 핀 위치입니다. 필요시 클릭하여 수정하세요.'
            ], color='info'),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6('핀 검출 결과 이미지', className='text-center mb-2'),
                        html.Img(src=f'data:image/png;base64,{display_img}', 
                                id='pin-image', className='img-fluid',
                                style={'cursor': 'crosshair', 'maxHeight': '500px'})
                    ], className='text-center'),
                    html.Div(id='pin-click-info', className='mt-2 text-center small text-muted')
                ], width=12, lg=8),
                dbc.Col([
                    html.H6('핀 설정 상태', className='mb-3'),
                    dbc.ListGroup(pin_status)
                ], width=12, lg=4)
            ]),
            
            dbc.Button('다음 단계: 컴포넌트 값 입력', id='confirm-pins', 
                      color='success', className='w-100 mt-3')
        ])
    ], style=CARD_STYLE)

# 5단계: 값 입력
def create_value_section(component_pins):
    value_inputs = []
    for i, comp in enumerate(component_pins):
        if comp['class'] in ['Resistor', 'Capacitor']:
            unit = 'Ω' if comp['class'] == 'Resistor' else 'F'
            default_val = 100 if comp['class'] == 'Resistor' else 0.001
            
            value_inputs.append(
                dbc.Row([
                    dbc.Col([
                        dbc.Label(f"{i+1}. {comp['class']}", className='form-label')
                    ], width=4),
                    dbc.Col([
                        dbc.InputGroup([
                            dbc.Input(id={'type': 'comp-value', 'idx': i}, 
                                    type='number', value=default_val, step='any'),
                            dbc.InputGroupText(unit)
                        ])
                    ], width=8)
                ], className='mb-3')
            )
    
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                create_prev_button('prev-to-pins-step5'),
                html.H5([
                    html.I(className='fas fa-sliders-h me-2'),
                    '5단계: 컴포넌트 값 입력'
                ], className='card-title d-inline')
            ], className='d-flex align-items-center mb-3'),
            
            dbc.Alert([
                html.I(className='fas fa-info-circle me-2'),
                '저항값, 캐패시터값 등을 입력하세요. 기본값이 설정되어 있습니다.'
            ], color='info'),
            
            html.Div(value_inputs) if value_inputs else html.P('값을 입력할 컴포넌트가 없습니다.', className='text-muted'),
            
            dbc.Button('다음 단계: 전원 설정', id='confirm-values', 
                      color='success', className='w-100 mt-3')
        ])
    ], style=CARD_STYLE)

# 6단계: 전원 설정
def create_power_section(warped_img_b64):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                create_prev_button('prev-to-values-step6'),
                html.H5([
                    html.I(className='fas fa-battery-full me-2'),
                    '6단계: 전원 설정'
                ], className='card-title d-inline')
            ], className='d-flex align-items-center mb-3'),
            
            dbc.Alert([
                html.I(className='fas fa-info-circle me-2'),
                '전원의 위치와 전압을 설정하세요. 이미지 위를 클릭하여 +/- 전원 위치를 설정할 수 있습니다.'
            ], color='info'),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Img(src=f'data:image/png;base64,{warped_img_b64}', 
                                id='power-image', className='img-fluid',
                                style={'cursor': 'crosshair', 'maxHeight': '400px'})
                    ], className='text-center'),
                    html.Div(id='power-click-info', className='mt-2 text-center small text-muted')
                ], width=12, lg=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6('전원 설정'),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label('전압 (V)', className='form-label'),
                                    dbc.Input(id='power-voltage', type='number', 
                                            value=5.0, step=0.1, min=0.1, max=50)
                                ], width=12, className='mb-3'),
                            ]),
                            html.Div(id='power-positions', className='mb-3'),
                            dbc.Button('전원 위치 설정', id='set-power-btn', 
                                     color='outline-primary', className='w-100 mb-3'),
                            dbc.Button('회로 생성 및 분석 시작', id='start-circuit-generation', 
                                     color='success', size='lg', className='w-100')
                        ])
                    ])
                ], width=12, lg=4)
            ])
        ])
    ], style=CARD_STYLE)

# 7단계: 결과 및 채팅
def create_result_section(result):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                create_prev_button('prev-to-power-step7'),
                html.H5([
                    html.I(className='fas fa-check-circle me-2'),
                    '7단계: 분석 완료'
                ], className='card-title d-inline')
            ], className='d-flex align-items-center mb-3'),
            
            dbc.Alert(
                '✅ 회로 분석이 성공적으로 완료되었습니다!' if result.get('success') else '❌ 분석 중 오류가 발생했습니다',
                color='success' if result.get('success') else 'danger'
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
                                    html.Span(result.get('analysis_text', '분석을 시작합니다...'))
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

# 앱 레이아웃
app.layout = html.Div([
    dcc.Store(id='login-state', data={'logged_in': False}),
    dcc.Store(id='session-store', data={}),
    dcc.Store(id='analysis-store', data={}),
    dcc.Store(id='current-component', data=-1),
    dcc.Store(id='power-positions-store', data=[]),
    dcc.Store(id='current-step', data=0),
    dcc.Interval(id='progress-interval', interval=2000, disabled=True),
    html.Div(id='page-content')
])

# 로그인 처리
VALID_USERS = {'user1': 'pass1', 'user2': 'pass2'}

@app.callback(
    Output('page-content', 'children'),
    Input('login-state', 'data')
)
def display_page(login_state):
    if login_state and login_state.get('logged_in'):
        return create_main_layout()
    return login_layout

@app.callback(
    [Output('login-state', 'data'),
     Output('login-msg', 'children')],
    Input('login-btn', 'n_clicks'),
    [State('username', 'value'), State('password', 'value')],
    prevent_initial_call=True
)
def handle_login(n_clicks, username, password):
    if n_clicks and username and password:
        if VALID_USERS.get(username) == password:
            session_id = str(uuid.uuid4())
            app_state['current_session'] = session_id
            app_state['sessions'][session_id] = {
                'username': username,
                'runner': WebRunnerComplete(),
                'step': 0,
                'data': {}
            }
            return {'logged_in': True, 'session_id': session_id}, ''
        else:
            return {'logged_in': False}, dbc.Alert('로그인 실패! 다시 시도하세요.', color='danger')
    return dash.no_update, ''

@app.callback(
    Output('login-state', 'data', allow_duplicate=True),
    Input('logout-btn', 'n_clicks'),
    prevent_initial_call=True
)
def handle_logout(n_clicks):
    if n_clicks:
        session_id = app_state.get('current_session')
        if session_id and session_id in app_state['sessions']:
            del app_state['sessions'][session_id]
        app_state['current_session'] = None
        return {'logged_in': False}
    return dash.no_update

# 1단계: 이미지 업로드 처리
@app.callback(
    [Output('upload-result', 'children'),
     Output('current-step', 'data'),
     Output('step-info', 'children')],
    Input('image-upload', 'contents'),
    State('image-upload', 'filename'),
    prevent_initial_call=True
)
def handle_image_upload(contents, filename):
    if not contents:
        return '', 0, '이미지를 선택하세요'
    
    try:
        # 이미지 디코딩 및 저장
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # 세션 가져오기
        session_id = app_state.get('current_session')
        if not session_id or session_id not in app_state['sessions']:
            return dbc.Alert('세션이 만료되었습니다. 다시 로그인하세요.', color='danger'), 0, '오류'
        
        session = app_state['sessions'][session_id]
        
        # 임시 파일 저장
        temp_dir = tempfile.mkdtemp()
        image_path = os.path.join(temp_dir, filename)
        with open(image_path, 'wb') as f:
            f.write(decoded)
        
        # 세션에 데이터 저장
        session['data']['image_path'] = image_path
        session['data']['filename'] = filename
        session['step'] = 1
        
        return (create_upload_result_section(filename), 
                1, 
                '이미지가 업로드되었습니다. 다음 단계를 진행하세요.')
        
    except Exception as e:
        return dbc.Alert(f'업로드 실패: {str(e)}', color='danger'), 0, '오류 발생'

# 1→2단계: 기준 회로 선택으로 이동
@app.callback(
    [Output('main-content', 'children'),
     Output('progress-bar', 'value'),
     Output('step-info', 'children', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True)],
    Input('proceed-to-reference', 'n_clicks'),
    prevent_initial_call=True
)
def proceed_to_reference_selection(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    return create_reference_section(), 20, '기준 회로를 선택하세요', 2

# 2단계: 기준 회로 선택 처리
@app.callback(
    Output('reference-result', 'children'),
    [Input({'type': 'ref-btn', 'circuit': ALL}, 'n_clicks'),
     Input('skip-reference', 'n_clicks')],
    prevent_initial_call=True
)
def handle_reference_selection(ref_clicks, skip_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return ''
    
    session_id = app_state.get('current_session')
    if not session_id or session_id not in app_state['sessions']:
        return dbc.Alert('세션 만료', color='danger')
    
    session = app_state['sessions'][session_id]
    
    # 어떤 버튼이 클릭되었는지 확인
    trigger_id = ctx.triggered[0]['prop_id']
    
    if 'skip-reference' in trigger_id:
        session['data']['reference_circuit'] = 'skip'
        return create_reference_result_section('skip')
    else:
        # 기준 회로 선택 파싱
        import json
        button_id = json.loads(trigger_id.split('.')[0])
        circuit_id = button_id['circuit']
        session['data']['reference_circuit'] = circuit_id
        return create_reference_result_section(circuit_id)

# 2→3단계: 컴포넌트 검출 시작
@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('progress-bar', 'value', allow_duplicate=True),
     Output('step-info', 'children', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True)],
    Input('start-component-detection', 'n_clicks'),
    prevent_initial_call=True
)
def start_component_detection(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    session_id = app_state.get('current_session')
    if not session_id or session_id not in app_state['sessions']:
        return dbc.Alert('세션 만료', color='danger'), 0, '오류', 2
    
    session = app_state['sessions'][session_id]
    
    try:
        # 백그라운드에서 컴포넌트 검출 실행
        runner = session['runner']
        result = runner.detect_components(session['data']['image_path'])
        
        if result['success']:
            session['data']['components'] = result['components']
            session['data']['warped_image'] = result['warped_image_b64']
            session['data']['component_image'] = result.get('component_image_b64')  # 시각화된 이미지
            session['step'] = 3
            
            content = create_component_section(
                result['components'], 
                result['warped_image_b64'],
                result.get('component_image_b64')
            )
            return content, 40, '컴포넌트를 확인하고 수정하세요', 3
        else:
            return dbc.Alert('컴포넌트 검출 실패', color='danger'), 20, '오류 발생', 2
            
    except Exception as e:
        return dbc.Alert(f'처리 오류: {str(e)}', color='danger'), 20, '오류 발생', 2

# 3→4단계: 핀 검출 시작
@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('progress-bar', 'value', allow_duplicate=True),
     Output('step-info', 'children', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True)],
    Input('confirm-components', 'n_clicks'),
    prevent_initial_call=True
)
def start_pin_detection(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    session_id = app_state.get('current_session')
    if not session_id or session_id not in app_state['sessions']:
        return dbc.Alert('세션 만료', color='danger'), 0, '오류', 3
    
    session = app_state['sessions'][session_id]
    
    try:
        runner = session['runner']
        result = runner.detect_pins_advanced(
            session['data']['components'], 
            session['data']['image_path'],
            session['data']['warped_image']
        )
        
        if result['success']:
            session['data']['component_pins'] = result['component_pins']
            session['data']['pin_image'] = result.get('pin_image_b64')  # 시각화된 이미지
            session['data']['holes'] = result.get('holes', [])  # 구멍 데이터 저장
            session['step'] = 4
            
            content = create_pin_section(
                result['component_pins'], 
                session['data']['warped_image'],
                result.get('pin_image_b64')
            )
            return content, 60, '각 컴포넌트의 핀 위치를 확인하세요', 4
        else:
            return dbc.Alert('핀 검출 실패', color='danger'), 40, '오류 발생', 3
            
    except Exception as e:
        return dbc.Alert(f'핀 검출 오류: {str(e)}', color='danger'), 40, '오류 발생', 3

# 4→5단계: 값 입력으로 이동
@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('progress-bar', 'value', allow_duplicate=True),
     Output('step-info', 'children', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True)],
    Input('confirm-pins', 'n_clicks'),
    prevent_initial_call=True
)
def proceed_to_values(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    session_id = app_state.get('current_session')
    if not session_id or session_id not in app_state['sessions']:
        return dbc.Alert('세션 만료', color='danger'), 0, '오류', 4
    
    session = app_state['sessions'][session_id]
    session['step'] = 5
    
    return (create_value_section(session['data']['component_pins']), 
            70, '컴포넌트 값을 입력하세요', 5)

# 5→6단계: 전원 설정으로 이동
@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('progress-bar', 'value', allow_duplicate=True),
     Output('step-info', 'children', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True)],
    Input('confirm-values', 'n_clicks'),
    [State({'type': 'comp-value', 'idx': ALL}, 'value')],
    prevent_initial_call=True
)
def proceed_to_power(n_clicks, values):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    session_id = app_state.get('current_session')
    if not session_id or session_id not in app_state['sessions']:
        return dbc.Alert('세션 만료', color='danger'), 0, '오류', 5
    
    session = app_state['sessions'][session_id]
    
    # 컴포넌트 값 업데이트
    value_idx = 0
    for i, comp in enumerate(session['data']['component_pins']):
        if comp['class'] in ['Resistor', 'Capacitor']:
            if value_idx < len(values) and values[value_idx] is not None:
                comp['value'] = values[value_idx]
            value_idx += 1
    
    session['step'] = 6
    
    return (create_power_section(session['data']['warped_image']), 
            80, '전원을 설정하세요', 6)

# 6→7단계: 최종 회로 생성
@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('progress-bar', 'value', allow_duplicate=True),
     Output('step-info', 'children', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True)],
    Input('start-circuit-generation', 'n_clicks'),
    State('power-voltage', 'value'),
    prevent_initial_call=True
)
def start_circuit_generation(n_clicks, voltage):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    session_id = app_state.get('current_session')
    if not session_id or session_id not in app_state['sessions']:
        return dbc.Alert('세션 만료', color='danger'), 0, '오류', 6
    
    session = app_state['sessions'][session_id]
    
    try:
        runner = session['runner']
        result = runner.generate_circuit_and_analyze(
            session['data']['component_pins'],
            voltage or 5.0,
            session['data']['reference_circuit']
        )
        
        session['data']['final_result'] = result
        session['step'] = 7
        
        return (create_result_section(result), 
                100, '분석 완료! AI와 대화해보세요', 7)
        
    except Exception as e:
        return (dbc.Alert(f'회로 생성 오류: {str(e)}', color='danger'), 
                80, '오류 발생', 6)

# 각 단계별 이전 버튼 콜백들 (개별로 분리)
@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('progress-bar', 'value', allow_duplicate=True),
     Output('step-info', 'children', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True)],
    Input('prev-to-upload-step2', 'n_clicks'),
    prevent_initial_call=True
)
def prev_to_upload_step2(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    session_id = app_state.get('current_session')
    if not session_id or session_id not in app_state['sessions']:
        return dbc.Alert('세션 만료', color='danger'), 0, '오류', 0
    
    session = app_state['sessions'][session_id]
    session['step'] = 1
    
    if session['data'].get('filename'):
        content = create_upload_result_section(session['data']['filename'])
    else:
        content = create_upload_section()
    
    return content, 10, '이미지 업로드 단계', 1

@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('progress-bar', 'value', allow_duplicate=True),
     Output('step-info', 'children', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True)],
    Input('prev-to-reference-step3', 'n_clicks'),
    prevent_initial_call=True
)
def prev_to_reference_step3(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    session_id = app_state.get('current_session')
    if not session_id or session_id not in app_state['sessions']:
        return dbc.Alert('세션 만료', color='danger'), 0, '오류', 0
    
    session = app_state['sessions'][session_id]
    session['step'] = 2
    
    return create_reference_section(), 20, '기준 회로를 선택하세요', 2

@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('progress-bar', 'value', allow_duplicate=True),
     Output('step-info', 'children', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True)],
    Input('prev-to-components-step4', 'n_clicks'),
    prevent_initial_call=True
)
def prev_to_components_step4(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    session_id = app_state.get('current_session')
    if not session_id or session_id not in app_state['sessions']:
        return dbc.Alert('세션 만료', color='danger'), 0, '오류', 0
    
    session = app_state['sessions'][session_id]
    session['step'] = 3
    
    if session['data'].get('components'):
        content = create_component_section(
            session['data']['components'], 
            session['data']['warped_image'],
            session['data'].get('component_image')
        )
        return content, 40, '컴포넌트를 확인하고 수정하세요', 3
    else:
        return create_reference_section(), 20, '기준 회로를 선택하세요', 2

@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('progress-bar', 'value', allow_duplicate=True),
     Output('step-info', 'children', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True)],
    Input('prev-to-pins-step5', 'n_clicks'),
    prevent_initial_call=True
)
def prev_to_pins_step5(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    session_id = app_state.get('current_session')
    if not session_id or session_id not in app_state['sessions']:
        return dbc.Alert('세션 만료', color='danger'), 0, '오류', 0
    
    session = app_state['sessions'][session_id]
    session['step'] = 4
    
    if session['data'].get('component_pins'):
        content = create_pin_section(
            session['data']['component_pins'], 
            session['data']['warped_image'],
            session['data'].get('pin_image')
        )
        return content, 60, '각 컴포넌트의 핀 위치를 확인하세요', 4
    else:
        content = create_component_section(
            session['data']['components'], 
            session['data']['warped_image'],
            session['data'].get('component_image')
        )
        return content, 40, '컴포넌트를 확인하고 수정하세요', 3

@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('progress-bar', 'value', allow_duplicate=True),
     Output('step-info', 'children', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True)],
    Input('prev-to-values-step6', 'n_clicks'),
    prevent_initial_call=True
)
def prev_to_values_step6(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    session_id = app_state.get('current_session')
    if not session_id or session_id not in app_state['sessions']:
        return dbc.Alert('세션 만료', color='danger'), 0, '오류', 0
    
    session = app_state['sessions'][session_id]
    session['step'] = 5
    
    return create_value_section(session['data']['component_pins']), 70, '컴포넌트 값을 입력하세요', 5

@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('progress-bar', 'value', allow_duplicate=True),
     Output('step-info', 'children', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True)],
    Input('prev-to-power-step7', 'n_clicks'),
    prevent_initial_call=True
)
def prev_to_power_step7(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    session_id = app_state.get('current_session')
    if not session_id or session_id not in app_state['sessions']:
        return dbc.Alert('세션 만료', color='danger'), 0, '오류', 0
    
    session = app_state['sessions'][session_id]
    session['step'] = 6
    
    return create_power_section(session['data']['warped_image']), 80, '전원을 설정하세요', 6

# 채팅 기능
@app.callback(
    [Output('chat-window', 'children'),
     Output('chat-input', 'value')],
    Input('send-message', 'n_clicks'),
    [State('chat-input', 'value'),
     State('chat-window', 'children')],
    prevent_initial_call=True
)
def handle_chat(n_clicks, message, current_messages):
    if not n_clicks or not message or not message.strip():
        return dash.no_update, dash.no_update
    
    session_id = app_state.get('current_session')
    if not session_id or session_id not in app_state['sessions']:
        return current_messages, ''
    
    session = app_state['sessions'][session_id]
    
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
        ai_response = runner.get_ai_response(message, session['data'].get('final_result', {}))
        
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

# 재시작
@app.callback(
    [Output('main-content', 'children', allow_duplicate=True),
     Output('progress-bar', 'value', allow_duplicate=True),
     Output('step-info', 'children', allow_duplicate=True),
     Output('current-step', 'data', allow_duplicate=True)],
    Input('restart-analysis', 'n_clicks'),
    prevent_initial_call=True
)
def restart_analysis(n_clicks):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    session_id = app_state.get('current_session')
    if session_id and session_id in app_state['sessions']:
        # 세션 데이터 초기화
        session = app_state['sessions'][session_id]
        session['step'] = 0
        session['data'] = {}
        session['runner'] = WebRunnerComplete()
    
    return create_upload_section(), 0, '새로운 이미지를 업로드하세요', 0

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

# 서버 실행
def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

if __name__ == '__main__': 
    local_ip = get_local_ip()
    print(f"🚀 회로 분석 AI 서버 시작")
    print(f"📱 모바일 접속: http://{local_ip}:8050")
    print(f"💻 PC 접속: http://localhost:8050")
    app.run(debug=True, host='0.0.0.0', port=8050)