# components/component_detect.py - 드래그 기능이 추가된 컴포넌트 검출 컴포넌트
from dash import html, dcc
import dash_bootstrap_components as dbc
from config import CARD_STYLE, COMPONENT_TYPES
from utils import create_step_header, create_alert, format_component_list_enhanced

def create_component_section(components, warped_img_b64, component_img_b64=None):
    """3단계: 컴포넌트 확인 섹션 (드래그 기능 추가)"""
    # 컴포넌트가 표시된 이미지 사용 (있으면)
    display_img = component_img_b64 if component_img_b64 else warped_img_b64
    
    # 컴포넌트 리스트 생성
    component_cards = format_component_list_enhanced(components)
    
    return dbc.Card([
        dbc.CardBody([
            create_step_header(3, f'컴포넌트 확인 ({len(components)}개 발견)', 
                             'fas fa-microchip', 'prev-to-reference-step3'),
            
            create_alert(
                '검출된 컴포넌트를 확인하고 수정/삭제하거나 새로운 컴포넌트를 추가하세요.',
                color='info'
            ),
            
            # 새 컴포넌트 추가 버튼 (항상 표시)
            dbc.Row([
                dbc.Col([
                    dbc.Button([
                        html.I(className='fas fa-plus me-2'),
                        '새 컴포넌트 추가'
                    ], id='toggle-add-component-panel', 
                       color='primary', size='sm', className='mb-3')
                ], width=12)
            ]),
            
            # 새 컴포넌트 추가 패널 (기본적으로 숨김)
            dbc.Collapse([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className='fas fa-crosshairs me-2'),
                        '새 컴포넌트 추가 - 드래그 모드'
                    ]),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label('컴포넌트 타입:', className='form-label'),
                                dcc.Dropdown(
                                    id='new-component-type',
                                    options=[{'label': name, 'value': name} for name in COMPONENT_TYPES],
                                    value='Resistor',
                                    className='mb-2'
                                ),
                            ], width=6),
                            dbc.Col([
                                dbc.Button('드래그 모드 활성화', id='activate-drag-mode', 
                                         color='outline-success', size='sm', className='mb-2'),
                                html.Div(id='drag-mode-status', className='small text-muted')
                            ], width=6)
                        ]),
                        html.Hr(),
                        html.P([
                            html.Strong('사용법: '),
                            '1) 컴포넌트 타입 선택 → 2) 드래그 모드 활성화 → 3) 이미지에서 영역 드래그 → 4) 추가 버튼 클릭'
                        ], className='small text-info mb-2'),
                        html.Div([
                            html.Strong('선택된 영역: '),
                            html.Span(id='selected-area-info', children='영역을 선택하세요', 
                                     className='text-muted')
                        ], className='mb-3'),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button('선택된 영역에 컴포넌트 추가', id='add-component-from-drag',
                                         color='success', size='sm', disabled=True, className='w-100')
                            ], width=8),
                            dbc.Col([
                                dbc.Button('취소', id='cancel-add-component',
                                         color='secondary', size='sm', className='w-100')
                            ], width=4)
                        ])
                    ], className='py-2')
                ], className='mb-3', style={'backgroundColor': '#f0f8ff', 'border': '2px dashed #007bff'})
            ], id='add-component-panel', is_open=False),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6('검출 결과 이미지', className='text-center mb-2'),
                        html.Div([
                            html.Img(
                                src=f'data:image/png;base64,{display_img}', 
                                id='component-image',
                                className='img-fluid',
                                style={
                                    'maxHeight': '500px',
                                    'border': '2px solid #dee2e6',
                                    'borderRadius': '0.375rem',
                                    'cursor': 'default'
                                }
                            ),
                            # 드래그 오버레이 (선택 박스 표시용)
                            html.Div(
                                id='drag-overlay',
                                style={
                                    'position': 'absolute',
                                    'top': '0',
                                    'left': '0',
                                    'width': '100%',
                                    'height': '100%',
                                    'pointerEvents': 'none',
                                    'zIndex': '10'
                                }
                            )
                        ], style={'position': 'relative', 'display': 'inline-block'}),
                        html.Div(id='image-click-info', 
                                className='mt-2 text-center small text-muted')
                    ], className='text-center')
                ], width=12, lg=7),
                dbc.Col([
                    html.H6('검출된 컴포넌트 목록', className='mb-3'),
                    html.Div(component_cards, 
                            id='component-list',
                            style={'maxHeight': '500px', 'overflowY': 'auto'})
                ], width=12, lg=5)
            ]),
            
            dbc.Button('다음 단계: 핀 위치 검출', id='confirm-components', 
                      color='success', className='w-100 mt-3'),
            
            # 숨겨진 스토어들
            dcc.Store(id='components-store', data=components),
            dcc.Store(id='drag-selection-store', data={}),
            dcc.Store(id='drag-mode-active', data=False)
        ])
    ], style=CARD_STYLE)

def create_component_edit_modal():
    """컴포넌트 편집 모달"""
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("컴포넌트 편집")),
        dbc.ModalBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label('컴포넌트 타입:', className='form-label'),
                    dcc.Dropdown(
                        id='edit-component-type',
                        options=[{'label': name, 'value': name} for name in COMPONENT_TYPES],
                        className='mb-3'
                    )
                ], width=12),
                dbc.Col([
                    dbc.Label('신뢰도:', className='form-label'),
                    html.P(id='edit-component-confidence', className='form-control-plaintext')
                ], width=6),
                dbc.Col([
                    dbc.Label('위치 (x1, y1, x2, y2):', className='form-label'),
                    dbc.InputGroup([
                        dbc.Input(id='edit-x1', type='number', placeholder='x1'),
                        dbc.Input(id='edit-y1', type='number', placeholder='y1'),
                        dbc.Input(id='edit-x2', type='number', placeholder='x2'),
                        dbc.Input(id='edit-y2', type='number', placeholder='y2')
                    ])
                ], width=6)
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button('저장', id='save-component-edit', color='primary', className='me-2'),
            dbc.Button('취소', id='cancel-component-edit', color='secondary')
        ])
    ], id='component-edit-modal', is_open=False)

def create_add_component_modal():
    """새 컴포넌트 추가 모달 (수동 좌표 입력용 - 백업)"""
    return dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("새 컴포넌트 추가 (수동 입력)")),
        dbc.ModalBody([
            dbc.Row([
                dbc.Col([
                    dbc.Label('컴포넌트 타입:', className='form-label'),
                    dcc.Dropdown(
                        id='manual-component-type',
                        options=[{'label': name, 'value': name} for name in COMPONENT_TYPES],
                        value='Resistor',
                        className='mb-3'
                    )
                ], width=12),
                dbc.Col([
                    dbc.Label('위치 설정:', className='form-label'),
                    html.P('좌상단 (x1, y1)과 우하단 (x2, y2) 좌표를 입력하세요.', 
                          className='small text-muted mb-2')
                ], width=12),
                dbc.Col([
                    dbc.Label('x1 (좌):', className='form-label'),
                    dbc.Input(id='manual-x1', type='number', value=50, min=0)
                ], width=3),
                dbc.Col([
                    dbc.Label('y1 (상):', className='form-label'),
                    dbc.Input(id='manual-y1', type='number', value=50, min=0)
                ], width=3),
                dbc.Col([
                    dbc.Label('x2 (우):', className='form-label'),
                    dbc.Input(id='manual-x2', type='number', value=150, min=0)
                ], width=3),
                dbc.Col([
                    dbc.Label('y2 (하):', className='form-label'),
                    dbc.Input(id='manual-y2', type='number', value=100, min=0)
                ], width=3)
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button('추가', id='add-manual-component', color='success', className='me-2'),
            dbc.Button('취소', id='cancel-manual-component', color='secondary')
        ])
    ], id='manual-add-component-modal', is_open=False)