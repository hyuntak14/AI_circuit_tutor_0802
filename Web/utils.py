# utils.py - 유틸리티 함수들
from dash import html
import dash_bootstrap_components as dbc
from config import CARD_STYLE

def create_loading_component(message="처리 중..."):
    """로딩 컴포넌트 생성"""
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                dbc.Spinner(color="primary", size="lg"),
                html.H5(message, className="mt-3 text-center"),
                html.P("잠시만 기다려주세요.", className="text-muted text-center")
            ], className="text-center p-4")
        ])
    ], style=CARD_STYLE)

def create_prev_button(button_id, is_disabled=False):
    """이전 단계 버튼 생성"""
    return dbc.Button([
        html.I(className='fas fa-arrow-left me-2'),
        '이전 단계'
    ], id=button_id, color='outline-secondary', size='sm', 
       disabled=is_disabled, className='me-2')

def create_alert(message, color='info', icon='fas fa-info-circle'):
    """알림 컴포넌트 생성"""
    return dbc.Alert([
        html.I(className=f'{icon} me-2'),
        message
    ], color=color)

def create_step_header(step_num, title, icon_class, prev_button_id=None):
    """단계별 헤더 생성"""
    header_content = []
    
    if prev_button_id:
        header_content.append(create_prev_button(prev_button_id))
    
    header_content.append(
        html.H5([
            html.I(className=f'{icon_class} me-2'),
            f'{step_num}단계: {title}'
        ], className='card-title d-inline')
    )
    
    return html.Div(header_content, className='d-flex align-items-center mb-3')

def format_component_list(components):
    """컴포넌트 리스트를 카드 형태로 포맷"""
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
    return component_cards

def format_pin_status(component_pins):
    """핀 상태를 리스트 형태로 포맷"""
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
    return pin_status