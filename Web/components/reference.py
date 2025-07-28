# components/reference.py - 기준 회로 선택 컴포넌트
from dash import html
import dash_bootstrap_components as dbc
from config import CARD_STYLE, CIRCUIT_TOPICS
from utils import create_step_header, create_alert

def create_reference_section():
    """2단계: 기준 회로 선택 섹션"""
    circuit_buttons = []
    for circuit_id, name in CIRCUIT_TOPICS.items():
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
            create_step_header(2, '기준 회로 선택', 'fas fa-sitemap', 'prev-to-upload-step2'),
            html.P('비교할 기준 회로를 선택하세요 (선택사항)', className='text-muted'),
            dbc.Row(circuit_buttons),
            dbc.Button('비교 건너뛰기', id='skip-reference', 
                      color='secondary', className='w-100 mt-2'),
            html.Div(id='reference-result', className='mt-3')
        ])
    ], style=CARD_STYLE)

def create_reference_result_section(selected_circuit):
    """2단계 결과 표시"""
    if selected_circuit == 'skip':
        message = '기준 회로 비교를 건너뛰기로 선택하였습니다.'
        icon = 'fas fa-forward'
    else:
        circuit_name = CIRCUIT_TOPICS.get(selected_circuit, f"회로 {selected_circuit}")
        message = f'기준 회로로 "{circuit_name}"를 선택하였습니다.'
        icon = 'fas fa-check-circle'
    
    return dbc.Card([
        dbc.CardBody([
            create_alert(message, color='info', icon=icon),
            dbc.Button('다음 단계: 컴포넌트 검출 시작', id='start-component-detection', 
                      color='primary', size='lg', className='w-100 mt-3')
        ])
    ], style=CARD_STYLE)