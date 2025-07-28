# components/value_input.py - 컴포넌트 값 입력 컴포넌트
from dash import html
import dash_bootstrap_components as dbc
from config import CARD_STYLE
from utils import create_step_header, create_alert

def create_value_section(component_pins):
    """5단계: 값 입력 섹션"""
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
            create_step_header(5, '컴포넌트 값 입력', 'fas fa-sliders-h', 'prev-to-pins-step5'),
            
            create_alert(
                '저항값, 캐패시터값 등을 입력하세요. 기본값이 설정되어 있습니다.',
                color='info'
            ),
            
            html.Div(value_inputs) if value_inputs else html.P(
                '값을 입력할 컴포넌트가 없습니다.', className='text-muted'
            ),
            
            dbc.Button('다음 단계: 전원 설정', id='confirm-values', 
                      color='success', className='w-100 mt-3')
        ])
    ], style=CARD_STYLE)