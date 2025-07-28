# components/power_setup.py - 전원 설정 컴포넌트
from dash import html
import dash_bootstrap_components as dbc
from config import CARD_STYLE
from utils import create_step_header, create_alert

def create_power_section(warped_img_b64):
    """6단계: 전원 설정 섹션"""
    return dbc.Card([
        dbc.CardBody([
            create_step_header(6, '전원 설정', 'fas fa-battery-full', 'prev-to-values-step6'),
            
            create_alert(
                '전원의 위치와 전압을 설정하세요. 이미지 위를 클릭하여 +/- 전원 위치를 설정할 수 있습니다.',
                color='info'
            ),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Img(src=f'data:image/png;base64,{warped_img_b64}', 
                                id='power-image', className='img-fluid',
                                style={'cursor': 'crosshair', 'maxHeight': '400px'})
                    ], className='text-center'),
                    html.Div(id='power-click-info', 
                            className='mt-2 text-center small text-muted')
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