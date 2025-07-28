# components/pin_setup.py - 핀 위치 설정 컴포넌트
from dash import html
import dash_bootstrap_components as dbc
from config import CARD_STYLE
from utils import create_step_header, create_alert, format_pin_status

def create_pin_section(component_pins, warped_img_b64, pin_img_b64=None):
    """4단계: 핀 위치 설정 섹션"""
    # 핀이 표시된 이미지 사용 (있으면)
    display_img = pin_img_b64 if pin_img_b64 else warped_img_b64
    
    # 핀 상태 리스트 생성
    pin_status = format_pin_status(component_pins)
    
    return dbc.Card([
        dbc.CardBody([
            create_step_header(4, '핀 위치 설정', 'fas fa-crosshairs', 'prev-to-components-step4'),
            
            create_alert(
                '각 컴포넌트의 핀 위치가 자동으로 설정되었습니다. 파란 점이 검출된 핀 위치입니다. 필요시 클릭하여 수정하세요.',
                color='info'
            ),
            
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6('핀 검출 결과 이미지', className='text-center mb-2'),
                        html.Img(src=f'data:image/png;base64,{display_img}', 
                                id='pin-image', className='img-fluid',
                                style={'cursor': 'crosshair', 'maxHeight': '500px'})
                    ], className='text-center'),
                    html.Div(id='pin-click-info', 
                            className='mt-2 text-center small text-muted')
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