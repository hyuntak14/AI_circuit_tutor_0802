# components/component_detect.py - 컴포넌트 검출 컴포넌트
from dash import html
import dash_bootstrap_components as dbc
from config import CARD_STYLE
from utils import create_step_header, create_alert, format_component_list

def create_component_section(components, warped_img_b64, component_img_b64=None):
    """3단계: 컴포넌트 확인 섹션"""
    # 컴포넌트가 표시된 이미지 사용 (있으면)
    display_img = component_img_b64 if component_img_b64 else warped_img_b64
    
    # 컴포넌트 리스트 생성
    component_cards = format_component_list(components)
    
    return dbc.Card([
        dbc.CardBody([
            create_step_header(3, f'컴포넌트 확인 ({len(components)}개 발견)', 
                             'fas fa-microchip', 'prev-to-reference-step3'),
            
            create_alert(
                '검출된 컴포넌트를 확인하고 필요시 수정 또는 삭제하세요. 빨간 박스로 표시된 영역이 검출된 컴포넌트입니다.',
                color='info'
            ),
            
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
                    html.Div(component_cards, 
                            style={'maxHeight': '400px', 'overflowY': 'auto'})
                ], width=12, lg=6)
            ]),
            
            dbc.Button('다음 단계: 핀 위치 검출', id='confirm-components', 
                      color='success', className='w-100 mt-3')
        ])
    ], style=CARD_STYLE)