# components/upload.py - 이미지 업로드 관련 컴포넌트
from dash import html, dcc
import dash_bootstrap_components as dbc
from config import CARD_STYLE
from utils import create_alert

def create_main_layout():
    """메인 레이아웃 생성"""
    return dbc.Container([
        # 헤더
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className='fas fa-microchip me-2'),
                    '회로 분석 AI'
                ], className='text-primary mb-2'),
                html.P('브레드보드 이미지를 업로드하여 회로를 자동 분석하세요', 
                      className='text-muted')
            ], width=True),
            dbc.Col([
                dbc.Button('로그아웃', id='logout-btn', color='outline-secondary', size='sm')
            ], width='auto')
        ], justify='between', align='center', className='mb-4'),
        
        # 진행 상황
        dbc.Card([
            dbc.CardBody([
                html.H6('분석 진행 상황', className='card-title mb-3'),
                dbc.Progress(id='progress-bar', value=0, striped=True, 
                           animated=True, className='mb-2'),
                html.Div(id='step-info', children='이미지를 업로드하여 시작하세요', 
                        className='text-center small text-muted')
            ])
        ], style=CARD_STYLE),
        
        # 메인 컨텐츠
        html.Div(id='main-content', children=[create_upload_section()]),
    ], fluid=True)

def create_upload_section():
    """1단계: 이미지 업로드 섹션"""
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

def create_upload_result_section(filename):
    """1단계 결과 표시"""
    return dbc.Card([
        dbc.CardBody([
            create_alert(f'이미지 업로드 완료: {filename}', 
                        color='success', icon='fas fa-check-circle'),
            html.Div([
                html.H6('업로드된 이미지:', className='mb-2'),
                html.P('이미지가 성공적으로 업로드되었습니다. 다음 단계로 진행하세요.', 
                      className='text-muted')
            ]),
            dbc.Button('다음 단계: 기준 회로 선택', id='proceed-to-reference', 
                      color='primary', size='lg', className='w-100 mt-3')
        ])
    ], style=CARD_STYLE)