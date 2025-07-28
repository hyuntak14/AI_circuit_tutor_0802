# components/login.py - 로그인 페이지 컴포넌트
from dash import html
import dash_bootstrap_components as dbc
from config import CARD_STYLE

login_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.I(className='fas fa-microchip fa-3x text-primary mb-3'),
                        html.H2('회로 분석 AI', className='text-center mb-4'),
                        dbc.Input(id='username', placeholder='사용자 ID', 
                                className='mb-3', size='lg'),
                        dbc.Input(id='password', type='password', placeholder='비밀번호', 
                                className='mb-3', size='lg'),
                        dbc.Button('로그인', id='login-btn', color='primary', 
                                 size='lg', className='w-100 mb-3'),
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