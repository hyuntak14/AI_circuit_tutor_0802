# main.py - 메인 애플리케이션 파일
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from flask import Flask
import socket
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from state_manager import app_state
    from components.login import login_layout
    from components.upload import create_main_layout
    from callbacks import register_all_callbacks
except ImportError as e:
    print(f"Import 오류: {e}")
    print("다음을 확인하세요:")
    print("1. 모든 파일이 올바른 위치에 있는지")
    print("2. __init__.py 파일들이 존재하는지")
    print("3. 폴더 구조가 올바른지")
    sys.exit(1)

# Flask 서버
server = Flask(__name__)
server.secret_key = 'circuit-analyzer-2025'

# Dash 앱 초기화 (직접 설정)
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}]
)

# 앱 레이아웃
app.layout = html.Div([
    dcc.Store(id='login-state', data={'logged_in': False}),
    dcc.Store(id='session-store', data={}),
    dcc.Store(id='analysis-store', data={}),
    dcc.Store(id='current-component', data=-1),
    dcc.Store(id='power-positions-store', data=[]),
    dcc.Store(id='current-step', data=0),
    dcc.Store(id='processing-state', data={'is_processing': False, 'operation': ''}),
    dcc.Store(id='upload-result-store', data={}),
    dcc.Store(id='component-result-store', data={}),
    dcc.Store(id='pin-result-store', data={}),
    dcc.Interval(id='progress-interval', interval=2000, disabled=True),
    html.Div(id='page-content')
])

# 모든 콜백 등록
register_all_callbacks(app)

def get_local_ip():
    """로컬 IP 주소 가져오기"""
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