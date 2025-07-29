# main.py - 메인 애플리케이션 파일 (드래그 기능 업데이트)
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from flask import Flask, Response  # Response를 추가로 import 합니다.
import socket
import sys
import os
import requests

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from state_manager import app_state
    from components.login import login_layout
    from components.upload import create_main_layout
    from components.component_detect import create_component_edit_modal, create_add_component_modal
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
# main.py의 app.layout 부분을 다음과 같이 수정하세요:

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
    dcc.Store(id='chat-messages-store', data=[]),  # 추가
    dcc.Store(id='rag-context-store', data={}),     # 추가
    dcc.Store(id='components-store', data={}),      # 추가
    dcc.Store(id='drag-selection-store', data={}),  # 추가
    dcc.Store(id='drag-mode-active', data=False),   # 추가
    dcc.Interval(id='progress-interval', interval=2000, disabled=True),
    html.Div(id='page-content'),
    
    # 컴포넌트 편집/추가 모달들
    create_component_edit_modal(),
    create_add_component_modal()
])

# 모든 콜백 등록
register_all_callbacks(app)

@server.after_request
def apply_csp(response: Response):
    if app.server.debug:
        response.headers['Content-Security-Policy'] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-eval' 'unsafe-inline' https://cdn.jsdelivr.net https://use.fontawesome.com; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://cdn.jsdelivr.net https://use.fontawesome.com; "
            "font-src 'self' https://fonts.gstatic.com https://use.fontawesome.com; "
            "img-src 'self' data: https://cdn.jsdelivr.net https://use.fontawesome.com;"
        )
    return response


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


#공인 ip : 220.68.82.134
def get_public_ip():
    try:
        response = requests.get('https://api.ipify.org?format=json')
        return response.json()['ip']
    except:
        return "IP 확인 실패"


if __name__ == '__main__':
    local_ip = get_public_ip()
    print(f"🚀 회로 분석 AI 서버 시작")
    print(f"📱 모바일 접속: http://{local_ip}:20008")
    print(f"💻 PC 접속: http://localhost:8050")
    app.run(debug=True, host='0.0.0.0', port=8050)