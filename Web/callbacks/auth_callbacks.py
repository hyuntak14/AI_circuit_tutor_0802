# callbacks/auth_callbacks.py - CSV 기반 인증 콜백
import dash
from dash import Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import os
from state_manager import create_new_session, delete_session
from components.login import login_layout
from components.chat_interface import create_initial_chat_layout

# CSV 파일 경로 (상위 디렉토리)
USERS_CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'users.csv')

def load_users():
    """CSV 파일에서 사용자 정보 로드"""
    try:
        if os.path.exists(USERS_CSV_PATH):
            df = pd.read_csv(USERS_CSV_PATH)
            # user_id를 키로, 나머지 정보를 값으로 하는 딕셔너리 생성
            users = {}
            for _, row in df.iterrows():
                users[row['user_id']] = {
                    'password': row['password'],
                    'name': row['name'],
                    'email': row['email'],
                    'role': row['role']
                }
            return users
        else:
            print(f"⚠️ users.csv 파일이 없습니다. 기본 사용자 계정을 사용합니다.")
            return {'user1': {'password': 'pass1', 'name': '테스트1', 'email': '', 'role': 'student'},
                    'user2': {'password': 'pass2', 'name': '테스트2', 'email': '', 'role': 'student'}}
    except Exception as e:
        print(f"❌ CSV 파일 로드 오류: {e}")
        return {}

def register_auth_callbacks(app):
    """인증 관련 콜백 등록"""
    
    @app.callback(
        Output('page-content', 'children'),
        Input('login-state', 'data')
    )
    def display_page(login_state):
        """로그인 상태에 따른 페이지 표시"""
        if login_state and login_state.get('logged_in'):
            username = login_state.get('username', 'User')
            user_info = login_state.get('user_info', {})
            name = user_info.get('name', username)
            # 로그인 후 채팅 인터페이스로 이동
            return create_initial_chat_layout(name)
        return login_layout

    @app.callback(
        [Output('login-state', 'data'),
         Output('login-msg', 'children')],
        Input('login-btn', 'n_clicks'),
        [State('username', 'value'), State('password', 'value')],
        prevent_initial_call=True
    )
    def handle_login(n_clicks, username, password):
        """CSV 기반 로그인 처리"""
        if n_clicks and username and password:
            users = load_users()
            
            if username in users and users[username]['password'] == password:
                # 세션 생성
                session_id = create_new_session(username)
                
                # 로그인 정보 저장
                login_data = {
                    'logged_in': True,
                    'session_id': session_id,
                    'username': username,
                    'user_info': users[username]
                }
                
                return login_data, ''
            else:
                return {'logged_in': False}, dbc.Alert('로그인 실패! ID 또는 비밀번호를 확인하세요.', color='danger')
        return dash.no_update, ''

    @app.callback(
        Output('login-state', 'data', allow_duplicate=True),
        Input('logout-btn', 'n_clicks'),
        State('login-state', 'data'),
        prevent_initial_call=True
    )
    def handle_logout(n_clicks, login_state):
        """로그아웃 처리"""
        if n_clicks:
            if login_state and 'session_id' in login_state:
                delete_session(login_state['session_id'])
            return {'logged_in': False}
        return dash.no_update