# callbacks/auth_callbacks.py - 인증 관련 콜백
import dash
from dash import Input, Output, State
import dash_bootstrap_components as dbc
from config import VALID_USERS
from state_manager import create_new_session, delete_session
from components.login import login_layout
from components.upload import create_main_layout

def register_auth_callbacks(app):
    """인증 관련 콜백 등록"""
    
    @app.callback(
        Output('page-content', 'children'),
        Input('login-state', 'data')
    )
    def display_page(login_state):
        """로그인 상태에 따른 페이지 표시"""
        if login_state and login_state.get('logged_in'):
            return create_main_layout()
        return login_layout

    @app.callback(
        [Output('login-state', 'data'),
         Output('login-msg', 'children')],
        Input('login-btn', 'n_clicks'),
        [State('username', 'value'), State('password', 'value')],
        prevent_initial_call=True
    )
    def handle_login(n_clicks, username, password):
        """로그인 처리"""
        if n_clicks and username and password:
            if VALID_USERS.get(username) == password:
                session_id = create_new_session(username)
                return {'logged_in': True, 'session_id': session_id}, ''
            else:
                return {'logged_in': False}, dbc.Alert('로그인 실패! 다시 시도하세요.', color='danger')
        return dash.no_update, ''

    @app.callback(
        Output('login-state', 'data', allow_duplicate=True),
        Input('logout-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def handle_logout(n_clicks):
        """로그아웃 처리"""
        if n_clicks:
            delete_session()
            return {'logged_in': False}
        return dash.no_update