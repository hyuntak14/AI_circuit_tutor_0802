# state_manager.py - 글로벌 상태 관리
import uuid
from web_runner import WebRunnerComplete

# 글로벌 앱 상태
app_state = {'sessions': {}, 'current_session': None}

def create_new_session(username):
    """새로운 세션 생성"""
    session_id = str(uuid.uuid4())
    app_state['current_session'] = session_id
    app_state['sessions'][session_id] = {
        'username': username,
        'runner': WebRunnerComplete(),
        'step': 0,
        'data': {}
    }
    return session_id

def get_current_session():
    """현재 세션 반환"""
    session_id = app_state.get('current_session')
    if session_id and session_id in app_state['sessions']:
        return app_state['sessions'][session_id]
    return None

def delete_session(session_id=None):
    """세션 삭제"""
    if not session_id:
        session_id = app_state.get('current_session')
    
    if session_id and session_id in app_state['sessions']:
        del app_state['sessions'][session_id]
    
    if app_state.get('current_session') == session_id:
        app_state['current_session'] = None

def update_session_step(step):
    """현재 세션의 단계 업데이트"""
    session = get_current_session()
    if session:
        session['step'] = step

def update_session_data(key, value):
    """현재 세션의 데이터 업데이트"""
    session = get_current_session()
    if session:
        session['data'][key] = value

def get_session_data(key, default=None):
    """현재 세션의 데이터 가져오기"""
    session = get_current_session()
    if session:
        return session['data'].get(key, default)
    return default

def reset_session_data():
    """현재 세션의 데이터 초기화"""
    session = get_current_session()
    if session:
        session['step'] = 0
        session['data'] = {}
        session['runner'] = WebRunnerComplete()