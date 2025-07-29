# state_manager.py - 사용자별 세션 관리 강화 버전
import uuid
from datetime import datetime
from web_runner import WebRunnerComplete

# 글로벌 앱 상태 - 여러 사용자의 세션을 관리
app_state = {
    'sessions': {},  # session_id: session_data
    'user_sessions': {}  # user_id: [session_ids] - 사용자별 세션 목록
}

def create_new_session(username):
    """새로운 세션 생성"""
    session_id = str(uuid.uuid4())
    
    # 세션 데이터 생성
    session_data = {
        'session_id': session_id,
        'username': username,
        'runner': WebRunnerComplete(),
        'step': 1,  # 채팅 UI부터 시작
        'data': {},
        'created_at': datetime.now(),
        'last_accessed': datetime.now(),
        'drag_state': {
            'active': False,
            'start_pos': None,
            'end_pos': None,
            'selection': None
        }
    }
    
    # 글로벌 상태에 저장
    app_state['sessions'][session_id] = session_data
    
    # 사용자별 세션 목록 업데이트
    if username not in app_state['user_sessions']:
        app_state['user_sessions'][username] = []
    app_state['user_sessions'][username].append(session_id)
    
    # 현재 활성 세션 설정
    app_state['current_session'] = session_id
    
    print(f"✅ 새 세션 생성: {username} - {session_id[:8]}...")
    return session_id

def get_session_by_id(session_id):
    """세션 ID로 세션 가져오기"""
    if session_id and session_id in app_state['sessions']:
        session = app_state['sessions'][session_id]
        # 마지막 접근 시간 업데이트
        session['last_accessed'] = datetime.now()
        return session
    return None

def get_current_session():
    """현재 활성 세션 반환"""
    session_id = app_state.get('current_session')
    return get_session_by_id(session_id)

def set_current_session(session_id):
    """현재 활성 세션 설정"""
    if session_id in app_state['sessions']:
        app_state['current_session'] = session_id
        return True
    return False

def get_user_sessions(username):
    """특정 사용자의 모든 세션 목록 반환"""
    session_ids = app_state['user_sessions'].get(username, [])
    sessions = []
    for sid in session_ids:
        if sid in app_state['sessions']:
            sessions.append(app_state['sessions'][sid])
    return sessions

def delete_session(session_id=None):
    """세션 삭제"""
    if not session_id:
        session_id = app_state.get('current_session')
    
    if session_id and session_id in app_state['sessions']:
        # 세션 데이터 가져오기
        session = app_state['sessions'][session_id]
        username = session.get('username')
        
        # runner 정리
        if 'runner' in session and hasattr(session['runner'], 'cleanup'):
            try:
                session['runner'].cleanup()
            except:
                pass
        
        # 세션 삭제
        del app_state['sessions'][session_id]
        
        # 사용자 세션 목록에서 제거
        if username and username in app_state['user_sessions']:
            app_state['user_sessions'][username].remove(session_id)
            if not app_state['user_sessions'][username]:
                del app_state['user_sessions'][username]
        
        # 현재 세션이었다면 초기화
        if app_state.get('current_session') == session_id:
            app_state['current_session'] = None
        
        print(f"🗑️ 세션 삭제: {session_id[:8]}...")

def cleanup_old_sessions(hours=24):
    """오래된 세션 정리"""
    from datetime import timedelta
    
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, session in app_state['sessions'].items():
        last_accessed = session.get('last_accessed', session.get('created_at'))
        if current_time - last_accessed > timedelta(hours=hours):
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        delete_session(session_id)
    
    if expired_sessions:
        print(f"🧹 {len(expired_sessions)}개의 만료된 세션 정리")

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
    """현재 세션의 데이터 초기화 (분석 재시작)"""
    session = get_current_session()
    if session:
        # 기존 runner 정리
        if hasattr(session['runner'], 'cleanup'):
            try:
                session['runner'].cleanup()
            except:
                pass
        
        # 새 runner 생성
        session['runner'] = WebRunnerComplete()
        session['step'] = 1
        session['data'] = {}
        session['drag_state'] = {
            'active': False,
            'start_pos': None,
            'end_pos': None,
            'selection': None
        }
        print(f"♻️ 세션 데이터 초기화: {session['username']}")

# 컴포넌트 관련 함수들
def add_component(component_type, box, confidence=0.9):
    """새로운 컴포넌트 추가"""
    session = get_current_session()
    if session:
        components = session['data'].get('components', [])
        new_component = (component_type, confidence, box)
        components.append(new_component)
        session['data']['components'] = components
        return True
    return False

def update_component(index, component_type=None, box=None, confidence=None):
    """기존 컴포넌트 업데이트"""
    session = get_current_session()
    if session:
        components = session['data'].get('components', [])
        if 0 <= index < len(components):
            old_comp = components[index]
            new_comp = (
                component_type if component_type is not None else old_comp[0],
                confidence if confidence is not None else old_comp[1],
                box if box is not None else old_comp[2]
            )
            components[index] = new_comp
            session['data']['components'] = components
            return True
    return False

def delete_component(index):
    """컴포넌트 삭제"""
    session = get_current_session()
    if session:
        components = session['data'].get('components', [])
        if 0 <= index < len(components):
            components.pop(index)
            session['data']['components'] = components
            return True
    return False

def get_components():
    """현재 컴포넌트 리스트 반환"""
    return get_session_data('components', [])

# 드래그 상태 관리
def set_drag_mode(active):
    """드래그 모드 설정"""
    session = get_current_session()
    if session:
        session['drag_state']['active'] = active
        if not active:
            session['drag_state']['start_pos'] = None
            session['drag_state']['end_pos'] = None
            session['drag_state']['selection'] = None

def get_drag_mode():
    """드래그 모드 상태 반환"""
    session = get_current_session()
    if session:
        return session['drag_state']['active']
    return False

def set_drag_selection(start_pos, end_pos):
    """드래그 선택 영역 설정"""
    session = get_current_session()
    if session:
        session['drag_state']['start_pos'] = start_pos
        session['drag_state']['end_pos'] = end_pos
        session['drag_state']['selection'] = {
            'x1': min(start_pos[0], end_pos[0]),
            'y1': min(start_pos[1], end_pos[1]),
            'x2': max(start_pos[0], end_pos[0]),
            'y2': max(start_pos[1], end_pos[1])
        }

def get_drag_selection():
    """드래그 선택 영역 반환"""
    session = get_current_session()
    if session:
        return session['drag_state']['selection']
    return None

def clear_drag_selection():
    """드래그 선택 영역 초기화"""
    session = get_current_session()
    if session:
        session['drag_state']['start_pos'] = None
        session['drag_state']['end_pos'] = None
        session['drag_state']['selection'] = None

# 세션 정보 요약
def get_session_summary():
    """현재 시스템의 세션 정보 요약"""
    total_sessions = len(app_state['sessions'])
    total_users = len(app_state['user_sessions'])
    current_session_id = app_state.get('current_session')
    
    summary = {
        'total_sessions': total_sessions,
        'total_users': total_users,
        'current_session_id': current_session_id,
        'users': {}
    }
    
    for username, session_ids in app_state['user_sessions'].items():
        summary['users'][username] = len(session_ids)
    
    return summary