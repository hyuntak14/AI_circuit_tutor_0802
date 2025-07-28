# state_manager.py - 글로벌 상태 관리 (강화된 버전)
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
        'data': {},
        'drag_state': {
            'active': False,
            'start_pos': None,
            'end_pos': None,
            'selection': None
        }
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
        session['drag_state'] = {
            'active': False,
            'start_pos': None,
            'end_pos': None,
            'selection': None
        }

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
            # 드래그 모드 비활성화시 선택 초기화
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