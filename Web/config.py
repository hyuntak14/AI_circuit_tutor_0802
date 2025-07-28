# config.py - 설정 및 상수 정의
import dash_bootstrap_components as dbc

# 유효 사용자 계정
VALID_USERS = {'user1': 'pass1', 'user2': 'pass2'}

# 스타일 상수
CARD_STYLE = {'marginBottom': '1rem', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'}
BUTTON_STYLE = {'marginBottom': '0.5rem'}

# 회로 주제 목록
CIRCUIT_TOPICS = {
    1: "병렬회로", 
    2: "직렬회로", 
    3: "키르히호프 1법칙", 
    4: "키르히호프 2법칙",
    5: "중첩의 원리-a", 
    6: "중첩의 원리-b", 
    7: "중첩의 원리-c", 
    8: "교류 전원", 
    9: "오실로스코프1", 
    10: "반파정류회로", 
    11: "반파정류회로2", 
    12: "비반전 증폭기"
}

# Dash 앱 설정
def get_dash_config(server):
    """Dash 앱 설정 반환"""
    return {
        '__name__': '__main__',
        'server': server,
        'external_stylesheets': [dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        'suppress_callback_exceptions': True,
        'meta_tags': [{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}]
    }

# 진행률 설정
PROGRESS_STEPS = {
    0: 0,   # 시작
    1: 10,  # 이미지 업로드
    2: 20,  # 기준 회로 선택
    3: 40,  # 컴포넌트 검출
    4: 60,  # 핀 위치 설정
    5: 70,  # 값 입력
    6: 80,  # 전원 설정
    7: 100  # 완료
}

# 각 단계별 메시지
STEP_MESSAGES = {
    0: '이미지를 업로드하여 시작하세요',
    1: '이미지가 업로드되었습니다. 다음 단계를 진행하세요.',
    2: '기준 회로를 선택하세요',
    3: '컴포넌트를 확인하고 수정하세요',
    4: '각 컴포넌트의 핀 위치를 확인하세요',
    5: '컴포넌트 값을 입력하세요',
    6: '전원을 설정하세요',
    7: '분석 완료! AI와 대화해보세요'
}