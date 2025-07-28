# callbacks/drag_callbacks.py - 컴포넌트 드래그 추가 기능 콜백
import dash
from dash import Input, Output, State, callback_context, no_update, html
import dash_bootstrap_components as dbc

from state_manager import add_component, get_components
from utils import format_component_list_enhanced

def register_drag_callbacks(app):
    """컴포넌트 드래그 추가 관련 콜백 등록"""

    # '새 컴포넌트 추가' 패널 열기/닫기
    @app.callback(
        Output('add-component-panel', 'is_open'),
        Input('toggle-add-component-panel', 'n_clicks'),
        State('add-component-panel', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_add_component_panel(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return no_update

    # 드래그 모드 활성화/비활성화
    @app.callback(
        [Output('drag-mode-active', 'data'),
         Output('activate-drag-mode', 'color'),
         Output('activate-drag-mode', 'children'),
         Output('drag-mode-status', 'children'),
         Output('drag-overlay', 'style'),
         Output('selected-area-info', 'children'),
         Output('add-component-from-drag', 'disabled')],
        Input('activate-drag-mode', 'n_clicks'),
        State('drag-mode-active', 'data'),
        prevent_initial_call=True
    )
    def toggle_drag_mode(n_clicks, is_active):
        if not n_clicks:
            return no_update

        new_state = not is_active
        if new_state:
            color = 'danger'
            button_text = [html.I(className='fas fa-times me-2'), '드래그 모드 취소']
            status_text = '활성화됨. 이미지에서 영역을 드래그하세요.'
        else:
            color = 'outline-success'
            button_text = '드래그 모드 활성화'
            status_text = '비활성화됨.'
        
        # 드래그 관련 상태 초기화
        overlay_style = {'display': 'none'}
        area_info = '영역을 선택하세요'
        add_button_disabled = True
        
        return new_state, color, button_text, status_text, overlay_style, area_info, add_button_disabled

    # 드래그로 선택된 영역 정보 업데이트
    @app.callback(
        [Output('selected-area-info', 'children', allow_duplicate=True),
         Output('add-component-from-drag', 'disabled', allow_duplicate=True)],
        Input('drag-selection-store', 'data'),
        prevent_initial_call=True
    )
    def update_selected_area_info(selection_data):
        if selection_data and all(k in selection_data for k in ['x1', 'y1', 'x2', 'y2']):
            info = f"x1: {selection_data['x1']}, y1: {selection_data['y1']}, x2: {selection_data['x2']}, y2: {selection_data['y2']}"
            return info, False  # 추가 버튼 활성화
        return '영역을 선택하세요', True # 추가 버튼 비활성화

    # 드래그로 선택한 영역에 컴포넌트 추가
    @app.callback(
        [Output('component-list', 'children', allow_duplicate=True),
         Output('add-component-panel', 'is_open', allow_duplicate=True),
         Output('drag-mode-active', 'data', allow_duplicate=True),
         Output('drag-selection-store', 'data', allow_duplicate=True)],
        Input('add-component-from-drag', 'n_clicks'),
        [State('new-component-type', 'value'),
         State('drag-selection-store', 'data')],
        prevent_initial_call=True
    )
    def add_component_from_drag(n_clicks, comp_type, selection_data):
        if not n_clicks or not selection_data:
            return no_update, no_update, no_update, no_update

        box = [
            selection_data['x1'],
            selection_data['y1'],
            selection_data['x2'],
            selection_data['y2']
        ]
        
        # 새 컴포넌트 추가 (신뢰도는 1.0으로 설정)
        add_component(comp_type, box, 1.0)
        
        updated_components = get_components()
        new_list = format_component_list_enhanced(updated_components)
        
        # 상태 초기화: 패널 닫고, 드래그 모드 비활성화, 선택 영역 초기화
        return new_list, False, False, {}