# callbacks/navigation_callbacks.py - 네비게이션 콜백
import dash
from dash import Input, Output
import dash_bootstrap_components as dbc

from config import PROGRESS_STEPS, STEP_MESSAGES
from state_manager import (
    get_current_session, update_session_step, 
    get_session_data, reset_session_data
)
from components import *

def register_navigation_callbacks(app):
    """네비게이션 관련 콜백 등록"""
    
    # 2→1단계: 업로드로 돌아가기
    @app.callback(
        [Output('main-content', 'children', allow_duplicate=True),
         Output('progress-bar', 'value', allow_duplicate=True),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True)],
        Input('prev-to-upload-step2', 'n_clicks'),
        prevent_initial_call=True
    )
    def prev_to_upload_step2(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return dbc.Alert('세션 만료', color='danger'), 0, '오류', 0
        
        update_session_step(1)
        
        filename = get_session_data('filename')
        if filename:
            content = create_upload_result_section(filename)
        else:
            content = create_upload_section()
        
        return content, PROGRESS_STEPS[1], STEP_MESSAGES[1], 1

    # 3→2단계: 기준 회로 선택으로 돌아가기
    @app.callback(
        [Output('main-content', 'children', allow_duplicate=True),
         Output('progress-bar', 'value', allow_duplicate=True),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True)],
        Input('prev-to-reference-step3', 'n_clicks'),
        prevent_initial_call=True
    )
    def prev_to_reference_step3(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return dbc.Alert('세션 만료', color='danger'), 0, '오류', 0
        
        update_session_step(2)
        return create_reference_section(), PROGRESS_STEPS[2], STEP_MESSAGES[2], 2

    # 4→3단계: 컴포넌트 검출로 돌아가기
    @app.callback(
        [Output('main-content', 'children', allow_duplicate=True),
         Output('progress-bar', 'value', allow_duplicate=True),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True)],
        Input('prev-to-components-step4', 'n_clicks'),
        prevent_initial_call=True
    )
    def prev_to_components_step4(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return dbc.Alert('세션 만료', color='danger'), 0, '오류', 0
        
        update_session_step(3)
        
        components = get_session_data('components')
        if components:
            content = create_component_section(
                components, 
                get_session_data('warped_image'),
                get_session_data('component_image')
            )
            return content, PROGRESS_STEPS[3], STEP_MESSAGES[3], 3
        else:
            return create_reference_section(), PROGRESS_STEPS[2], STEP_MESSAGES[2], 2

    # 5→4단계: 핀 위치 설정으로 돌아가기
    @app.callback(
        [Output('main-content', 'children', allow_duplicate=True),
         Output('progress-bar', 'value', allow_duplicate=True),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True)],
        Input('prev-to-pins-step5', 'n_clicks'),
        prevent_initial_call=True
    )
    def prev_to_pins_step5(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return dbc.Alert('세션 만료', color='danger'), 0, '오류', 0
        
        update_session_step(4)
        
        component_pins = get_session_data('component_pins')
        if component_pins:
            content = create_pin_section(
                component_pins, 
                get_session_data('warped_image'),
                get_session_data('pin_image')
            )
            return content, PROGRESS_STEPS[4], STEP_MESSAGES[4], 4
        else:
            content = create_component_section(
                get_session_data('components'), 
                get_session_data('warped_image'),
                get_session_data('component_image')
            )
            return content, PROGRESS_STEPS[3], STEP_MESSAGES[3], 3

    # 6→5단계: 값 입력으로 돌아가기
    @app.callback(
        [Output('main-content', 'children', allow_duplicate=True),
         Output('progress-bar', 'value', allow_duplicate=True),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True)],
        Input('prev-to-values-step6', 'n_clicks'),
        prevent_initial_call=True
    )
    def prev_to_values_step6(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return dbc.Alert('세션 만료', color='danger'), 0, '오류', 0
        
        update_session_step(5)
        return (create_value_section(get_session_data('component_pins')), 
                PROGRESS_STEPS[5], STEP_MESSAGES[5], 5)

    # 7→6단계: 전원 설정으로 돌아가기
    @app.callback(
        [Output('main-content', 'children', allow_duplicate=True),
         Output('progress-bar', 'value', allow_duplicate=True),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True)],
        Input('prev-to-power-step7', 'n_clicks'),
        prevent_initial_call=True
    )
    def prev_to_power_step7(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return dbc.Alert('세션 만료', color='danger'), 0, '오류', 0
        
        update_session_step(6)
        return (create_power_section(get_session_data('warped_image')), 
                PROGRESS_STEPS[6], STEP_MESSAGES[6], 6)

    # 재시작
    @app.callback(
        [Output('main-content', 'children', allow_duplicate=True),
         Output('progress-bar', 'value', allow_duplicate=True),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True)],
        Input('restart-analysis', 'n_clicks'),
        prevent_initial_call=True
    )
    def restart_analysis(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if session:
            reset_session_data()
        
        return (create_upload_section(), PROGRESS_STEPS[0], STEP_MESSAGES[0], 0)