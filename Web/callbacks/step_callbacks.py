# callbacks/step_callbacks.py - 단계별 처리 콜백
import dash
from dash import Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import base64
import tempfile
import os
import json

from config import PROGRESS_STEPS, STEP_MESSAGES
from state_manager import (
    get_current_session, update_session_step, 
    update_session_data, get_session_data
)
from components import *

def register_step_callbacks(app):
    """단계별 처리 콜백 등록"""
    
    # 1단계: 이미지 업로드 처리
    @app.callback(
        [Output('upload-result', 'children'),
         Output('current-step', 'data'),
         Output('step-info', 'children')],
        Input('image-upload', 'contents'),
        State('image-upload', 'filename'),
        prevent_initial_call=True
    )
    def handle_image_upload(contents, filename):
        """이미지 업로드 처리"""
        if not contents:
            return '', 0, '이미지를 선택하세요'
        
        try:
            # 이미지 디코딩 및 저장
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            session = get_current_session()
            if not session:
                return dbc.Alert('세션이 만료되었습니다. 다시 로그인하세요.', color='danger'), 0, '오류'
            
            # 임시 파일 저장
            temp_dir = tempfile.mkdtemp()
            image_path = os.path.join(temp_dir, filename)
            with open(image_path, 'wb') as f:
                f.write(decoded)
            
            # 세션에 데이터 저장
            update_session_data('image_path', image_path)
            update_session_data('filename', filename)
            update_session_step(1)
            
            return (create_upload_result_section(filename), 1, 
                    '이미지가 업로드되었습니다. 다음 단계를 진행하세요.')
            
        except Exception as e:
            return dbc.Alert(f'업로드 실패: {str(e)}', color='danger'), 0, '오류 발생'

    # 1→2단계: 기준 회로 선택으로 이동
    @app.callback(
        [Output('main-content', 'children'),
         Output('progress-bar', 'value'),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True)],
        Input('proceed-to-reference', 'n_clicks'),
        prevent_initial_call=True
    )
    def proceed_to_reference_selection(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        return create_reference_section(), PROGRESS_STEPS[2], STEP_MESSAGES[2], 2

    # 2단계: 기준 회로 선택 처리
    @app.callback(
        Output('reference-result', 'children'),
        [Input({'type': 'ref-btn', 'circuit': ALL}, 'n_clicks'),
         Input('skip-reference', 'n_clicks')],
        prevent_initial_call=True
    )
    def handle_reference_selection(ref_clicks, skip_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return ''
        
        session = get_current_session()
        if not session:
            return dbc.Alert('세션 만료', color='danger')
        
        trigger_id = ctx.triggered[0]['prop_id']
        
        if 'skip-reference' in trigger_id:
            update_session_data('reference_circuit', 'skip')
            return create_reference_result_section('skip')
        else:
            button_id = json.loads(trigger_id.split('.')[0])
            circuit_id = button_id['circuit']
            update_session_data('reference_circuit', circuit_id)
            return create_reference_result_section(circuit_id)

    # 2→3단계: 컴포넌트 검출 시작
    @app.callback(
        [Output('main-content', 'children', allow_duplicate=True),
         Output('progress-bar', 'value', allow_duplicate=True),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True)],
        Input('start-component-detection', 'n_clicks'),
        prevent_initial_call=True
    )
    def start_component_detection(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return dbc.Alert('세션 만료', color='danger'), 0, '오류', 2
        
        try:
            runner = session['runner']
            result = runner.detect_components(get_session_data('image_path'))
            
            if result['success']:
                update_session_data('components', result['components'])
                update_session_data('warped_image', result['warped_image_b64'])
                update_session_data('component_image', result.get('component_image_b64'))
                update_session_step(3)
                
                content = create_component_section(
                    result['components'], 
                    result['warped_image_b64'],
                    result.get('component_image_b64')
                )
                return content, PROGRESS_STEPS[3], STEP_MESSAGES[3], 3
            else:
                return dbc.Alert('컴포넌트 검출 실패', color='danger'), PROGRESS_STEPS[2], '오류 발생', 2
                
        except Exception as e:
            return dbc.Alert(f'처리 오류: {str(e)}', color='danger'), PROGRESS_STEPS[2], '오류 발생', 2

    # 3→4단계: 핀 검출 시작
    @app.callback(
        [Output('main-content', 'children', allow_duplicate=True),
         Output('progress-bar', 'value', allow_duplicate=True),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True)],
        Input('confirm-components', 'n_clicks'),
        prevent_initial_call=True
    )
    def start_pin_detection(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return dbc.Alert('세션 만료', color='danger'), 0, '오류', 3
        
        try:
            runner = session['runner']
            result = runner.detect_pins_advanced(
                get_session_data('components'), 
                get_session_data('image_path'),
                get_session_data('warped_image')
            )
            
            if result['success']:
                update_session_data('component_pins', result['component_pins'])
                update_session_data('pin_image', result.get('pin_image_b64'))
                update_session_data('holes', result.get('holes', []))
                update_session_step(4)
                
                content = create_pin_section(
                    result['component_pins'], 
                    get_session_data('warped_image'),
                    result.get('pin_image_b64')
                )
                return content, PROGRESS_STEPS[4], STEP_MESSAGES[4], 4
            else:
                return dbc.Alert('핀 검출 실패', color='danger'), PROGRESS_STEPS[3], '오류 발생', 3
                
        except Exception as e:
            return dbc.Alert(f'핀 검출 오류: {str(e)}', color='danger'), PROGRESS_STEPS[3], '오류 발생', 3

    # 4→5단계: 값 입력으로 이동
    @app.callback(
        [Output('main-content', 'children', allow_duplicate=True),
         Output('progress-bar', 'value', allow_duplicate=True),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True)],
        Input('confirm-pins', 'n_clicks'),
        prevent_initial_call=True
    )
    def proceed_to_values(n_clicks):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return dbc.Alert('세션 만료', color='danger'), 0, '오류', 4
        
        update_session_step(5)
        return (create_value_section(get_session_data('component_pins')), 
                PROGRESS_STEPS[5], STEP_MESSAGES[5], 5)

    # 5→6단계: 전원 설정으로 이동
    @app.callback(
        [Output('main-content', 'children', allow_duplicate=True),
         Output('progress-bar', 'value', allow_duplicate=True),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True)],
        Input('confirm-values', 'n_clicks'),
        [State({'type': 'comp-value', 'idx': ALL}, 'value')],
        prevent_initial_call=True
    )
    def proceed_to_power(n_clicks, values):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return dbc.Alert('세션 만료', color='danger'), 0, '오류', 5
        
        # 컴포넌트 값 업데이트
        component_pins = get_session_data('component_pins')
        value_idx = 0
        for i, comp in enumerate(component_pins):
            if comp['class'] in ['Resistor', 'Capacitor']:
                if value_idx < len(values) and values[value_idx] is not None:
                    comp['value'] = values[value_idx]
                value_idx += 1
        
        update_session_data('component_pins', component_pins)
        update_session_step(6)
        
        return (create_power_section(get_session_data('warped_image')), 
                PROGRESS_STEPS[6], STEP_MESSAGES[6], 6)

    # 6→7단계: 최종 회로 생성
    @app.callback(
        [Output('main-content', 'children', allow_duplicate=True),
         Output('progress-bar', 'value', allow_duplicate=True),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True)],
        Input('start-circuit-generation', 'n_clicks'),
        State('power-voltage', 'value'),
        prevent_initial_call=True
    )
    def start_circuit_generation(n_clicks, voltage):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return dbc.Alert('세션 만료', color='danger'), 0, '오류', 6
        
        try:
            runner = session['runner']
            result = runner.generate_circuit_and_analyze(
                get_session_data('component_pins'),
                voltage or 5.0,
                get_session_data('reference_circuit')
            )
            
            update_session_data('final_result', result)
            update_session_step(7)
            
            return (create_result_section(result), 
                    PROGRESS_STEPS[7], STEP_MESSAGES[7], 7)
            
        except Exception as e:
            return (dbc.Alert(f'회로 생성 오류: {str(e)}', color='danger'), 
                    PROGRESS_STEPS[6], '오류 발생', 6)