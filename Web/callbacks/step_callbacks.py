# callbacks/step_callbacks.py - 채팅 UI 통합 버전
import dash
from dash import Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import base64
import tempfile
import os
import json
from datetime import datetime

from config import PROGRESS_STEPS, STEP_MESSAGES
from state_manager import (
    get_current_session, update_session_step, 
    update_session_data, get_session_data,
    add_component, update_component, delete_component, get_components
)
from components import *
from components.chat_interface import create_chat_interface, create_final_chat_interface

def register_step_callbacks(app):
    """단계별 처리 콜백 등록 (채팅 UI 통합 버전)"""
    
    # ===== 1단계: 채팅 UI에서 이미지 업로드 처리 =====
    @app.callback(
        [Output('main-content', 'children'),
         Output('progress-bar', 'value'),
         Output('step-info', 'children'),
         Output('current-step', 'data'),
         Output('chat-messages-store', 'data')],
        Input('chat-image-upload', 'contents'),
        [State('chat-image-upload', 'filename'),
         State('chat-messages-store', 'data')],
        prevent_initial_call=True
    )
    def handle_chat_image_upload(contents, filename, messages):
        """채팅 UI에서 이미지 업로드 처리"""
        if not contents:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        try:
            # 이미지 디코딩 및 저장
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            session = get_current_session()
            if not session:
                # 에러 메시지를 채팅에 추가
                messages.append({
                    'type': 'ai',
                    'content': '세션이 만료되었습니다. 다시 로그인해주세요.',
                    'time': datetime.now().strftime('%H:%M')
                })
                return create_chat_interface(messages, show_upload=True), 0, '오류', 0, messages
            
            # 임시 파일 저장
            temp_dir = tempfile.mkdtemp()
            image_path = os.path.join(temp_dir, filename)
            with open(image_path, 'wb') as f:
                f.write(decoded)
            
            # 세션에 데이터 저장
            update_session_data('image_path', image_path)
            update_session_data('filename', filename)
            update_session_data('chat_messages', messages)  # 채팅 메시지도 저장
            update_session_step(2)
            
            # 채팅 메시지 업데이트
            messages.append({
                'type': 'user',
                'content': f'이미지 업로드: {filename}',
                'time': datetime.now().strftime('%H:%M')
            })
            messages.append({
                'type': 'ai',
                'content': '이미지가 성공적으로 업로드되었습니다. 이제 기준 회로를 선택해주세요.',
                'time': datetime.now().strftime('%H:%M')
            })
            
            # 2단계 UI로 전환
            return create_reference_section(), PROGRESS_STEPS[2], STEP_MESSAGES[2], 2, messages
            
        except Exception as e:
            messages.append({
                'type': 'ai',
                'content': f'업로드 실패: {str(e)}',
                'time': datetime.now().strftime('%H:%M')
            })
            return create_chat_interface(messages, show_upload=True), 0, '오류 발생', 0, messages

    # ===== 2단계: 기준 회로 선택 처리 =====
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

    # ===== 2→3단계: 컴포넌트 검출 시작 =====
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

    # ===== 6→7단계: 최종 회로 생성 후 채팅 UI로 전환 =====
    @app.callback(
        [Output('main-content', 'children', allow_duplicate=True),
         Output('progress-bar', 'value', allow_duplicate=True),
         Output('step-info', 'children', allow_duplicate=True),
         Output('current-step', 'data', allow_duplicate=True),
         Output('rag-context-store', 'data')],
        Input('start-circuit-generation', 'n_clicks'),
        State('power-voltage', 'value'),
        prevent_initial_call=True
    )
    def start_circuit_generation(n_clicks, voltage):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        session = get_current_session()
        if not session:
            return dbc.Alert('세션 만료', color='danger'), 0, '오류', 6, {}
        
        try:
            runner = session['runner']
            result = runner.generate_circuit_and_analyze(
                get_session_data('component_pins'),
                voltage or 5.0,
                get_session_data('reference_circuit')
            )
            
            update_session_data('final_result', result)
            update_session_step(7)
            
            # RAG 컨텍스트 준비
            rag_context = {
                'circuit_data': result.get('circuit_data', {}),
                'component_pins': get_session_data('component_pins'),
                'reference_circuit': get_session_data('reference_circuit'),
                'voltage': voltage or 5.0,
                'analysis_text': result.get('analysis_text', '')
            }
            
            # 채팅 UI로 전환하면서 분석 결과 표시
            chat_ui = create_final_chat_interface(result.get('analysis_text', ''))
            
            return chat_ui, PROGRESS_STEPS[7], STEP_MESSAGES[7], 7, rag_context
            
        except Exception as e:
            return (dbc.Alert(f'회로 생성 오류: {str(e)}', color='danger'), 
                    PROGRESS_STEPS[6], '오류 발생', 6, {})

    # ===== 나머지 콜백들은 기존과 동일 =====
    # (컴포넌트 편집, 삭제, 핀 검출 등의 콜백은 그대로 유지)
    
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
            current_components = get_components()
            result = runner.detect_pins_advanced(
                current_components, 
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
        if component_pins:
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
    

    # 채팅 초기화 콜백
    @app.callback(
        Output('chat-messages-store', 'data', allow_duplicate=True),
        Input('page-content', 'children'),
        State('login-state', 'data'),
        prevent_initial_call=True
    )
    def initialize_chat_messages(page_content, login_state):
        """로그인 후 채팅 메시지 초기화"""
        if login_state and login_state.get('logged_in'):
            # 로그인 직후라면 초기 메시지 설정
            session = get_current_session()
            if session and not get_session_data('chat_messages'):
                initial_messages = [{
                    'type': 'ai',
                    'content': '안녕하세요! 회로 분석 AI입니다. 브레드보드 이미지를 업로드해주세요.',
                    'time': datetime.now().strftime('%H:%M')
                }]
                update_session_data('chat_messages', initial_messages)
                return initial_messages
        return dash.no_update