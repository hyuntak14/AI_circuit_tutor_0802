# callbacks/step_callbacks.py - 단계별 처리 콜백 (완전한 개선된 버전)
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
    update_session_data, get_session_data,
    add_component, update_component, delete_component, get_components
)
from components import *

def register_step_callbacks(app):
    """단계별 처리 콜백 등록 (완전한 개선된 버전)"""
    
    # ===== 1단계: 이미지 업로드 처리 =====
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

    # ===== 1→2단계: 기준 회로 선택으로 이동 =====
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

    # ===== 새 컴포넌트 추가 모달 열기/닫기 =====
    @app.callback(
        Output('add-component-modal', 'is_open'),
        [Input('open-add-component-modal', 'n_clicks'),
         Input('cancel-add-component', 'n_clicks')],
        State('add-component-modal', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_add_component_modal(open_clicks, cancel_clicks, is_open):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update
        
        return not is_open

    # ===== 새 컴포넌트 추가 처리 =====
    @app.callback(
        [Output('component-list', 'children'),
         Output('add-component-modal', 'is_open', allow_duplicate=True),
         Output('add-component-status', 'children')],
        Input('add-new-component', 'n_clicks'),
        [State('new-component-type', 'value'),
         State('new-x1', 'value'),
         State('new-y1', 'value'),
         State('new-x2', 'value'),
         State('new-y2', 'value'),
         State('new-confidence', 'value')],
        prevent_initial_call=True
    )
    def add_new_component_handler(n_clicks, comp_type, x1, y1, x2, y2, confidence):
        if not n_clicks:
            return dash.no_update, dash.no_update, dash.no_update
        
        # 입력값 검증
        if None in [x1, y1, x2, y2] or x1 >= x2 or y1 >= y2:
            return (
                dash.no_update, 
                True,  # 모달 열어둠
                dbc.Alert('좌표 값을 올바르게 입력하세요. (x1 < x2, y1 < y2)', color='danger')
            )
        
        # 새 컴포넌트 추가
        box = [int(x1), int(y1), int(x2), int(y2)]
        success = add_component(comp_type, box, confidence or 0.9)
        
        if success:
            # 업데이트된 컴포넌트 리스트 반환
            updated_components = get_components()
            from utils import format_component_list_enhanced
            return (
                format_component_list_enhanced(updated_components),
                False,  # 모달 닫기
                dbc.Alert(f'{comp_type} 컴포넌트가 추가되었습니다.', color='success')
            )
        else:
            return (
                dash.no_update,
                True,  # 모달 열어둠
                dbc.Alert('컴포넌트 추가에 실패했습니다.', color='danger')
            )

    # ===== 컴포넌트 삭제 처리 =====
    @app.callback(
        Output('component-list', 'children', allow_duplicate=True),
        Input({'type': 'del-comp', 'idx': ALL}, 'n_clicks'),
        prevent_initial_call=True
    )
    def delete_component_callback(n_clicks_list):
        ctx = callback_context
        if not ctx.triggered or not any(n_clicks_list):
            return dash.no_update
        
        # 클릭된 버튼 찾기
        button_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])
        component_idx = button_id['idx']
        
        # 컴포넌트 삭제
        success = delete_component(component_idx)
        
        if success:
            updated_components = get_components()
            from utils import format_component_list_enhanced
            return format_component_list_enhanced(updated_components)
        
        return dash.no_update

    # ===== 컴포넌트 편집 모달 열기/닫기 =====
    @app.callback(
        [Output('component-edit-modal', 'is_open'),
         Output('edit-component-type', 'value'),
         Output('edit-component-confidence', 'children'),
         Output('edit-x1', 'value'),
         Output('edit-y1', 'value'),
         Output('edit-x2', 'value'),
         Output('edit-y2', 'value'),
         Output('components-store', 'data')],  # 편집할 컴포넌트 인덱스 저장
        [Input({'type': 'edit-comp', 'idx': ALL}, 'n_clicks'),
         Input('cancel-component-edit', 'n_clicks')],
        State('component-edit-modal', 'is_open'),
        prevent_initial_call=True
    )
    def toggle_edit_component_modal(edit_clicks, cancel_clicks, is_open):
        ctx = callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
        trigger_id = ctx.triggered[0]['prop_id']
        
        # 취소 버튼 클릭
        if 'cancel-component-edit' in trigger_id:
            return False, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, {'editing_idx': -1}
        
        # 편집 버튼 클릭
        if any(edit_clicks):
            button_id = json.loads(trigger_id.split('.')[0])
            component_idx = button_id['idx']
            
            # 해당 컴포넌트 정보 가져오기
            components = get_components()
            if 0 <= component_idx < len(components):
                comp_type, confidence, box = components[component_idx]
                return True, comp_type, f'{confidence:.1%}', box[0], box[1], box[2], box[3], {'editing_idx': component_idx}
        
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    # ===== 컴포넌트 편집 저장 =====
    @app.callback(
        [Output('component-list', 'children', allow_duplicate=True),
         Output('component-edit-modal', 'is_open', allow_duplicate=True)],
        Input('save-component-edit', 'n_clicks'),
        [State('edit-component-type', 'value'),
         State('edit-x1', 'value'),
         State('edit-y1', 'value'),
         State('edit-x2', 'value'),
         State('edit-y2', 'value'),
         State('components-store', 'data')],
        prevent_initial_call=True
    )
    def save_component_edit(n_clicks, comp_type, x1, y1, x2, y2, store_data):
        if not n_clicks:
            return dash.no_update, dash.no_update
        
        # 편집 중인 컴포넌트 인덱스 가져오기
        editing_idx = store_data.get('editing_idx', -1) if store_data else -1
        
        if editing_idx == -1:
            return dash.no_update, True  # 편집 인덱스가 없으면 모달 열어둠
        
        # 입력값 검증
        if None not in [x1, y1, x2, y2] and x1 < x2 and y1 < y2:
            box = [int(x1), int(y1), int(x2), int(y2)]
            success = update_component(editing_idx, comp_type, box)
            
            if success:
                updated_components = get_components()
                from utils import format_component_list_enhanced
                return format_component_list_enhanced(updated_components), False
        
        return dash.no_update, True  # 실패시 모달 열어둠

    # ===== 3→4단계: 핀 검출 시작 =====
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
            # 업데이트된 컴포넌트 리스트 사용
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

    # ===== 4→5단계: 값 입력으로 이동 =====
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

    # ===== 5→6단계: 전원 설정으로 이동 =====
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

    # ===== 6→7단계: 최종 회로 생성 =====
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