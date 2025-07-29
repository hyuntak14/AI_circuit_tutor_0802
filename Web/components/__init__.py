# components/__init__.py - 채팅 인터페이스 추가
from .login import login_layout
from .upload import create_main_layout, create_upload_section, create_upload_result_section
from .reference import create_reference_section, create_reference_result_section
from .component_detect import create_component_section, create_component_edit_modal, create_add_component_modal
from .pin_setup import create_pin_section
from .value_input import create_value_section
from .power_setup import create_power_section
from .results import create_result_section
from .chat_interface import create_chat_interface, create_initial_chat_layout, create_final_chat_interface

__all__ = [
    'login_layout',
    'create_main_layout',
    'create_upload_section',
    'create_upload_result_section',
    'create_reference_section',
    'create_reference_result_section',
    'create_component_section',
    'create_component_edit_modal',
    'create_add_component_modal',
    'create_pin_section',
    'create_value_section',
    'create_power_section',
    'create_result_section',
    'create_chat_interface',
    'create_initial_chat_layout',
    'create_final_chat_interface'
]