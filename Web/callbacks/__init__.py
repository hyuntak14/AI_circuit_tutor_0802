# callbacks/__init__.py
from .auth_callbacks import register_auth_callbacks
from .step_callbacks import register_step_callbacks
from .navigation_callbacks import register_navigation_callbacks
from .chat_callbacks import register_chat_callbacks

def register_all_callbacks(app):
    """모든 콜백 등록"""
    register_auth_callbacks(app)
    register_step_callbacks(app)
    register_navigation_callbacks(app)
    register_chat_callbacks(app)

__all__ = ['register_all_callbacks']