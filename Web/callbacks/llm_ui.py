# llm_ui.py (크기 조정 버전)

import tkinter as tk
from tkinter import scrolledtext, ttk
import threading

class LLMChatWindow:
    """
    LLM 피드백 및 대화를 위한 GUI 창
    """
    def __init__(self, parent, llm_manager, initial_analysis: str):
        self.llm_manager = llm_manager
        
        # UI 창 설정 (크기 증가)
        self.window = tk.Toplevel(parent)
        self.window.title("🤖 AI 회로 분석 도우미")
        self.window.geometry("1200x800") # 창 크기 증가 (700x600 -> 800x700)
        
        # 스타일
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", padding=6, relief="flat", background="#007bff", foreground="white")
        style.map("TButton", background=[('active', '#0056b3')])

        # 메인 프레임
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 대화 내용 표시 영역 (글꼴 크기 증가)
        self.chat_area = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, state='disabled', font=("Arial", 22)) # 글꼴 크기 증가 (11 -> 12)
        self.chat_area.pack(pady=5, padx=5, fill=tk.BOTH, expand=True)
        
        # 태그 설정 (글꼴 크기 조정)
        self.chat_area.tag_config('user', foreground='blue', font=("Arial", 12, "bold")) # 글꼴 크기 증가
        self.chat_area.tag_config('ai', foreground='black')
        self.chat_area.tag_config('error', foreground='red')
        self.chat_area.tag_config('system', foreground='green', font=("Arial", 11, "italic")) # 시스템 메시지는 약간 작게

        # 입력 프레임
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(pady=5, padx=5, fill=tk.X)

        # 사용자 입력창 (글꼴 크기 증가)
        self.user_input = ttk.Entry(input_frame, font=("Arial", 22)) # 글꼴 크기 증가
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5)
        self.user_input.bind("<Return>", self.send_message)

        # 전송 버튼
        self.send_button = ttk.Button(input_frame, text="전송", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT, padx=(5, 0))

        # 초기 분석 결과 표시
        self.display_message(initial_analysis, 'ai')
        self.display_message("\n추가 질문이 있으시면 아래에 입력해주세요. (종료: 창 닫기)", 'system')
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.user_input.focus_set()

    def send_message(self, event=None):
        """사용자 메시지를 전송하고 AI 응답을 요청합니다."""
        query = self.user_input.get().strip()
        if not query:
            return

        self.display_message(f"You: {query}\n", 'user')
        self.user_input.delete(0, tk.END)
        
        # UI가 멈추지 않도록 스레드에서 AI 응답 요청
        thread = threading.Thread(target=self.get_ai_response, args=(query,))
        thread.start()

    def get_ai_response(self, query: str):
        """LLM Manager를 통해 AI 응답을 가져옵니다."""
        try:
            # RAG 검색
            results = self.llm_manager.rag_system.search_similar_documents(query, top_k=3)
            context = self.llm_manager.rag_system.create_context(results)
            
            # 프롬프트 생성 (항상 후속 턴으로 간주)
            prompt = self.llm_manager.create_rag_prompt(
                query, context, False, self.llm_manager.practice_circuit_topic
            )

            # AI 응답 생성
            response = self.llm_manager.model.generate_content(
                prompt,
                generation_config=self.llm_manager.generation_config
            )
            
            # UI 업데이트는 메인 스레드에서 실행
            self.window.after(0, self.display_message, f"AI: {response.text}\n", 'ai')

        except Exception as e:
            self.window.after(0, self.display_message, f"오류 발생: {e}\n", 'error')

    def display_message(self, message: str, tag: str):
        """채팅창에 메시지를 표시합니다."""
        self.chat_area.config(state='normal')
        self.chat_area.insert(tk.END, message + "\n", tag)
        self.chat_area.config(state='disabled')
        self.chat_area.see(tk.END) # 자동 스크롤

    def on_closing(self):
        """창을 닫을 때 호출됩니다."""
        self.window.destroy()