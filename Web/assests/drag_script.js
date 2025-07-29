// assets/drag_script.js - 이미지 드래그 영역 선택 스크립트

(function() {
    // 전역 상태 변수
    let dragModeActive = false;
    let isDragging = false;
    let startX, startY, imgEl, imgRect;

    // 드래그 오버레이 스타일 업데이트 함수
    function updateOverlay(style) {
        const overlay = document.getElementById('drag-overlay');
        if (overlay) {
            Object.assign(overlay.style, style);
        }
    }

    // 드래그 종료 및 Dash로 좌표 전송 함수
    function endDrag(e) {
        if (!isDragging) return;
        isDragging = false;
        document.body.style.userSelect = 'auto';

        const currentX = (e.clientX - imgRect.left) / imgRect.width * imgEl.naturalWidth;
        const currentY = (e.clientY - imgRect.top) / imgRect.height * imgEl.naturalHeight;

        const x1 = Math.round(Math.min(startX, currentX));
        const y1 = Math.round(Math.min(startY, currentY));
        const x2 = Math.round(Math.max(startX, currentX));
        const y2 = Math.round(Math.max(startY, currentY));

        // 최소 5x5 픽셀 이상 드래그한 경우에만 좌표 전송
        if (x2 - x1 > 5 && y2 - y1 > 5) {
            window.dash_clientside.setProps('drag-selection-store', { data: { x1, y1, x2, y2 } });
        } else {
            updateOverlay({ display: 'none' });
        }
    }

    // 드래그 기능 초기화 및 이벤트 리스너 설정 함수
    function initializeDrag() {
        imgEl = document.getElementById('component-image');
        if (!imgEl) return;

        // 드래그 모드 상태 확인
        const storeEl = document.getElementById('drag-mode-active');
        if (storeEl && storeEl.textContent) {
            try {
                dragModeActive = JSON.parse(storeEl.textContent);
            } catch (e) {
                dragModeActive = false;
            }
        }
        imgEl.style.cursor = dragModeActive ? 'crosshair' : 'default';

        // mousedown: 드래그 시작
        imgEl.onmousedown = (e) => {
            if (!dragModeActive) return;
            e.preventDefault();
            isDragging = true;
            imgRect = imgEl.getBoundingClientRect();
            document.body.style.userSelect = 'none'; // 드래그 중 텍스트 선택 방지

            startX = (e.clientX - imgRect.left) / imgRect.width * imgEl.naturalWidth;
            startY = (e.clientY - imgRect.top) / imgRect.height * imgEl.naturalHeight;

            updateOverlay({
                left: `${(startX / imgEl.naturalWidth) * 100}%`,
                top: `${(startY / imgEl.naturalHeight) * 100}%`,
                width: '0px',
                height: '0px',
                display: 'block',
                border: '2px dashed #0d6efd'
            });
        };

        // mousemove: 드래그 중 오버레이 크기 변경
        document.body.onmousemove = (e) => {
            if (!isDragging || !dragModeActive) return;
            const currentX = (e.clientX - imgRect.left) / imgRect.width * imgEl.naturalWidth;
            const currentY = (e.clientY - imgRect.top) / imgRect.height * imgEl.naturalHeight;

            const left = Math.min(startX, currentX);
            const top = Math.min(startY, currentY);
            const width = Math.abs(currentX - startX);
            const height = Math.abs(currentY - startY);

            updateOverlay({
                left: `${(left / imgEl.naturalWidth) * 100}%`,
                top: `${(top / imgEl.naturalHeight) * 100}%`,
                width: `${(width / imgEl.naturalWidth) * 100}%`,
                height: `${(height / imgEl.naturalHeight) * 100}%`
            });
        };

        // mouseup: 드래그 종료
        document.body.onmouseup = endDrag;
    }

    // Dash 앱의 동적 UI 변경을 감지하여 초기화 함수 재호출
    const observer = new MutationObserver((mutations) => {
        const componentImageExists = document.getElementById('component-image');
        const needsInitialization = componentImageExists && !componentImageExists.onmousedown;

        if (needsInitialization) {
            initializeDrag();
        }

        for (const mutation of mutations) {
            if (mutation.target.id === 'drag-mode-active' && mutation.characterData) {
                 initializeDrag();
                 break;
            }
        }
    });

    observer.observe(document.body, { childList: true, subtree: true, characterData: true });
})();