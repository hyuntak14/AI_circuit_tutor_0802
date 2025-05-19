import streamlit as st
from streamlit_app_part1 import page_1_upload, page_2_corner_adjust, page_3_transformed
from streamlit_app_part2 import page_4_component_edit, page_5_hole_detection, page_6_pin_detection
from streamlit_app_part3 import page_7_value_input, page_8_manual_pin_adjustment, page_9_power_selection
from streamlit_app_part4 import page_10_circuit_generation, page_11_error_checking, page_12_summary

TOTAL_PAGES = 12

def main():
    st.set_page_config(
            page_title="Breadboard to Diagram", 
            page_icon="🔌",
            layout="wide"
        )
    
    # 사이드바에 페이지 네비게이션
    with st.sidebar:
        st.title("🔌 Navigation")
        page_names = [
            "1. Upload Image",
            "2. Adjust Corners", 
            "3. View Transform",
            "4. Edit Components",
            "5. Detect Holes",
            "6. Detect Pins",
            "7. Enter Values",
            "8. Adjust Pins",
            "9. Select Power",
            "10. Generate Circuit",
            "11. Check Errors",
            "12. Summary"
        ]
        
        current_page = st.session_state.page
        for i, name in enumerate(page_names, 1):
            if i == current_page:
                st.markdown(f"**➤ {name}**")
            elif i < current_page:
                st.markdown(f"✅ {name}")
            else:
                st.markdown(f"⏸️ {name}")
        
        st.markdown("---")
        st.markdown("### 📋 Progress")
        progress = (current_page - 1) / (TOTAL_PAGES - 1)
        st.progress(progress)
        st.write(f"{progress*100:.0f}% Complete")

    
    page = st.session_state.page

    if page == 1:
        page_1_upload()
    elif page == 2:
        page_2_corner_adjust()
    elif page == 3:
        page_3_transformed()
    elif page == 4:
        page_4_component_edit()
    elif page == 5:
        page_5_hole_detection()
    elif page == 6:
        page_6_pin_detection()
    elif page == 7:
        page_7_value_input()
    elif page == 8:
        page_8_manual_pin_adjustment()
    elif page == 9:
        page_9_power_selection()
    elif page == 10:
        page_10_circuit_generation()
    elif page == 11:
        page_11_error_checking()
    elif page == 12:
        page_12_summary()
    else:
        st.error('Invalid page number. Restarting...')
        st.session_state.page = 1
        st.rerun()

if __name__ == '__main__':
    main()
