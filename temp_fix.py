"""
This is a temporary fix file to reimplement the YouTube button section.
This will be used to restore the original functionality.
"""

# YouTube link section with proper HTML button
def youtube_button_section(video, video_id, i, j, selected_user, user_emotion):
    # Add a direct HTML link for YouTube
    st.markdown(f'''
    <div style="text-align: center;">
        <a href="https://www.youtube.com/watch?v={video_id}" target="_blank">
            <button style="
                background-color: #FF0000;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 16px;
                font-size: 16px;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                margin: 10px 0;">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="white" style="margin-right: 8px;">
                    <path d="M10,16.5V7.5L16,12M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2Z" />
                </svg>
                Watch on YouTube
            </button>
        </a>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("---")
