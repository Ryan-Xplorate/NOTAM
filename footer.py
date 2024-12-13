import streamlit as st

def footer():
    st.markdown(
        """
        <style>
        footer {
            position: fixed;
            bottom: 0;
            right: 0;
            width: calc(100% - 350px);  /* Adjust width to account for sidebar */
            margin-left: 250px;  /* Match Streamlit's sidebar width */
            background-color: white;
            padding: 5px 0;
            z-index: 999;
            border-top: 1px solid #eee;
        }
        footer p {
            color: #888;
            font-size: 8px;
            text-align: center;
            margin: 20px auto;  /* Center the paragraph */
            padding: 0 40px;
            max-width: 1200px;
        }
        .main .block-container {
            padding-bottom: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <footer>
            <p>
            Legal Disclaimer: The information provided on this site is for general informational purposes only and should not be solely relied upon. Please verify the information independently and confirm with the NOTAM Data Quality Requirements available through Airservices.
            </p>
        </footer>
        """,
        unsafe_allow_html=True,
    )
