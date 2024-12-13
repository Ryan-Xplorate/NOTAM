# footer for streamlit app
import streamlit as st

def footer():
    st.markdown(
        """
        <style>
        footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: white;
            padding: 10px 0;
            z-index: 999;
            border-top: 1px solid #eee;
            margin: 0;
        }
        footer p {
            color: #888;
            font-size: 10px;
            text-align: center;
            margin: 20px;
            padding: 0 20px;
            margin-left: 40%;  /* Offset text to avoid sidebar */
            max-width: 60%;  /* Ensure text doesn't stretch too far */
        }
        .main .block-container {
            padding-bottom: 5rem;
            margin-left: 0;
            margin-right: 0;
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
