import streamlit as st

def footer():
    st.markdown(
        """
        <style>
        footer {
            position: fixed;
            bottom: 0;
            right: 0;
            width: 100%;
            background-color: white;
            padding: 10px 0;
            z-index: 999;
            border-top: 1px solid #eee;
        }
        footer p {
            color: #888;
            font-size: 10px;
            text-align: center;
            margin: 20px;
            padding: 0 20px;
            margin-left: 400px;  /* Large enough margin to clear the sidebar */
            margin-right: 20px;  /* Prevent text from touching the right edge */
        }
        .main .block-container {
            padding-bottom: 5rem;
        }
        /* Make footer responsive */
        @media screen and (max-width: 1200px) {
            footer p {
                margin-left: 300px;
            }
        }
        @media screen and (max-width: 992px) {
            footer p {
                margin-left: 250px;
            }
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
