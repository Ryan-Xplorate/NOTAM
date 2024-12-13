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
        }
        footer p {
            color: #888;
            font-size: 10px;
            text-align: center;
            margin: 20px auto;
            padding: 0 20px;
            max-width: 800px;  /* Set a max-width for better readability */
            /* Calculate the width accounting for sidebar */
            width: calc(100% - 300px);  /* 300px is default Streamlit sidebar width */
            /* Push content to the right to account for sidebar */
            margin-left: calc(21rem + 20px);  /* 21rem matches Streamlit's sidebar width + padding */
        }
        /* Ensure main content doesn't get hidden behind footer */
        .main .block-container {
            padding-bottom: 6rem !important;  /* Increased padding to prevent content overlap */
            max-width: 100%;
        }
        /* Adjust for different screen sizes */
        @media (max-width: 768px) {
            footer p {
                margin-left: 20px;
                width: calc(100% - 40px);
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
