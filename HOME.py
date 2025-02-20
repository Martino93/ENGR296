import streamlit as st


st.set_page_config(
    page_title="Health Data Platform",
    page_icon=":heart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Main Page")
st.sidebar.success("Select a page above.")


########################################
# CSS utilities
########################################
# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# home page for health data platform
def main():
    load_css("config/styles.css")
    st.title("Welcome to the Health Data Platform")
    st.write("This platform is designed to help you manage your health data.")
    st.write(
        "You can upload your health data, view your health data, and analyze your health data."
    )

    st.write("Please select an option from the sidebar to get started.")

    # Adding more sections to the homepage
    st.header("Features")
    st.write(
        """
    - **Upload Data**: Easily upload your health data in various formats.
    - **View Data**: View your uploaded health data in a structured format.
    - **Analyze Data**: Use our tools to analyze your health data and gain insights.
    - **Secure**: Your data is stored securely and privately.
    """
    )

    st.header("How to Use")
    st.write(
        """
    1. Use the sidebar to navigate to different sections.
    2. Upload your health data in the 'Upload Data' section.
    3. View your uploaded data in the 'View Data' section.
    4. Analyze your data in the 'Analyze Data' section.
    """
    )


# Call the main function to display the navigation
if __name__ == "__main__":
    main()
