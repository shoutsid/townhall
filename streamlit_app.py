import streamlit as st
import pandas as pd
from app.agents.buddy import Buddy


def main():
    st.title("Townhall Streamlit Interface")

    # Initialize session state variables
    if "buddy" not in st.session_state:
        st.session_state.buddy = Buddy()
        st.session_state.response = None

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", ["Home", "Agents", "Database"])

    if page == "Home":
        st.write("Welcome to the Townhall Streamlit interface!")

    elif page == "Agents":
        st.subheader("Agent Interaction")

        # Agent selection
        agent_selection = st.selectbox(
            "Choose an agent:", ["Buddy", "Planner"])

        # Message input and interaction with the selected agent
        user_input = st.text_input(
            f"Enter a message to send to the {agent_selection} agent:"
        )

        if st.button("Send"):
            if agent_selection == "Buddy":
                st.session_state.response = st.session_state.buddy.start(
                    user_input)
            elif agent_selection == "Planner":
                st.session_state.response = (
                    "Planner agent interaction not yet implemented."
                )

        if st.session_state.response:
            st.write(f"{agent_selection} Response: {st.session_state.response}")


if __name__ == "__main__":
    main()
