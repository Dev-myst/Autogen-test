import asyncio
import streamlit as st
from agent import AgentTeam

def main() -> None:
    st.set_page_config(page_title="Autogen Literature Review", page_icon="📚")
    st.title("📚 Autogen Literature Review Assistant")
    st.write("Enter a research topic, and the agent team will find papers and generate a literature review.")


    if "agent_team" not in st.session_state:
    
        loop = asyncio.new_event_loop()
        st.session_state["event_loop"] = loop
        asyncio.set_event_loop(loop)
        st.session_state["agent_team"] = AgentTeam()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("e.g., search for multi-agent system on customer service, find 3 papers"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("The AI team is collaborating... This may take a moment."):
             
                loop = st.session_state["event_loop"]
    
                response = loop.run_until_complete(
                    st.session_state["agent_team"].run_chat(prompt)
                )
                st.markdown(response)
        
        st.session_state["messages"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
