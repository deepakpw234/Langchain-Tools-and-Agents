import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_groq import ChatGroq

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler


wiki_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

search = DuckDuckGoSearchRun(name="Search")


# Setting the Title
st.title("Langchain - Chat with Search")


st.sidebar.title("Setting")
groq_api_key = st.sidebar.text_input("Enter the Groq API",type='password')


if "message" not in st.session_state:
    st.session_state["message"]=[
        {"role":"assitant","content":"Hi, I am a chatbot who can search on the web. How can I help you?"}
    ]

for msg in st.session_state.message:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:= st.chat_input(placeholder="What is the machine learning?"):
    st.session_state.message.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(model="llama3-8b-8192",api_key=groq_api_key,streaming=True)
    tools = [search,arxiv,wiki]

    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(st.session_state.message,callbacks=[st_cb])

        st.session_state.message.append({"role":"assitant","content":response})
        st.write(response)