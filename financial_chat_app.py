import streamlit as st
import requests
import json
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "api_key_sectors" not in st.session_state:
        st.session_state.api_key_sectors = ""
    if "api_key_groq" not in st.session_state:
        st.session_state.api_key_groq = ""

def retrieve_from_endpoint(url: str) -> dict:
    headers = {"Authorization": st.session_state.api_key_sectors}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.HTTPError as err:
        st.error(f"API Error: {err}")
        return None
    return json.dumps(data)

@tool
def get_company_overview(stock: str) -> str:
    """Get the company overview for a given stock."""
    url = f"https://api.sectors.app/v1/company/report/{stock}/?sections=overview"
    return retrieve_from_endpoint(url)

@tool
def get_top_companies_by_tx_volume(start_date: str, end_date: str, top_n: int = 5) -> str:
    """Get the top companies by transaction volume."""
    url = f"https://api.sectors.app/v1/most-traded/?start={start_date}&end={end_date}&n_stock={top_n}"
    return retrieve_from_endpoint(url)

@tool
def get_daily_tx(stock: str, start_date: str, end_date: str) -> str:
    """Get the daily transaction for a given stock."""
    url = f"https://api.sectors.app/v1/daily/{stock}/?start={start_date}&end={end_date}"
    return retrieve_from_endpoint(url)

def setup_agent():
    tools = [
        get_company_overview,
        get_top_companies_by_tx_volume,
        get_daily_tx
    ]

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Answer the following queries, being as factual and analytical 
            as you can. If you need the start and end dates but they are not 
            explicitly provided, infer from the query. Whenever you return a 
            list of names, return also the corresponding values for each name. 
            If the volume was about a single day, the start and end 
            parameter should be the same."""
        ),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    llm = ChatGroq(
        temperature=0,
        model_name="llama3-groq-70b-8192-tool-use-preview",
        groq_api_key=st.session_state.api_key_groq,
    )

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt,
    )

    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def add_message(role, content, intermediate_steps=None):
    st.session_state.messages.append({
        "role": role,
        "content": content,
        "steps": intermediate_steps,
        "timestamp": datetime.now()
    })

def main():
    st.title("Stock Market Analysis Chat")
    
    # Initialize session state
    init_session_state()
    
    # Sidebar for API keys
    with st.sidebar:
        st.header("API Configuration")
        st.session_state.api_key_sectors = st.text_input("Sectors API Key", type="password")
        st.session_state.api_key_groq = st.text_input("Groq API Key", type="password")
    
    # Check if API keys are provided
    if not st.session_state.api_key_sectors or not st.session_state.api_key_groq:
        st.warning("Please enter your API keys in the sidebar to continue.")
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Display intermediate steps if they exist
            if message["steps"]:
                with st.expander("Show intermediate steps"):
                    for step in message["steps"]:
                        st.write("🔍 Tool Called:", step[0])
                        st.write("📊 Result:", step[1])
    
    # Chat input
    if prompt := st.chat_input("Ask about stock market data..."):
        # Add user message
        add_message("user", prompt)
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process with agent
        try:
            agent_executor = setup_agent()
            with st.spinner("Thinking..."):
                response = agent_executor.invoke(
                    {"input": prompt},
                    {"return_intermediate_steps": True}
                )
                
                # Extract response and steps
                answer = response["output"]
                steps = response["intermediate_steps"]
                
                # Add assistant response with intermediate steps
                add_message("assistant", answer, steps)
                
                # Display response and steps
                with st.chat_message("assistant"):
                    st.write(answer)
                    with st.expander("Show intermediate steps"):
                        for step in steps:
                            st.write("🔍 Tool Called:", step[0])
                            st.write("📊 Result:", step[1])
                            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()