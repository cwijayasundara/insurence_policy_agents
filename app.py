from insurence_agents import InsurencePolicyAgent, agent_rag_tool, ProgressEvent
import streamlit as st
import asyncio
import nest_asyncio

nest_asyncio.apply()

st.title("Insurence Policy Agent")
st.write("This agent can answer questions about AXA insurence policies and refund details.")

with st.sidebar:
    st.image("images/agents.png", width=600)

query = st.text_input("Enter your query here:")

async def get_answer(query):
    agent = InsurencePolicyAgent(timeout=600, verbose=True)
    handler = agent.run(
        query=query,
        tools=[agent_rag_tool],
    )

    # async for ev in handler.stream_events():
    #     if isinstance(ev, ProgressEvent):
    #         st.write(ev.progress)
    final_result = await handler
    st.write("------- Final Answer ----------\n", final_result)

if st.button("Get Answer"):
    if query:
        asyncio.run(get_answer(query))
    else:
        st.warning("Please enter a query first.")