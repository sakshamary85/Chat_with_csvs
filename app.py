import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import os

st.set_page_config(layout='wide')

load_dotenv()

def chat_with_csv(df, query):
    load_dotenv()
    
    groq_api_key = os.environ.get('CHATGROQ_API_KEY')
    
    if not groq_api_key:
        return "Error: CHATGROQ_API_KEY not found. Please create a .env file with your Groq API key."

    # --- Chart detection and plotting ---
    chart_types = {
        "bar chart": "bar",
        "line chart": "line",
        "histogram": "hist",
        "scatter plot": "scatter",
        'pie chart': 'pie',
    }
    for chart_text, chart_type in chart_types.items():
        if chart_text in query.lower():
            # Find column(s) in query
            columns_in_query = [col for col in df.columns if col.lower() in query.lower()]
            if chart_type == "scatter":
                if len(columns_in_query) >= 2:
                    x, y = columns_in_query[:2]
                    fig, ax = plt.subplots()
                    ax.scatter(df[x], df[y])
                    ax.set_xlabel(x)
                    ax.set_ylabel(y)
                    ax.set_title(f"Scatter Plot of {x} vs {y}")
                    st.pyplot(fig)
                    return f"Scatter plot of `{x}` vs `{y}` displayed above."
                else:
                    return "Please specify two columns for scatter plot."
            elif len(columns_in_query) >= 1:
                col = columns_in_query[0]
                fig, ax = plt.subplots()
                if chart_type == "bar":
                    ax.bar(df.index, df[col])
                elif chart_type == "line":
                    ax.plot(df.index, df[col])
                elif chart_type == "hist":
                    ax.hist(df[col], bins=20)
                ax.set_title(f"{chart_text.title()} of {col}")
                ax.set_xlabel("Index")
                ax.set_ylabel(col)
                st.pyplot(fig)
                return f"{chart_text.title()} of `{col}` displayed above."
            else:
                return "Could not find a matching column for the chart."

    # --- LLM fallback for other queries ---
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-70b-8192",
        temperature=0.2
    )
    df_info = f"""
    DataFrame Info:
    Shape: {df.shape}
    Columns: {list(df.columns)}
    First few rows:
    {df.head().to_string()}

    User Query: {query}

    Please analyze this data and provide insights based on the query.
    """
    response = llm.invoke(df_info)
    return response.content
    
st.markdown(
    """
    <style>
    body, .stApp { background-color: #111 !important; color: #fff !important; }
    .stTextInput, .stTextArea, .stDataFrame, .stButton, .stSelectbox, .stFileUploader {
        background-color: #222 !important; color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Multiple-CSV Chatbot powered by LLM (Large language model)")

input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# --- Chat history state ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'reset_chat' not in st.session_state:
    st.session_state.reset_chat = False

if st.sidebar.button("Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.reset_chat = True
    
if input_csvs:
    selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
    selected_index = [file.name for file in input_csvs].index(selected_file)

    st.info("CSV uploaded successfully")
    data = pd.read_csv(input_csvs[selected_index])
    st.dataframe(data.head(5), use_container_width=True)

    st.info("Chat Below")

    # --- Show chat history first (WhatsApp style) ---
    for q, r in st.session_state.chat_history:
        st.markdown(f"<div style='background:#222;padding:8px;border-radius:6px;margin-bottom:2px'><b>You:</b> {q}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='background:#333;padding:8px;border-radius:6px;margin-bottom:10px'><b>Bot:</b> {r}</div>", unsafe_allow_html=True)

    # --- Input box at the bottom ---
    with st.form("chat_form", clear_on_submit=True):
        input_text = st.text_input("Enter the query", key="input_text")
        send = st.form_submit_button("Send")

    if send and input_text.strip():
        response = chat_with_csv(data, input_text)
        st.session_state.chat_history.append((input_text, response))
        # st.experimental_rerun()
# if input_csvs:

    # selected_file = st.selectbox("Select a CSV file", [file.name for file in input_csvs])
    # selected_index = [file.name for file in input_csvs].index(selected_file)

    # st.info("CSV uploaded successfully")
    # data = pd.read_csv(input_csvs[selected_index])
    # st.dataframe(data.head(5), use_container_width=True)

    # st.info("Chat Below")
    # input_text = st.text_area("Enter the query")
    
    

    # if input_text:
    #     if st.button('Chat With CSV'):
    #         st.info('Your Query: ' + input_text)
    #         response = chat_with_csv(data, input_text)
    #         st.success(response)

