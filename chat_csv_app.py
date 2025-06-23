import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import io
import base64
import re

st.set_page_config(layout='wide')

load_dotenv()

mpl.style.use('dark_background')

# --- Helper: Export chat history as text ---
def export_chat_history(chat_history):
    output = io.StringIO()
    for q, r in chat_history:
        output.write(f'You: {q}\nBot: {r}\n\n')
    return output.getvalue()

# --- Helper: Save matplotlib figure to PNG and return bytes ---
def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf.read()

# --- Helper: Parse chart options from prompt ---
def parse_chart_options(query):
    options = {}
    color_match = re.search(r'color\s*:?\s*(\w+)', query, re.IGNORECASE)
    if color_match:
        options['color'] = color_match.group(1)
    bins_match = re.search(r'bins\s*:?\s*(\d+)', query, re.IGNORECASE)
    if bins_match:
        options['bins'] = int(bins_match.group(1))
    return options

# --- Helper: Validate CSV file ---
def is_valid_csv(file):
    try:
        pd.read_csv(file, nrows=1)
        file.seek(0)
        return True
    except Exception:
        return False

# --- Chatbot logic ---
def chat_with_csv(df, query):
    load_dotenv()
    groq_api_key = os.environ.get('CHATGROQ_API_KEY')
    if not groq_api_key:
        return "Error: CHATGROQ_API_KEY not found. Please create a .env file with your Groq API key."

    chart_types = {
        "bar chart": "bar",
        "line chart": "line",
        "histogram": "hist",
        "scatter plot": "scatter",
        'pie chart': 'pie',
    }
    for chart_text, chart_type in chart_types.items():
        if chart_text in query.lower():
            # Find all columns mentioned in query
            columns_in_query = [col for col in df.columns if col.lower() in query.lower()]
            options = parse_chart_options(query)
            if chart_type == "scatter":
                if len(columns_in_query) >= 2:
                    x, y = columns_in_query[:2]
                    return f"__SHOW_CHART__|scatter|{x}|{y}|{options}"
                else:
                    return "Please specify two columns for scatter plot."
            elif len(columns_in_query) >= 1:
                # Support multi-column for bar/line
                col_str = '|'.join(columns_in_query)
                return f"__SHOW_CHART__|{chart_type}|{col_str}|{options}"
            else:
                return "Could not find a matching column for the chart."

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

# --- Custom CSS for chat, sticky input, and auto-scroll ---
st.markdown(
    """
    <style>
    body, .stApp { background-color: #111 !important; color: #fff !important; }
    .stTextInput, .stTextArea, .stDataFrame, .stButton, .stSelectbox, .stFileUploader {
        background-color: #222 !important; color: #fff !important;
    }
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding-bottom: 20px;
        margin-bottom: 0;
    }
    .chat-bubble-user {
        background: #1a73e8;
        color: #fff;
        padding: 10px 16px;
        border-radius: 16px 16px 4px 16px;
        margin-bottom: 4px;
        max-width: 70%;
        margin-left: auto;
        margin-right: 0;
    }
    .chat-bubble-bot {
        background: #333;
        color: #fff;
        padding: 10px 16px;
        border-radius: 16px 16px 16px 4px;
        margin-bottom: 10px;
        max-width: 70%;
        margin-left: 0;
        margin-right: auto;
    }
    .sticky-bottom {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100vw;
        background: #111;
        z-index: 100;
        padding: 1rem 0.5rem 1rem 0.5rem;
        box-shadow: 0 -2px 8px #00000033;
    }
    .stForm button { display: none !important; }
    </style>
    <script>
    window.onload = function() {
      var chatDiv = window.parent.document.getElementById('chat-container');
      if (chatDiv) { chatDiv.scrollTop = chatDiv.scrollHeight; }
    }
    </script>
    """,
    unsafe_allow_html=True
)

st.title("Multiple-CSV Chatbot powered by LLM")

input_csvs = st.sidebar.file_uploader("Upload your CSV files", type=['csv'], accept_multiple_files=True)

# --- Chat and chart history state ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chart_history' not in st.session_state:
    st.session_state.chart_history = []
if 'reset_chat' not in st.session_state:
    st.session_state.reset_chat = False
if 'loading' not in st.session_state:
    st.session_state.loading = False
if 'show_count' not in st.session_state:
    st.session_state.show_count = 20  # Pagination: show last 20 by default

if st.sidebar.button("Reset Chat"):
    st.session_state.chat_history = []
    st.session_state.reset_chat = True
    st.session_state.chart_history = []
    st.session_state.show_count = 20

# --- Export chat history ---
if st.sidebar.button("Export Chat History"):
    chat_txt = export_chat_history(st.session_state.chat_history)
    st.sidebar.download_button("Download Chat History", chat_txt, file_name="chat_history.txt")

# --- File validation ---
if input_csvs:
    valid_files = [f for f in input_csvs if is_valid_csv(f)]
    if not valid_files:
        st.error("No valid CSV files uploaded.")
        st.stop()
    selected_file = st.selectbox("Select a CSV file", [file.name for file in valid_files])
    selected_index = [file.name for file in valid_files].index(selected_file)
    try:
        data = pd.read_csv(valid_files[selected_index])
        st.info("CSV uploaded successfully")
        st.dataframe(data.head(5), use_container_width=True)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        st.stop()

    st.info("Chat Below")

    # --- Pagination for chat history ---
    total_msgs = len(st.session_state.chat_history)
    show_count = st.session_state.show_count
    start_idx = max(0, total_msgs - show_count)

    # --- Render chat and charts interleaved ---
    chat_html = "<div class='chat-container' id='chat-container'>"
    chart_idx = 0
    for i, (q, r) in enumerate(st.session_state.chat_history[start_idx:]):
        real_idx = i + start_idx
        chat_html += f"<div class='chat-bubble-user'><b>You:</b> {q}</div>"
        chat_html += f"<div class='chat-bubble-bot'><b>Bot:</b> {r}</div>"
        # If a chart was requested at this chat index, show it here
        while chart_idx < len(st.session_state.chart_history) and st.session_state.chart_history[chart_idx]['index'] == real_idx:
            chart_info = st.session_state.chart_history[chart_idx]
            chart_type = chart_info['type']
            cols = chart_info['columns']
            options = chart_info.get('options', {})
            # Debug print for chart type/columns
            # print(f"Rendering chart: type={chart_type}, cols={cols}, options={options}")
            supported_types = ['bar', 'line', 'hist', 'scatter', 'pie']
            if chart_type not in supported_types:
                st.warning(f"Chart type '{chart_type}' is not supported. Please try bar, line, hist, scatter, or pie.")
                chart_idx += 1
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            try:
                color = options.get('color', None)
                bins = options.get('bins', 20)
                if chart_type == 'scatter' and len(cols) == 2:
                    if pd.api.types.is_numeric_dtype(data[cols[0]]) and pd.api.types.is_numeric_dtype(data[cols[1]]):
                        ax.scatter(data[cols[0]], data[cols[1]], c=color)
                        ax.set_xlabel(cols[0])
                        ax.set_ylabel(cols[1])
                        ax.set_title(f"Scatter Plot of {cols[0]} vs {cols[1]}")
                    else:
                        ax.text(0.5, 0.5, 'Scatter plot requires numeric columns.', ha='center', va='center')
                elif chart_type in ['bar', 'line'] and len(cols) >= 1:
                    for col in cols:
                        if pd.api.types.is_numeric_dtype(data[col]):
                            if chart_type == 'bar':
                                ax.bar(data.index, data[col], label=col, color=color)
                            else:
                                ax.plot(data.index, data[col], label=col, color=color)
                        else:
                            vc = data[col].value_counts()
                            ax.bar(vc.index.astype(str), vc.values, label=col, color=color)
                            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
                    ax.set_title(f"{chart_type.title()} Chart of {', '.join(cols)}")
                    ax.set_xlabel("Index")
                    ax.set_ylabel(", ".join(cols))
                    ax.legend()
                elif chart_type == 'hist' and len(cols) == 1:
                    if pd.api.types.is_numeric_dtype(data[cols[0]]):
                        ax.hist(data[cols[0]], bins=bins, color=color)
                        ax.set_title(f"Histogram of {cols[0]}")
                        ax.set_xlabel(cols[0])
                        ax.set_ylabel("Frequency")
                    else:
                        vc = data[cols[0]].value_counts()
                        ax.bar(vc.index.astype(str), vc.values, color=color)
                        ax.set_title(f"Histogram of {cols[0]} (Counts)")
                        ax.set_xlabel(cols[0])
                        ax.set_ylabel("Count")
                        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
                elif chart_type == 'pie' and len(cols) == 1:
                    vc = data[cols[0]].value_counts()
                    n_unique = len(vc)
                    if n_unique < 2:
                        st.warning(f"Pie chart requires at least 2 unique values in column '{cols[0]}'.")
                        chart_idx += 1
                        continue
                    elif n_unique > 10:
                        st.warning(f"Pie chart is only supported for columns with 2-10 unique values. Column '{cols[0]}' has {n_unique} unique values.")
                        chart_idx += 1
                        continue
                    elif pd.api.types.is_numeric_dtype(data[cols[0]]):
                        st.warning(f"Pie chart is not supported for numeric columns. Please select a categorical column.")
                        chart_idx += 1
                        continue
                    else:
                        ax.pie(vc.values, labels=vc.index.astype(str), autopct='%1.1f%%', colors=[color]*len(vc) if color else None)
                        ax.set_title(f"Pie Chart of {cols[0]}")
                else:
                    st.warning(f"Chart type '{chart_type}' is not supported for the selected columns.")
                    chart_idx += 1
                    continue
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center')
            left, center, right = st.columns([2, 7, 2])
            with center:
                st.pyplot(fig)
                # --- Download chart as PNG ---
                png_bytes = fig_to_png_bytes(fig)
                b64 = base64.b64encode(png_bytes).decode()
                href = f'<a href="data:image/png;base64,{b64}" download="chart_{real_idx}.png">Download Chart as PNG</a>'
                st.markdown(href, unsafe_allow_html=True)
            chart_idx += 1
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    # --- Pagination controls ---
    if total_msgs > show_count:
        if st.button('Show More Chat History'):
            st.session_state.show_count += 20
            st.experimental_rerun()

    # --- Sticky chat input at the bottom ---
    st.markdown('<div class="sticky-bottom">', unsafe_allow_html=True)
    with st.form("chat_form", clear_on_submit=True):
        input_text = st.text_input("Enter the query", key="input_text_streamlit")
        send = st.form_submit_button("Send")
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Loading spinner ---
    if st.session_state.loading:
        with st.spinner('Bot is typing...'):
            st.empty()

    if send and input_text.strip():
        st.session_state.chat_history.append((input_text, "Bot is typing..."))
        st.session_state.loading = True
        st.rerun()

    # If last message is 'Bot is typing...', get bot response and update
    if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "Bot is typing...":
        user_query = st.session_state.chat_history[-1][0]
        try:
            response = chat_with_csv(data, user_query)
            if isinstance(response, str) and response.startswith("__SHOW_CHART__"):
                parts = response.split("|")
                chart_type = parts[1]
                columns = []
                options = {}
                # Parse columns and options, filter out empty/option dicts
                for part in parts[2:]:
                    part = part.strip()
                    if part.startswith("{'") or part.startswith('{\"') or (part.startswith('{') and part.endswith('}')):
                        try:
                            options = eval(part)
                        except Exception:
                            options = {}
                    elif part and part != '{}' and part != 'None':
                        columns.append(part)
                # Remove any accidental empty or non-column entries
                columns = [c for c in columns if c and c != '{}' and c != 'None']
                st.session_state.chart_history.append({'index': len(st.session_state.chat_history)-1, 'type': chart_type, 'columns': columns, 'options': options})
                if columns:
                    if chart_type == 'scatter' and len(columns) == 2:
                        bot_msg = f"Scatter plot of `{columns[0]}` vs `{columns[1]}` displayed above."
                    elif len(columns) >= 1:
                        bot_msg = f"{chart_type.title()} of {', '.join(f'`{c}`' for c in columns)} displayed above."
                    else:
                        bot_msg = "Chart displayed above."
                else:
                    bot_msg = "Chart displayed above."
                st.session_state.chat_history[-1] = (user_query, bot_msg)
            else:
                # Render bot response as markdown
                st.session_state.chat_history[-1] = (user_query, response)
        except Exception as e:
            st.session_state.chat_history[-1] = (user_query, f"Error: {e}")
        st.session_state.loading = False
        st.rerun() 