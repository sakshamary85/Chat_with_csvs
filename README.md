# Multiple‑CSV Chatbot powered by LLM

A powerful Streamlit web app that lets you chat with your CSV files using natural language! Upload multiple CSVs, ask questions, and get instant answers, insights, and even visualizations (charts/plots) — all powered by an LLM (Llama3 via Groq API).

---

## 🚀 Features
- **Multiple CSV Upload**: Upload and switch between multiple CSV files effortlessly.
- **Natural Language Chat**: Interact with your data using plain English (e.g., “Show me a bar chart of species count”).
- **Automatic Charting**: Generate bar, line, histogram, scatter, and pie charts based on your queries.
- **Downloadable Charts**: Save any generated chart directly as a PNG.
- **Chat History**: Track and export your conversation history.
- **Modern UI**: Clean, dark-themed, responsive interface with a sticky chat input.
- **Error Handling**: Intelligent feedback for invalid queries or unsupported chart types.

---

## 🧪 Demo  
![Demo Screenshot](https://github.com/sakshamary85/Chat_with_csvs/blob/main/Screenshot%202025-06-23%20142303.png)
![Demo Screenshot](https://github.com/sakshamary85/Chat_with_csvs/blob/main/Screenshot%202025-06-23%20142332.png)
![Demo Screenshot](https://github.com/sakshamary85/Chat_with_csvs/blob/main/Screenshot%202025-06-23%20142401.png)

---

## 🛠️ Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/sakshamary85/chat-with-csv.git
   cd chat-with-csv
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   
3. **Configure Groq API Key**
   ***Create a .env file in the project root directory and add your Groq API key like this:***
   ```bash
   GROQ_API_KEY=your_api_key_here

4. **Run the App**
   ```bash
   streamlit run chat_csv_app.py
