# Enterprise Chat Assistant for FCD & NCR Analysis – Version 2

This repository contains a conversational AI application designed to assist with querying and analyzing **Field Change Documents (FCD)** and **Non-Conformance Reports (NCR)** using natural language. The assistant interprets user queries and returns precise, structured results or statistics, depending on the request.

The application is built using Python, Pandas, LangChain, and Streamlit, with backend support from an **NVIDIA LLM**, and provides retrieval-augmented logic for querying engineering document workflows and delay records.

---

## About the Application

This project enables engineering teams and project stakeholders to ask detailed questions about project documents — such as how many FCDs are approved, which NCRs are delayed, or the average time taken for document approvals — all through natural language.

The assistant supports a range of functionalities:
- Text-based querying of FCDs and NCRs
- Filters by status, user, delay, project, and document metadata
- Aggregation metrics (count, ratio, average, percentage)
- Simultaneous handling of queries involving both document types
- Safe and structured Pandas code execution

It is optimized for both **web-based interaction** and **local desktop use**.

---

## Core Logic and Functionality

The system works by performing three key steps:

### 1. **Query Classification**
Each user query is first categorized into one of the following:
- `count`: Returns numeric metrics like counts, ratios, averages (query using, "how many", "count", "total number", "number of")
- `table`: Lists documents filtered by project, stage, or delay (query using, "list", "display", "filter", "which", "entries", "table", "pending", "summarize", "overview", "highlights")
- `summary`: Generates table of required field in suggested format in prompt (if document context is loaded)
- `chat`: Fallback for general queries

### 2. **Prompt Construction**
The `build_pandas_prompt()` function generates a detailed prompt that instructs the LLM to return **executable Pandas code only**, based on:
- Query intent
- Targeted DataFrame (FCD or NCR)
- Filtering conditions (status, delay, user, project)
- Column selection rules

You can modify the logic, rules, or formatting of the generated code by editing the `build_pandas_prompt()` function if needed.

### 3. **Code Execution**
The generated Pandas code is executed within a controlled environment using a secure `safe_execute_pandas_code()` function. It validates, parses, and executes the code while safeguarding the data from unintended mutations.

---

##  Data Preparation & Updates

The app works with two Parquet files:
- `fcd_data.parquet`
- `ncr_data.parquet`

These files are refreshed daily using a Jupyter notebook.

### Daily Data Update Workflow:

1. Open the file `data_paraquet_former.ipynb` from this repository.
2. Run all cells to extract and process the latest data.
3. This will generate two updated files:
   - `C:\Temp\Py_Files_sid\fcd_data.parquet`
   - `C:\Temp\Py_Files_sid\ncr_data.parquet`
4. Copy these files into your GitHub repository to update the deployed application with the latest data.

---

##  How to Launch the Application

### Web Interface
You can access the hosted web version here:  
🔗 [https://ai-model-jkvtiyxmikxpmdldhixscr.streamlit.app/](https://ai-model-jkvtiyxmikxpmdldhixscr.streamlit.app/)  
> This link launches the Streamlit-based Version 2 of the Retrieval-Augmented Chat Assistant.

---

### Run Locally (Desktop Version)

To run the app locally on your desktop:

1. Navigate to the following path in your system: C:\Temp\Py_Files_sid
2. Open Command Prompt in that directory and execute:
```bash
python -m streamlit run streamlit_test_2.py
Replace streamlit_test_2.py with your specific script name if different.

This will open a local Streamlit session with full functionality.

| Component      | Technology                                 |
| -------------- | ------------------------------------------ |
| Language Model | NVIDIA Chat LLM (via LangChain)            |
| Embeddings     | HuggingFace / SentenceTransformers         |
| Vector Store   | FAISS                                      |
| UI Framework   | Streamlit                                  |
| Code Execution | Python `exec()` (with `ast` safety checks) |
| Data Format    | `.parquet` (efficient tabular storage)     |

Repository Structure

├── data_paraquet_former.ipynb        # Prepares daily FCD & NCR data
├── fcd_data.parquet                  # Field Change Document data
├── ncr_data.parquet                  # Non-Conformance Report data
├── streamlit_v2.py                   # Main app file (Streamlit UI)
├── llm_logic.py                      # For llm calling
├── README.md                         # Project documentation

🚧 Future Enhancements
The following upgrades are under consideration for future versions:\
🎙Voice-based input using NVIDIA Riva Speech-to-Text
📊 Dashboard visualizations using Plotly or Power BI integration
🧾 Document-level semantic summarization using OCR and NLP
🔐 Role-based authentication for enterprise deployment
💬 Multi-turn memory threading for more natural conversations

# Contact & Contributions
This application is developed by Siddharth Singh,
MBA (Data Science & AI) Candidate – IIT Mandi.

Feel free to fork, contribute, or raise issues as needed.
For best results, ensure the parquet files are up to date before launching the app. Use the instructions above to refresh the data.

