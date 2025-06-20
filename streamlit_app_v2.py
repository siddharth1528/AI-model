import streamlit as st
st.set_page_config(page_title="L&T Enterprise Assistant", layout="wide")
import pandas as pd
import re
import io
import contextlib
import requests
import ast
import torch

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from llm_logic import NvidiaChatLLM
from IPython.core.display import HTML as ipyHTML
from IPython.display import display
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceBgeEmbeddings

#embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu") 
from langchain.memory import ConversationBufferMemory

# -------------------- LOAD DATA --------------------

def load_data():
    df_ncr = pd.read_parquet("ncr_data.parquet")
    df_fcd = pd.read_parquet("fcd_data.parquet")
    return df_ncr, df_fcd

df_NCR, df_FCD = load_data()

# -------------------- VECTORSTORE SETUP --------------------

def initialize_vectorstore():
    combined_df = pd.concat([df_NCR, df_FCD], ignore_index=True)
    
    # Ensure 'DOC_Description' is a string and not NaN
    if "DOC_Description" in combined_df.columns:
        combined_df["DOC_Description"] = combined_df["DOC_Description"].fillna("").astype(str)
    
    loader = DataFrameLoader(combined_df, page_content_column="DOC_Description")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    texts = splitter.split_documents(documents)
    
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(texts, embedding)

vectorstore = initialize_vectorstore()

# -------------------- lsit columns request --------------------
def extract_requested_columns(query, available_columns):
    query_lower = query.lower()
    extracted = []
    for col in available_columns:
        col_parts = col.lower().replace("_", " ").split()
        if any(part in query_lower for part in col_parts):
            extracted.append(col)
    return list(set(extracted))

# -------------------- SAFE PANDAS EXECUTION --------------------
def safe_execute_pandas_code(code: str, df_NCR=None, df_FCD=None, user_query: str = ""):
    if not isinstance(code, str):
        return f"❌ Error: LLM returned a non-string response: {type(code)}"

    match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
    code_to_run = match.group(1).strip() if match else code.strip()

    code_to_run = re.sub(r"[^\x20-\x7E\n\t]", "", code_to_run)
    code_to_run = re.sub(r"\b0+(\d+)", r"\1", code_to_run)

    try:
        ast.parse(code_to_run)
    except SyntaxError as e:
        return f"❌ Syntax error: {e}"

    for df in [df_NCR, df_FCD]:
        if df is not None and "Ongoing_Delay_Days" in df.columns:
            df["Ongoing_Delay_Days"] = pd.to_numeric(df["Ongoing_Delay_Days"], errors="coerce")

    for df in [df_NCR, df_FCD]:
            if df is not None:
                if "ORDER_DESCRIPTION" in df.columns and "Order_Description" not in df.columns:
                    df["Order_Description"] = df["ORDER_DESCRIPTION"]
                elif "Order_Description" in df.columns and "ORDER_DESCRIPTION" not in df.columns:
                    df["ORDER_DESCRIPTION"] = df["Order_Description"]

    local_vars = {
        "df_NCR": df_NCR,
        "df_FCD": df_FCD,
        "pd": pd,
        "display": display,
        "HTML": ipyHTML,
        "filtered_df": None,
    }

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code_to_run, {}, local_vars)

        # ✅ First, check if anything was printed
        printed_output = output.getvalue().strip()
        if printed_output and "<IPython.core.display.HTML object>" not in printed_output:
            return printed_output

        # ✅ Handle HTML display with dynamic column selection
        html_candidates = ["filtered_df", "result", "output_df"]
        for var_name in html_candidates:
            val = local_vars.get(var_name)
            if isinstance(val, pd.DataFrame) and not val.empty:
                requested_cols = extract_requested_columns(user_query, list(val.columns))
                if requested_cols:
                    val = val[requested_cols]
                st.dataframe(val)
                return "✅ Filtered results displayed."

        # ✅ Search for any HTML-renderable object
        for val in local_vars.values():
            if hasattr(val, "_repr_html_"):
                st.markdown(val._repr_html_(), unsafe_allow_html=True)
                return "✅ Custom HTML table rendered."

        return "✅ Code executed successfully (no output returned)."

    except Exception as e:
        return f"❌ Execution error: {e}"

    
# -------------------- LLM API calling --------------------
llm = NvidiaChatLLM(api_key="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")
memory = ConversationBufferMemory(return_messages=True)
chat_chain = ConversationChain(llm=llm, memory=memory) 

# ------------------ CLASSIFIER & Routing ------------------

def classify_query(query: str):
    query_lower = query.lower()

    if any(k in query_lower for k in ["how many", "count", "total number", "number of"]):
        return "count"
    elif any(k in query_lower for k in ["list", "show", "display", "filter", "which", "entries", "table", "pending"]):
        return "table"
    elif any(k in query_lower for k in ["what is this document", "describe this", "what does this document talk about",
                                        "summarize", "overall theme", "key issues", "highlights"]):
        return "summary"
    else:
        return "chat"
    

def retrieve_context(query: str, k: int = 10):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])

# ------------------ PANDAS PROMPT BUILDER ------------------

def build_pandas_prompt(query, df_NCR, df_FCD, chat_history=None):
    return f"""
You are a highly skilled Python and Pandas expert.

You are working with two Pandas DataFrames:
- df_NCR: Contains Non-Conformance Reports (NCRs) with columns: {list(df_NCR.columns)}
- df_FCD: Contains Field Change Documents (FCDs) with columns: {list(df_FCD.columns)}

Your job:
- Understand the user's query and generate **valid, executable Pandas code only**.
- Never include markdown, triple backticks, comments, explanations, or extra text.
- The final line **must always** be a `print(...)` or `display(...)` statement that shows the result clearly.
- Display result using: display(ipyHTML(filtered_df.to_html(index=False)))

General Guidelines:
- NEVER define or simulate data; use only df_NCR and df_FCD.
- Use `from IPython.display import display` and `from IPython.core.display import HTML as ipyHTML` when displaying tables.
- If the result is a filtered DataFrame, use:  
  `display(ipyHTML(filtered_df.to_html(index=False)))`
- If returning scalar values (counts, unique items, etc.), use:  
  `print(...)`
- If using Streamlit, use: st.dataframe(filtered_df)
- Final result **must** be assigned to a variable named `filtered_df`.
- Display result using: display(ipyHTML(filtered_df.to_html(index=False)))
- Always sanitize strings before comparison:
  - Use: `df["col_name"].fillna('').str.upper().str.contains("SOME_TEXT")`

📌 Project Reference Handling:
If the user query mentions a **project name** (e.g., "Dolvi project", "Debari plant", "Barmer job"), interpret it as a keyword search in the column `Order_Description`.

- Use:
  `(df['ORDER_DESCRIPTION'].fillna('').str.upper().str.contains('KEYWORD') if 'ORDER_DESCRIPTION' in df.columns else df['Order_Description'].fillna('').str.upper().str.contains('KEYWORD'))`
  where `KEYWORD` is the uppercase form of the project name mentioned in the query.

- Do NOT attempt to match projects in other fields like `Sub contractor`.

🔎 Examples:
- "FCDs of Debari project" → filter df_FCD where `Order_Description` contains "DEBARI"
- "how many NCRs are for Mangalore plant" → filter df_NCR where `Order_Description` contains "MANGALORE"


Filtering Rules:
- For string comparisons, always use `.str.upper()`
- Normalize status like this (before filtering if needed):
    - "WIP", "IN PROGRESS", "PENDING", "NOT STARTED", "ON HOLD", "OPEN" → "WORK IN PROGRESS"
    - "DONE", "FINISHED", "COMPLETE", "COMPLETED" → "APPROVED"
- Use `.str.upper()` on both sides for these status comparisons.
- For delay filtering, always convert `'Ongoing_Delay_Days'` using:  
  `pd.to_numeric(..., errors='coerce')`  
  and filter after dropping nulls using `.notna()`
- For dates, use:  
  `pd.to_datetime(..., errors='coerce')`

If the user says "summarize" or wants to "show a table":
- Use the following columns for filtered tables:
    - For **NCRs**:  
      `['DOC_NO', 'DOC_DESCRIPTION', 'ORDER_DESCRIPTION' ,'DOC_STATUS', 'Discipline', 'Approval_Stage_user', 'Current_Workflow_stage', 'Workflow_stage_users', 'Ongoing_Delay_Days']`
    - For **FCDs**:  
      `['DOC_Number', 'DOC_Description', 'DOC_Status', 'Discipline', 'Sub contractor', 'Order_Description', 'Current_Workflow_Stage', 'Workflow_Stage_Users', 'Ongoing_Delay_Days']`
- In such cases, assign filtered results to `filtered_df` and then use:  
  `display(ipyHTML(filtered_df.to_html(index=False)))`

Final Output Rules:
- If the query is about **counts** or scalar values (like "how many", "count", "number of"):
    ➤ Use `print(...)` only. Do NOT return a table or DataFrame.
- If the query is to **show** rows (like "list", "show", "display", "entries"):
    ➤ Assign result to `filtered_df` and display using:
        `display(ipyHTML(filtered_df.to_html(index=False)))`
-Always assign final DataFrame output to filtered_df before displaying.


Semantic continuity:
- If vague words like "such", "those", "these", etc. are used, infer their meaning from this chat history:
{chat_history or '[no prior context]'}

Now return ONLY valid Python Pandas code — no explanations or markdown — that answers this user query:
{query}
"""

# ------------------ ANSWER MAIN ENTRY ------------------

def answer_query(query: str):
    intent = classify_query(query)
    query_lower = query.lower()

    if intent in ["count", "table"]:
        chat_history = memory.buffer.strip() if memory.buffer else "[no prior context]"
        prompt = build_pandas_prompt(query, df_NCR, df_FCD, chat_history=chat_history)
        response = llm.invoke(prompt)
        code = response.content if hasattr(response, "content") else str(response)
        print("\n🔍 Generated Code:\n", code)

        memory.chat_memory.add_user_message(query)
        memory.chat_memory.add_ai_message(code)

        return safe_execute_pandas_code(code, df_NCR=df_NCR, df_FCD=df_FCD)

    elif intent == "summary":
        context = retrieve_context(query)
        if not context.strip():
            return "❌ No relevant context found in the document."

        # Direct table summary (if user is asking for specific fields)
        df_to_use = df_NCR if "ncr" in query_lower else df_FCD
        summary_columns = (
            ["DOC_NO", "DOC_DESCRIPTION", "DOC_STATUS", "Discipline", "Approval_Stage_user", "Current_Workflow_stage", "Workflow_stage_users", "Ongoing_Delay_Days"]
            if "ncr" in query_lower else
            ["DOC_Number", "DOC_Description", "DOC_Status", "Discipline", "Sub contractor", "Technical Requirments", "Current_Workflow_Stage", "Workflow_Stage_Users", "Ongoing_Delay_Days"]
        )
        df_filtered = df_to_use.copy()

        if "delay" in query_lower or "ongoing" in query_lower:
            df_filtered["Ongoing_Delay_Days"] = pd.to_numeric(df_filtered["Ongoing_Delay_Days"], errors="coerce")
            df_filtered = df_filtered[df_filtered["Ongoing_Delay_Days"].notna()]
            match = re.search(r"(\d+)\s*day", query_lower)
            delay_threshold = int(match.group(1)) if match else 60
            df_filtered = df_filtered[df_filtered["Ongoing_Delay_Days"] > delay_threshold]

        from IPython.display import HTML
        selected_cols = [col for col in summary_columns if col in df_filtered.columns]
        table_df = df_filtered[selected_cols]

        if not table_df.empty:
            display(HTML(table_df.to_html(index=False)))
            return  # Avoid fallback summary if table worked
        else:
            print("No records match the summary filter.")

        # Fallback to LLM summarization
        prompt = f"""
You are a data analyst AI that answers strictly based on structured engineering project records.
Only use the provided document excerpt — do not invent or guess if the answer is not explicitly present.

Document Excerpt:
-----------------
{context}
-----------------

Now answer this question based only on the document above:
Q: {query}
"""
        return chat_chain.run(prompt)

    else:
        return chat_chain.run(query)


# -------------------- STREAMLIT UI --------------------

st.markdown("""
    <style>
    body {
        background-color: #F5EEDC;
        font-family: "Segoe UI", sans-serif;
    }
    .main {
        background-color: #F5EEDC;
    }
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding: 0 2rem;
    }
    .header-text h1 {
        color: #131D4F;
        font-size: 2.5rem;
        margin: 0;
    }
    .header-subtitle {
        color: #254D70;
        font-size: 1.1rem;
        margin-top: 0.3rem;
    }
    .chat-container {
        background-color: #EFE4D2;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0px 6px 16px rgba(0,0,0,0.07);
        margin: 1rem auto;
        max-width: 85%;
    }
    .user-msg {
        background-color: #DDA853;
        color: #131D4F;
        padding: 1rem;
        border-radius: 0.6rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .bot-msg {
        background-color: #254D70;
        color: white;
        padding: 1rem;
        border-radius: 0.6rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .stTextInput > div > input {
        background-color: #F5EEDC;
        padding: 0.8rem;
        border-radius: 0.5rem;
        font-size: 1rem;
        border: 1px solid #954C2E;
    }
    .stButton > button {
        background-color: #954C2E;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.6rem 1.4rem;
        font-size: 1rem;
        font-weight: 600;
    }
    .stButton > button:hover {
        background-color: #7a3e26;
    }
    .robot-image {
        width: 120px;
        margin-right: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ Header with Logo ------------------

col1, col2 = st.columns([1, 6])

with col1:
    st.image("https://www.larsentoubro.com/media/30891/ltgrouplogo.jpg", width=200)

with col2:
    st.markdown('''
        <div class="header-text">
            <h1>🤖 Quality Chat Assistant</h1>
            <div class="header-subtitle">
                Ask about NCRs & FCDs using natural language.
            </div>
        </div>
    ''', unsafe_allow_html=True)

# ------------------ Chat Input Section ------------------

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

input_col, img_col = st.columns([6, 1])

with input_col:
    with st.form("chat_form"):
        user_query = st.text_input("💬 Ask your question:", placeholder="e.g. List all FCDs in validation stage", key="user_input")
        submitted = st.form_submit_button("🔍 Ask")

with img_col:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Output Response Section ------------------

if submitted and user_query:
    with st.spinner("Got it.. Processing Your Query!"):
        result = answer_query(user_query)

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.markdown(f'<div class="user-msg">👤 <strong>You:</strong><br>{user_query}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="bot-msg">🤖 <strong>Assistant:</strong><br>{result}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# if user_query:
#     with st.spinner("Processing your query..."):
#         result = answer_query(user_query)
#         st.success("Answer Ready")
#         st.write(result)
