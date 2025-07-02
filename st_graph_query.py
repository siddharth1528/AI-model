import streamlit as st
st.set_page_config(page_title="Retrieval Augmented Logic Engine-2 for NCR & FCD", layout="wide")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


import pandas as pd
import re
import io
import contextlib
import requests
import ast
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from langchain.tools import tool
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
from difflib import get_close_matches

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
    query_words = query.lower().replace("_", " ").split()
    matched = []
    for col in available_columns:
        col_tokens = col.lower().replace("_", " ").split()
        # Match based on exact token overlap or fuzzy match
        if any(token in query_words for token in col_tokens) or get_close_matches(col.lower(), query_words, cutoff=0.7):
            matched.append(col)
    return list(set(matched))


# -------------------- SAFE PANDAS EXECUTION --------------------
def safe_execute_pandas_code(code: str, df_NCR=None, df_FCD=None, user_query: str = "", intent: str = "table"):
    if not isinstance(code, str):
        return f"❌ Error: LLM returned a non-string response: {type(code)}"

    match = re.search(r"```(?:python)?\n(.*?)```", code, re.DOTALL)
    code_to_run = match.group(1).strip() if match else code.strip()

    # st.code(code_to_run, language="python")
    # st.info("✅ [DEBUG] Generated code block shown above.")

    code_to_run = re.sub(r"[^\x20-\x7E\n\t]", "", code_to_run)
    code_to_run = re.sub(r"\b0+(\d+)", r"\1", code_to_run)

    # Redirect print() of DataFrame-like objects to display()
    code_to_run = re.sub(
        r"print\s*\(\s*(filtered_df|result|output_df)\s*\)",
        r"display(ipyHTML(\1.to_html(index=False)))",
        code_to_run,
        flags=re.IGNORECASE
    )

    # Replace plt.show() with Streamlit rendering
    # code_to_run = re.sub(r"\bplt\.show\(\)", "st.pyplot(plt.gcf())", code_to_run)

    try:
        ast.parse(code_to_run)
    except SyntaxError as e:
        return f"❌ Syntax error: {e}"

    for df in [df_NCR, df_FCD]:
        if isinstance(df, pd.DataFrame):
            if "Ongoing_Delay_Days" in df.columns:
                df["Ongoing_Delay_Days"] = pd.to_numeric(df["Ongoing_Delay_Days"], errors="coerce")
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
        "ipyHTML": ipyHTML,
        "plt": plt,
        "sns": sns,
        "px": px,
        "st": st,
        "filtered_df": None,
    }

    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(code_to_run, {}, local_vars)

        printed_output = output.getvalue().strip()

        # Try to get filtered dataframe output
        val = None
        for name in ["filtered_df", "result", "output_df"]:
            candidate = local_vars.get(name)
            if isinstance(candidate, pd.DataFrame):
                val = candidate
                break

        # Handle Plot Intent
        if intent == "plot":
            fig = plt.gcf()
            if fig.get_axes():
                st.pyplot(fig)
                plt.clf()
                return "📊 Plot displayed."
            elif isinstance(val, pd.DataFrame) and not val.empty:
                try:
                    if all(col in val.columns for col in ['Discipline', 'Count']):
                        plt.figure(figsize=(10, 6))
                        sns.barplot(data=val, x='Discipline', y='Count')
                        plt.title("Auto-Plot: Discipline vs Count")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(plt.gcf())
                        plt.clf()
                        return "📊 Auto-generated bar chart."
                except Exception as e:
                    print("[DEBUG] Auto-plot fallback failed:", e)

        # Handle Count output
        if intent == "count" and printed_output:
            return printed_output

        # Handle Table or Plot fallback
        if isinstance(val, pd.DataFrame) and not val.empty:
            if "ncr" in user_query.lower() and "fcd" not in user_query.lower():
                st.subheader("📄 NCR Records")
            elif "fcd" in user_query.lower() and "ncr" not in user_query.lower():
                st.subheader("📄 FCD Records")

            requested_cols = extract_requested_columns(user_query, list(val.columns))
            if requested_cols:
                val = val[requested_cols]

            st.dataframe(val)
            return "✅ Filtered results displayed."

        # 🪪 Empty or fallback
        if isinstance(val, pd.DataFrame) and val.empty:
            st.warning("⚠️ No matching records found.")
            return "⚠️ No results."

        # Fallback: Custom HTML rendering
        for v in local_vars.values():
            if hasattr(v, "_repr_html_"):
                st.markdown(v._repr_html_(), unsafe_allow_html=True)
                return "✅ Custom HTML table rendered."

        return "✅ Code executed successfully (no output returned)."

    except Exception as e:
        return f"❌ Execution error: {e}"
    
# -------------------- LLM API calling --------------------
llm = NvidiaChatLLM(api_key="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")
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

memory = st.session_state.memory
chat_chain = ConversationChain(llm=llm, memory=memory)

# ------------------ CLASSIFIER & Routing ------------------

def classify_query(query: str):
    query_lower = query.lower()

    if any(k in query_lower for k in ["how many", "count", "total number", "number of"]):
        return "count"
    elif any(k in query_lower for k in ["list", "display", "filter", "which", "entries", "table", "pending", "summarize", "overview", "highlights"]):
        return "table"
    elif any(k in query_lower for k in ["what is this document", "describe this", "what does this document talk about",
                                        "overall theme", "key issues"]):
        return "summary"
    
    elif any(k in query_lower for k in ["plot", "graph", "chart", "visualize", "draw", "show"]):
        return "plot"
    elif any(k in query_lower for k in ["dashboard", "summary view", "overview panel"]):
        return "dashboard"
    else:
        return "chat"
    

def retrieve_context(query: str, k: int = 10):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join([doc.page_content for doc in docs])

# ------------------ PANDAS PROMPT BUILDER ------------------
def build_pandas_prompt(query, df_NCR, df_FCD, chat_history=None, mode="default", source=None):
    """
    Build a contextualized prompt for an AI model to generate Pandas code,
    compatible with Streamlit, LangChain, or notebook environments.

    Parameters:
    - query (str): The user query.
    - df_NCR (pd.DataFrame or None): Non-Conformance Reports DataFrame.
    - df_FCD (pd.DataFrame or None): Field Change Documents DataFrame.
    - chat_history (str or None): Optional conversation history.
    - mode (str): "default", "streamlit", or "langchain"
    - source (str or None): Explicitly specify "ncr", "fcd", or "both" to override inference.

    Returns:
    - str: The full prompt string.
    """

    query_lower = query.lower()

    if source:
        source = source.lower()
        if source == "both":
            primary_df = "both"
        elif source == "fcd":
            primary_df = "df_FCD"
        elif source == "ncr":
            primary_df = "df_NCR"
        else:
            primary_df = "[invalid source specified]"
    else:
        if "fcd" in query_lower and "ncr" in query_lower:
            primary_df = "both"
        elif "fcd" in query_lower:
            primary_df = "df_FCD"
        elif "ncr" in query_lower:
            primary_df = "df_NCR"
        else:
            primary_df = "[no explicit dataframe mentioned]"

    # Fallbacks for None DataFrames
    ncr_cols = list(df_NCR.columns) if df_NCR is not None else '[not provided]'
    fcd_cols = list(df_FCD.columns) if df_FCD is not None else '[not provided]'

    
    # Prompt construction
    prompt = f"""
You are a highly skilled Python and Pandas expert.

You are working with two Pandas DataFrames:
- df_NCR: Non-Conformance Reports (NCRs) with columns: {ncr_cols}
- df_FCD: Field Change Documents (FCDs) with columns: {fcd_cols}

Domain Context:
- FCD stands for **Field Change Document** — it is a project engineering document.
- NCR stands for **Non-Conformance Report** — it logs deviations from project quality standards.
- Delay is tracked using the column: `Ongoing_Delay_Days` in both DataFrames.
- Approval is tracked using the column: `DOC_Status`, with normalized values like "APPROVED" and "WORK IN PROGRESS".

Objective:
Generate only executable Pandas code that satisfies the user query.
Do not include markdown, explanations, comments, or text outside the code.

Use only **{primary_df}** for this query.

Instructions:
- Use the correct DataFrame: `{primary_df}`
- Apply filters based on the query (e.g., filter status = 'WIP', or stage = 'WIP')
- Select only the columns asked for, if mentioned (e.g., project name, doc no, status, stage)
- Do not return the full dataframe unless explicitly asked
- Do not use display(), just return the final pandas code
- If query includes both FCD and NCR, assign each to a separate variable.
  Example: `df_ncr_out = ...`, `df_fcd_out = ...`
- If asking to "list", "show", or "display", only return the relevant columns.


Filtering Rules:
- For string filtering, always use `.str.upper()`
- Normalize statuses:
    - ["WIP", "IN PROGRESS", "PENDING", "NOT STARTED", "ON HOLD", "OPEN"] → "WORK IN PROGRESS"
    - ["DONE", "FINISHED", "COMPLETE", "COMPLETED"] → "APPROVED"
- Compare statuses using `.str.upper()` on both sides.
- For project filters:
    df['Order_Description'].fillna('').str.upper().str.contains('KEYWORD')
- Delay filtering:
    pd.to_numeric(df['Ongoing_Delay_Days'], errors='coerce').notna()
- Date conversion:
    pd.to_datetime(df['SomeDateColumn'], errors='coerce')

Column Selection Requirement:
If the query is to **list**, **display**, or **summarize** FCDs or NCRs:
- You MUST restrict the displayed DataFrame to the following columns:
  - For **df_NCR**:
    ['DOC_NO', 'DOC_DESCRIPTION', 'DOC_STATUS', 'ORDER_DESCRIPTION', 'Discipline',
     'Approval_Stage_user', 'Current_Workflow_stage', 'Workflow_stage_users', 'Ongoing_Delay_Days']
  - For **df_FCD**:
    ['DOC_Number', 'DOC_Description', 'DOC_Status', 'Discipline', 'Sub contractor',
     'Order_Description', 'Current_Workflow_Stage', 'Workflow_Stage_Users', 'Ongoing_Delay_Days']

📌 Final Output Instructions for DataFrame Queries:
- The **last line** of your code **must** assign the filtered result to a variable named `filtered_df`.
- Then render it using: `display(ipyHTML(filtered_df.to_html(index=False)))`
- ❌ Do NOT use `print(...)` for DataFrame display.
- ❌ Never use `print(filtered_df)` — it will not render correctly.
- ✅ Correct format:
    filtered_df = df_FCD[...filtered...]
    display(ipyHTML(filtered_df.to_html(index=False)))
-❌ NEVER use print(filtered_df)
-❌ NEVER use print(...) for DataFrames
-✅ ALWAYS use display(ipyHTML(...)) for rendering tables

Final Output Rules:
- If the query is about **counts** or scalar values (like "how many", "count", "number of"):
    ➤ Use `print(...)` only. Do NOT return a table or DataFrame.

- If the query requires **scalar results** — such as:
    • count / number of entries  
    • ratio or proportion  
    • average / mean / median  
    • percentage calculations  
    • total / sum  
    • min / max / std deviation  
    • any single numerical/statistical/algebraic result
        ➤ You must output using: `print(...)`  
        ➤ Do **not** assign intermediate variables unless necessary for clarity of computation.  
        ➤ Do **not** display any DataFrame or table unless explicitly asked in the query.  
        ➤ Do NOT show steps or explain anything.
        ➤ Do NOT include variable names, comments, or markdown — just print the final result. 
        ➤ No steps, no commentary, no variable explanations.
        ➤ Do NOT break the answer into steps.
        ➤ Just return pure valid Pandas code that directly calculates and prints the result. 
    
Approval Time Calculation:
- If the query refers to the "number of days taken for approval":
    ➤ Assume you must calculate the difference between `Approval_Date` and `Submission_Date`
    ➤ First, convert both columns to datetime:
        df['Approval_Date'] = pd.to_datetime(df['Approval_Date'], errors='coerce')
        df['Submission_Date'] = pd.to_datetime(df['Submission_Date'], errors='coerce')
    ➤ Then compute:
        df['Approval_Days'] = (df['Approval_Date'] - df['Submission_Date']).dt.days
- Use `Approval_Days` only if it is present; otherwise, compute it as shown.
- For averages or ratios based on approval time, apply `df['Approval_Days'].mean()` etc. **only after** filtering `APPROVED` entries using `DOC_Status`.

⛔ You MUST NOT use `print(...)` or `print(filtered_df)` for table display.

✅ INSTEAD:
- Always assign the result to a variable named `filtered_df`
- Then display the DataFrame using:
    display(ipyHTML(filtered_df.to_html(index=False)))
✅ Example:
filtered_df = df_FCD[...]  # your filtering logic here
display(ipyHTML(filtered_df.to_html(index=False)))

- Always assign final DataFrame output to filtered_df before displaying.
- You MUST use: `display(ipyHTML(filtered_df.to_html(index=False)))` to render the final table
- NEVER use: `print(filtered_df)` or `print(...)` for DataFrame display
- NEVER show the full DataFrame unless explicitly asked

🔧 Delay and Status Filtering:
- Always normalize `DOC_Status` before filtering:
  - WIP, IN PROGRESS, PENDING, NOT STARTED, ON HOLD, OPEN → WORK IN PROGRESS
  - DONE, FINISHED, COMPLETE, COMPLETED → APPROVED
- Always convert `Ongoing_Delay_Days` using:
  df['Ongoing_Delay_Days'] = pd.to_numeric(df['Ongoing_Delay_Days'], errors='coerce')
- Filter only after conversion.

If query mentions only FCD → Use only df_FCD  
If query mentions only NCR → Use only df_NCR  
If both → Filter and display both separately
Never use df_NCR when the query only refers to FCD or vice-versa

If both NCR and FCD are being filtered, assign results separately:
- For FCDs, use `df_fcd_out = ...`
- For NCRs, use `df_ncr_out = ...`

Multi-Source Handling:
- If query includes both NCR and FCD:
    - Assign separately:
        df_ncr_out = ...
        df_fcd_out = ...
    - Display both independently.
- Apply the **same filters** (e.g., status = "WIP") **independently** to both DataFrames.
- Subset each DataFrame to required columns before displaying.
- Do not return full dataframe unless explicitly asked.

Contextual Understanding:
- Interpret vague references ("such", "these", "those") using:
  {chat_history or '[no prior context]'}

Safety Constraints:
- Never simulate or define fake data.
- Use only df_NCR and df_FCD.
- Do not use df_NCR if only FCD is mentioned, and vice versa.

Now return ONLY valid Python Pandas code that answers this user query:
→ {query}
"""

    return prompt.strip()

# ------------------ Plot Prompt ------------------

def build_plot_prompt(query: str, df_NCR, df_FCD, chat_history=None, source: str = None) -> str:
    query_lower = query.lower()

    if source:
        source = source.lower()
        if source == "both":
            primary_df = "both"
        elif source == "fcd":
            primary_df = "df_FCD"
        elif source == "ncr":
            primary_df = "df_NCR"
        else:
            primary_df = "[invalid]"
    else:
        if "fcd" in query_lower and "ncr" in query_lower:
            primary_df = "both"
        elif "fcd" in query_lower:
            primary_df = "df_FCD"
        elif "ncr" in query_lower:
            primary_df = "df_NCR"
        else:
            primary_df = "[no explicit dataframe mentioned]"

    ncr_cols = list(df_NCR.columns) if df_NCR is not None else '[not provided]'
    fcd_cols = list(df_FCD.columns) if df_FCD is not None else '[not provided]'

    prompt = f"""
You are a Python data visualization expert.

You are working with two Pandas DataFrames:
- df_NCR: {ncr_cols}
- df_FCD: {fcd_cols}

The user query is primarily about: **{primary_df}**

Your task is to generate a matplotlib/seaborn plot based on the following user query:
→ {query}

Strict Rules:
1. ❌ Do NOT use `value_counts()` or `reset_index()` or `to_html()` or `display(...)`
2. ✅ Instead, use Seaborn (e.g., `sns.countplot`, `sns.barplot`, `sns.histplot`, etc.)
3. ✅ At the end, add `plt.tight_layout()` and `plt.show()`
4. ✅ Assign filtered data to a variable `filtered_df` if needed
5. ✅ The chart must be shown using `plt.show()` (so that Streamlit can render it)
6. ✅ Avoid printing or returning tables — only generate visualization code

Always:
- Use the correct dataframe: `df_FCD` or `df_NCR`
- If the chart involves a count, use `sns.countplot(...)`
- Title your plots meaningfully using `plt.title(...)`
- Label axes using `plt.xlabel(...)`, `plt.ylabel(...)`
- Set `figsize` if needed using `plt.figure(figsize=(8,5))`
- For project filters:
    df['Order_Description'].fillna('').str.upper().str.contains('KEYWORD')
- Delay filtering:
    pd.to_numeric(df['Ongoing_Delay_Days'], errors='coerce').notna()

You must always:
- Use `sns.countplot(...)` or similar Seaborn method when bar chart is implied.
- End with `plt.tight_layout()` and `plt.show()`
- Assign filtered data to `filtered_df` if needed.
- Do not just use value_counts() and display a table.
- Always generate a visual plot using sns or matplotlib (e.g., bar, pie, line).
- End with plt.tight_layout() and st.pyplot(plt.gcf()).

Safety Constraints:
- Use only df_NCR and df_FCD.
- Use `df_FCD` only if the query is about FCDs
- Use `df_NCR` only if the query is about NCRs
- Never simulate or define fake data

Examples:

✅ Good:
```python
plt.figure(figsize=(10,6))
sns.countplot(data=df_FCD, x='Discipline')
plt.title('Number of FCDs by Discipline')
plt.xlabel('Discipline')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

❌ Bad:
df_FCD['Discipline'].value_counts().reset_index()
display(...)

Context:
{chat_history or '[no prior context]'}

Now return ONLY valid Python code using matplotlib + seaborn that fulfills this user query:
→ {query}
"""
    return prompt.strip()


# ------------------ ANSWER MAIN ENTRY ------------------

def answer_query(query: str):
    intent = classify_query(query)
    query_lower = query.lower()

    # Create readable chat history string for prompt
    if memory.buffer:
        chat_history = "\n".join([f"{msg.type.upper()}: {msg.content}" for msg in memory.buffer])
    else:
        chat_history = "[no prior context]"

    show_ncr = "ncr" in query_lower
    show_fcd = "fcd" in query_lower
    show_both = not show_ncr and not show_fcd

    final_response = ""  # What will be shown in assistant's chat bubble

    # 👉 INTENT: Count or Table
    if intent in ["count", "table"]:
        if show_ncr or show_both:
            prompt_ncr = build_pandas_prompt(query, df_NCR, None, chat_history=chat_history, source="ncr")
            response_ncr = llm.invoke(prompt_ncr)
            code_ncr = response_ncr.content if hasattr(response_ncr, "content") else str(response_ncr)

            memory.chat_memory.add_user_message(query + " [NCR]")
            memory.chat_memory.add_ai_message(code_ncr)

            result_ncr = safe_execute_pandas_code(code_ncr, df_NCR=df_NCR, df_FCD=None, user_query=query, intent=intent)

            # Return meaningful chat bubble only if it's a count or text
            if isinstance(result_ncr, str) and result_ncr.strip() and not any(
                w in result_ncr.lower() for w in ["displayed", "no output", "executed"]
            ):
                final_response = result_ncr
            else:
                final_response = "✅ NCR table displayed below."

        if show_fcd or show_both:
            prompt_fcd = build_pandas_prompt(query, None, df_FCD, chat_history=chat_history, source="fcd")
            response_fcd = llm.invoke(prompt_fcd)
            code_fcd = response_fcd.content if hasattr(response_fcd, "content") else str(response_fcd)

            memory.chat_memory.add_user_message(query + " [FCD]")
            memory.chat_memory.add_ai_message(code_fcd)

            result_fcd = safe_execute_pandas_code(code_fcd, df_NCR=None, df_FCD=df_FCD, user_query=query, intent=intent)

            if isinstance(result_fcd, str) and result_fcd.strip() and not any(
                w in result_fcd.lower() for w in ["displayed", "no output", "executed"]
            ):
                final_response = result_fcd
            else:
                final_response = "📒 FCD table displayed below."

        return final_response if final_response else "📒 Table(s) displayed below."

    # 👉 INTENT: Summary
    elif intent == "summary":
        context = retrieve_context(query)
        if not context.strip():
            return "🤐 No relevant context found in the document."
        return chat_chain.run(context)

    # 👉 INTENT: plot 
    elif intent == "plot":
        source = None
        if "fcd" in query_lower and "ncr" in query_lower:
            source = "both"
        elif "fcd" in query_lower:
            source = "fcd"
        elif "ncr" in query_lower:
            source = "ncr"

        plot_prompt = build_plot_prompt(
            query=query,
            df_NCR=df_NCR,
            df_FCD=df_FCD,
            chat_history=chat_history,
            source=source
        )
        response = llm.invoke(plot_prompt)
        code = response.content if hasattr(response, "content") else str(response)

        return safe_execute_pandas_code(
            code, df_NCR=df_NCR, df_FCD=df_FCD,
            user_query=query, intent="plot"
        )


    # 👉 INTENT: General Chat
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

# -------------------- STREAMLIT UI --------------------

# Style section stays unchanged (already good)

# ------------------ Header ------------------

col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://www.larsentoubro.com/media/30891/ltgrouplogo.jpg", width=220)
with col2:
    st.markdown('''
        <div class="header-text">
            <h1>🦢 NCR-FCD Insight Engine</h1>
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
        user_query = st.text_input("💬 Ask your question:", placeholder="e.g. List all FCDs in WIP", key="user_input")
        submitted = st.form_submit_button("O_O 🔍 Ask")
with img_col:
    st.image("https://cdn-icons-png.flaticon.com/512/14201/14201937.png", width=80)

st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Handle Submission (Process & Store) ------------------

if submitted and user_query:
    # Step 1: Append user query
    st.session_state.chat_history.append(("user", user_query))

    # Step 2: Process and append assistant response
    with st.spinner("Got it.. Processing Your Query!"):
        result = answer_query(user_query)

        # Do NOT duplicate these appends if answer_query already appends!
        # Only use this if it doesn't
        if isinstance(result, str) and result.strip() and not any(
            w in result.lower() for w in ["no output returned", "executed"]
        ):
            st.session_state.chat_history.append(("assistant", result))
        else:
            if "plot" in user_query.lower() or "graph" in user_query.lower() or "chart" in user_query.lower():
                st.session_state.chat_history.append(("assistant", "📊 Plot displayed below."))
            else:
                st.session_state.chat_history.append(("assistant", "📒 Table displayed below."))

# ------------------ Chat History Display (Always after processing) ------------------

st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="user-msg">👤 <strong>You:</strong><br>{msg}</div>', unsafe_allow_html=True)
    elif role == "assistant":
        if isinstance(msg, str) and msg.strip() and not any(
            w in msg.lower() for w in ["no output returned", "code executed"]
        ):
            st.markdown(f'<div class="bot-msg">🤖 <strong>Assistant:</strong><br>{msg}</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
