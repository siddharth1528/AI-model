import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ------------------ Load Parquet Files ------------------

df_NCR = pd.read_parquet("ncr_data.parquet")
df_FCD = pd.read_parquet("fcd_data.parquet")

# ------------------ Clean Columns ------------------

for df in [df_NCR, df_FCD]:
    df.columns = df.columns.str.strip().str.replace("\n", "_").str.replace(" ", "_")
    if "Ongoing_Delay_Days" in df.columns:
        df["Ongoing_Delay_Days"] = pd.to_numeric(df["Ongoing_Delay_Days"], errors="coerce")

# ------------------ Prepare Vectorstore for Summary/RAG ------------------

def prepare_documents(df, source_name=""):
    docs = []
    for _, row in df.iterrows():
        text = " | ".join([f"{col}: {str(row[col])}" for col in df.columns if pd.notnull(row[col])])
        docs.append({"text": text, "metadata": {"source": source_name}})
    return pd.DataFrame(docs)

df_combined = pd.concat([
    prepare_documents(df_NCR, "NCR"),
    prepare_documents(df_FCD, "FCD")
])

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n", ".", "|", ","]
)

documents = text_splitter.create_documents(
    texts=df_combined["text"].tolist(),
    metadatas=df_combined["metadata"].tolist()
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embedding_model)
