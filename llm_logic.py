import re
import requests
from langchain_core.language_models import BaseChatModel
from typing import List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import Field


from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional
import requests
import re

class NvidiaChatLLM(BaseChatModel):
    api_key: str = Field(..., description="NVIDIA API Key")

    @property
    def _llm_type(self) -> str:
        return "nvidia-chat-llm"

    def _generate(self, messages: List[HumanMessage], stop: Optional[List[str]] = None) -> ChatResult:
        user_message = messages[-1].content  # Use the latest message as input

        invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
        }
        payload = {
            "model": "meta/llama-4-maverick-17b-128e-instruct",
            "messages": [{"role": "user", "content": user_message}],
            "max_tokens": 512,
            "temperature": 0.2,
            "top_p": 1.0,
            "stream": False
        }

        try:
            response = requests.post(invoke_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            raw_output = result["choices"][0]["message"]["content"]
            cleaned_output = re.sub(r"```(?:python)?\n(.*?)```", r"\1", raw_output, flags=re.DOTALL).strip()

            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=cleaned_output))]
            )

        except Exception as e:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=f"❌ Request failed: {e}"))]
            )