import os
import logging

from dotenv import load_dotenv
from langchain.agents import create_agent

from langchain_openai import AzureChatOpenAI, ChatOpenAI, AzureOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import (
    END,
    START,
    MessagesState,
    StateGraph,
)
from typing_extensions import Literal
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

logger = logging.getLogger(__name__)

load_dotenv()
#os.environ['AZURE_OPENAI_ENDPOINT']=endpoint
model = 'Kimi-K2.5'
endpoint =  "https://taylo-mlrsu5qj-eastus2.services.ai.azure.com"
api_version = '2024-05-01-preview'
model = init_chat_model(
    f"azure_openai:{model}",
    azure_endpoint=endpoint,
    api_version=api_version,
    azure_ad_token_provider=get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
)
print(model.invoke('are u kimi?'))
