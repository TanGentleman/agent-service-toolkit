from functools import cache
from typing import TypeAlias

from langchain_anthropic import ChatAnthropic
from langchain_aws import ChatBedrock
from langchain_community.chat_models import FakeListChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

TAN_BASE_URL = "http://localhost:4000/v1"

from schema.models import (
    AllModelEnum,
    AnthropicModelName,
    AWSModelName,
    FakeModelName,
    GoogleModelName,
    GroqModelName,
    OpenAIModelName,
    TanModelName,
)

_MODEL_TABLE = {
    OpenAIModelName.GPT_4O_MINI: "gpt-4o-mini",
    OpenAIModelName.GPT_4O: "gpt-4o",
    AnthropicModelName.HAIKU_3: "claude-3-haiku-20240307",
    AnthropicModelName.HAIKU_35: "claude-3-5-haiku-latest",
    AnthropicModelName.SONNET_35: "claude-3-5-sonnet-latest",
    GoogleModelName.GEMINI_15_FLASH: "gemini-1.5-flash",
    GoogleModelName.GEMINI_2_EXP: "gemini-2.0-flash-exp",
    GroqModelName.LLAMA_31_8B: "llama-3.1-8b-instant",
    GroqModelName.LLAMA_33_70B: "llama-3.3-70b-versatile",
    GroqModelName.LLAMA_GUARD_3_8B: "llama-guard-3-8b",
    AWSModelName.BEDROCK_HAIKU: "anthropic.claude-3-5-haiku-20241022-v1:0",
    FakeModelName.FAKE: "fake",
    TanModelName.LLAMA: "Llama-3.3-70B",
    TanModelName.LMSTUDIO: "lmstudio/qwen2.5-7b-instruct",
    TanModelName.SAMBANOVA: "sambanova/Meta-Llama-3.1-70B-Instruct",
}

ModelT: TypeAlias = ChatOpenAI | ChatAnthropic | ChatGoogleGenerativeAI | ChatGroq | ChatBedrock


@cache
def get_model(model_name: AllModelEnum, /) -> ModelT:
    # NOTE: models with streaming=True will send tokens as they are generated
    # if the /stream endpoint is called with stream_tokens=True (the default)
    api_model_name = _MODEL_TABLE.get(model_name)
    if not api_model_name:
        raise ValueError(f"Unsupported model: {model_name}")

    if model_name in OpenAIModelName:
        return ChatOpenAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in AnthropicModelName:
        return ChatAnthropic(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GoogleModelName:
        return ChatGoogleGenerativeAI(model=api_model_name, temperature=0.5, streaming=True)
    if model_name in GroqModelName:
        if model_name == GroqModelName.LLAMA_GUARD_3_8B:
            return ChatGroq(model=api_model_name, temperature=0.0)
        return ChatGroq(model=api_model_name, temperature=0.5)
    if model_name in AWSModelName:
        return ChatBedrock(model_id=api_model_name, temperature=0.5)
    if model_name in FakeModelName:
        return FakeListChatModel(responses=["This is a test response from the fake model."])
    if model_name in TanModelName:
        import os
        return ChatOpenAI(model=api_model_name, base_url=TAN_BASE_URL, api_key=os.getenv("TAN_API_KEY"), temperature=0.5)
