from enum import StrEnum, auto
from typing import TypeAlias


class Provider(StrEnum):
    OPENAI = auto()
    ANTHROPIC = auto()
    GOOGLE = auto()
    GROQ = auto()
    AWS = auto()
    FAKE = auto()
    TAN = auto()


class OpenAIModelName(StrEnum):
    """https://platform.openai.com/docs/models/gpt-4o"""

    GPT_4O_MINI = "gpt-4o-mini"
    GPT_4O = "gpt-4o"


class AnthropicModelName(StrEnum):
    """https://docs.anthropic.com/en/docs/about-claude/models#model-names"""

    HAIKU_3 = "claude-3-haiku"
    HAIKU_35 = "claude-3.5-haiku"
    SONNET_35 = "claude-3.5-sonnet"


class GoogleModelName(StrEnum):
    """https://ai.google.dev/gemini-api/docs/models/gemini"""

    GEMINI_15_FLASH = "gemini-1.5-flash"
    GEMINI_2_EXP = "gemini-2.0-flash-exp"


class GroqModelName(StrEnum):
    """https://console.groq.com/docs/models"""

    LLAMA_31_8B = "groq-llama-3.1-8b"
    LLAMA_33_70B = "groq-llama-3.3-70b"

    LLAMA_GUARD_3_8B = "groq-llama-guard-3-8b"


class AWSModelName(StrEnum):
    """https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html"""

    BEDROCK_HAIKU = "bedrock-3.5-haiku"


class FakeModelName(StrEnum):
    """Fake model for testing."""

    FAKE = "fake"


class TanModelName(StrEnum):
    """Tan model for testing."""

    LLAMA = "Llama-3.3-70B"
    LMSTUDIO = "lmstudio/qwen2.5-7b-instruct"
    SAMBANOVA = "sambanova/Qwen2.5-72B-Instruct"


AllModelEnum: TypeAlias = (
    OpenAIModelName
    | AnthropicModelName
    | GoogleModelName
    | GroqModelName
    | AWSModelName
    | FakeModelName
    | TanModelName
)
