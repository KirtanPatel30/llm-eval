from pydantic import BaseModel


class AdHocPromptRequest(BaseModel):
    prompt: str
    context: str | None = None


class ProviderInfo(BaseModel):
    name: str
    is_demo: bool
