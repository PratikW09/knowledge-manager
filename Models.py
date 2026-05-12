from pydantic import BaseModel


class URLInput(BaseModel):
    url: str


class SaveInput(BaseModel):
    url:      str
    category: str  # confirmed by user


class ChatInput(BaseModel):
    question: str
    history:  list[dict] = []