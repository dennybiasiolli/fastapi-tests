from enum import Enum

from pydantic import BaseModel


class Tags(Enum):
    USERS = "users"
    ITEMS = "items"
    FILES = "files"
    MODELS = "models"
    QUERY_PARAMS = "query-params"
    AI_AGENT = "ai-agent"


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    email: str | None = None
    full_name: str | None = None
    disabled: bool | None = None


class UserInDB(User):
    hashed_password: str
