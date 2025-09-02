from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Annotated
from uuid import UUID

from dotenv import load_dotenv
from fastapi import Body, FastAPI, File, HTTPException, Path, Query, UploadFile, status
from fastapi.encoders import jsonable_encoder
from pydantic import AfterValidator, BaseModel, EmailStr, Field
from pydantic_ai import Agent, RunContext

# Load environment variables from .env
load_dotenv()


class Tags(Enum):
    USERS = "users"
    ITEMS = "items"
    FILES = "files"
    MODELS = "models"
    QUERY_PARAMS = "query-params"
    AI_AGENT = "ai-agent"


class ModelName(str, Enum):
    ALEXNET = "alexnet"
    RESNET = "resnet"
    LENET = "lenet"


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Item Name",
                "description": "Item Description",
                "price": 9.99,
                "tax": 0.5,
            }
        }
    }


class PaginationParams(BaseModel):
    skip: int = Field(0, ge=0)
    limit: int = Field(10, ge=1, le=100)


# to disable automatic docs, use these parameters in `FastAPI()`
# `docs_url=None, redoc_url=None, openapi_url=None`
app = FastAPI()

fake_items_db = [{"item_id": "Foo"}, {"item_id": "Bar"}, {"item_id": "Baz"}]


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get(
    "/users/me",
    tags=[Tags.USERS],
    summary="Get the current user",
    description="Retrieve the current authenticated user",
)
async def read_user_me():
    return {"user_id": "the current user"}


@app.get(
    "/users/{user_id}",
    tags=[Tags.USERS],
    summary="Get a user by ID",
    description="Retrieve a user by their unique user ID",
)
async def read_user(user_id: int, q: str | None = None):
    return {"user_id": user_id, "query": q}


@app.get("/models", tags=[Tags.MODELS])
async def get_models():
    return list(ModelName)


@app.get("/models/{model_name}", tags=[Tags.MODELS])
async def get_model(model_name: ModelName):
    match model_name:
        case ModelName.ALEXNET:
            return {"model_name": model_name, "message": "Deep Learning FTW!"}
        case ModelName.LENET:
            return {"model_name": model_name, "message": "LeCNN all the images"}
        case ModelName.RESNET:
            return {"model_name": model_name, "message": "ResNet model"}
        case _:
            return {"model_name": model_name, "message": "Unmapped model"}


@app.get("/files/{file_path:path}", tags=[Tags.FILES], deprecated=True)
async def get_file(file_path: str):
    return {"file_path": file_path}


@app.get("/items/", tags=[Tags.ITEMS])
async def get_items(
    pagination_query: Annotated[PaginationParams, Query()],
):
    return fake_items_db[
        pagination_query.skip : (pagination_query.skip + pagination_query.limit)
    ]


items = {"foo": "The Foo Wrestlers"}


@app.get("/items/{item_id}/", tags=[Tags.ITEMS])
async def get_item(item_id: str):
    if item_id not in items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Item not found"
        )
    return items[item_id]


@app.get("/query-params/{required_path_param}", tags=[Tags.QUERY_PARAMS])
async def query_params(
    required_path_param: Annotated[str, Path(title="The required path parameter")],
    required_query_param: Annotated[str, Query(alias="required-query-param")],
    optional_query_param_with_default: Annotated[
        str, Query(alias="optional-query-param-with-default")
    ] = "default",
    optional_query_param: Annotated[
        str | None, Query(alias="optional-query-param")
    ] = None,
):
    return {
        "required_path_param": required_path_param,
        "required_query_param": required_query_param,
        "optional_query_param_with_default": optional_query_param_with_default,
        "optional_query_param": optional_query_param,
    }


@app.post(
    "/items/",
    tags=[Tags.ITEMS],
    summary="Create an item",
    response_description="The created item",
)
async def create_item(item: Item):
    """
    Create an item with all the information:
    - **item_id**: The unique identifier for the item
    - **description**: A brief description of the item
    - **price**: The price of the item
    - **tax**: The tax amount for the item (optional)
    """
    item_dict = item.model_dump()
    if item.tax is not None:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict


def check_valid_custom(value: str | None) -> str | None:
    if value == "invalid":
        raise ValueError("Invalid custom value")
    return value


@app.get("/items/search", tags=[Tags.ITEMS])
async def search_items(
    q: Annotated[
        str,
        Query(
            title="Query string",
            description="Query string for the items to search",
            min_length=3,
        ),
    ],
    firms: Annotated[set[str], Query()] = set(),
    firm_list: Annotated[
        set[str],
        Query(
            alias="firm-list",
            deprecated=True,
            description="Old field to search for firms, not used anymore",
        ),
    ] = set(),
    hidden_query: Annotated[
        str | None, Query(alias="hidden-query", include_in_schema=False)
    ] = None,
    custom: Annotated[
        str | None, Query(max_length=10), AfterValidator(check_valid_custom)
    ] = None,
):
    if firm_list and not firms:
        firms = firm_list
    results = {
        "items": [item for item in fake_items_db if q in item["item_id"]],
        "q": q,
        "firms": firms,
    }
    if hidden_query:
        results.update({"hidden_query": hidden_query})
    if custom:
        results.update({"custom": custom})
    return results


@app.put("/items/{item_id}", tags=[Tags.ITEMS])
async def read_items(
    item_id: UUID,
    start_datetime: Annotated[datetime, Body()],
    end_datetime: Annotated[datetime, Body()],
    process_after: Annotated[timedelta, Body()],
    repeat_at: Annotated[time | None, Body()] = None,
):
    start_process = start_datetime + process_after
    duration = end_datetime - start_process
    return {
        "item_id": item_id,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "process_after": process_after,
        "repeat_at": repeat_at,
        "start_process": start_process,
        "duration": duration,
    }


# region AI Agent Section


@dataclass
class RequestDependencies:
    customer_name: str | None = None


class RequestOutput(BaseModel):
    response: str = Field(description="Response to the customer's query")
    block_nsfw: bool = Field(description="Whether to block query in case it's NSFW")
    courtesy_level: int = Field(
        description="Courtesy level of the request", ge=0, le=10
    )


ai_agent = Agent(
    "google-gla:gemini-2.0-flash",
    deps_type=RequestDependencies,
    output_type=RequestOutput,
    system_prompt=(
        "You are an helpful online assistant, replying to simple questions "
        "and evaluating the courtesy level of the request on a scale of 0 to 10. "
        # "Be concise, reply with one sentence."
    ),
)


@ai_agent.system_prompt
async def add_customer_name(ctx: RunContext[RequestDependencies]) -> str:
    if ctx.deps.customer_name:
        return f"The customer name is {ctx.deps.customer_name!r}."
    return "No customer name provided."


@app.post(
    "/ai-query",
    tags=[Tags.AI_AGENT],
    summary="AI Query",
    description="Ask a question to the AI agent",
)
async def ai_query(
    query: Annotated[str, Body(examples=["What is the capital of France?"])],
    customer_name: Annotated[str | None, Body(examples=["John Doe"])] = None,
):
    deps = RequestDependencies(customer_name=customer_name)
    result = await ai_agent.run(query, deps=deps)

    return {
        "query": query,
        "result": result,
    }


# endregion


# region Custom Response Model


class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: str | None = None


class UserIn(UserBase):
    password: str


class UserOut(UserBase):
    pass


class UserInDB(UserBase):
    hashed_password: str


def fake_password_hasher(raw_password: str):
    return "supersecret_" + raw_password + "_hashed"


def fake_save_user(user_in: UserIn):
    hashed_password = fake_password_hasher(user_in.password)
    user_in_db = UserInDB(**user_in.model_dump(), hashed_password=hashed_password)
    print(f"User saved! {user_in_db}")
    return user_in_db


@app.post("/user/", status_code=status.HTTP_201_CREATED, tags=[Tags.USERS])
async def create_user(user_in: UserIn) -> UserOut:
    user_saved = fake_save_user(user_in)
    return UserOut(**user_saved.model_dump())


# endregion


# region File upload


@app.post("/uploadfile/", tags=[Tags.FILES])
async def create_upload_file(
    comment: Annotated[str, Body(examples=["This is a test file"])],
    file: UploadFile,
    additional_files: Annotated[list[UploadFile], File()] = [],
):
    # contents = await file.read()
    return {
        "comment": comment,
        "filename": file.filename,
        "content_type": file.content_type,
        "size": file.size,
        "additional_files": [
            {
                "filename": f.filename,
                "content_type": f.content_type,
                "size": f.size,
            }
            for f in additional_files
        ],
    }


# endregion


# region PUT and PATCH examples


class ItemV2(BaseModel):
    name: str | None = None
    description: str | None = None
    price: float | None = None
    tax: float = 10.5
    tags: list[str] = []


items_v2 = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}


@app.get(
    "/items-v2/{item_id}",
    response_model=ItemV2,
    tags=[Tags.ITEMS],
    summary="Get an item by ID",
)
async def read_item(item_id: str):
    return items_v2[item_id]


@app.put(
    "/items-v2/{item_id}",
    response_model=ItemV2,
    tags=[Tags.ITEMS],
    summary="Fully update an item by ID",
)
async def update_item_v2(item_id: str, item: ItemV2):
    update_item_encoded = jsonable_encoder(item)
    items_v2[item_id] = update_item_encoded
    return update_item_encoded


@app.patch(
    "/items-v2/{item_id}",
    response_model=ItemV2,
    tags=[Tags.ITEMS],
    summary="Partially update an item by ID",
)
async def partially_update_item_v2(item_id: str, item: ItemV2):
    stored_item_data = items_v2[item_id]
    stored_item_model = ItemV2(**stored_item_data)
    update_data = item.model_dump(exclude_unset=True)
    updated_item = stored_item_model.model_copy(update=update_data)
    items_v2[item_id] = jsonable_encoder(updated_item)
    return updated_item


# endregion
