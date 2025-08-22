from enum import Enum

from fastapi import FastAPI


class ModelName(str, Enum):
    ALEXNET = "alexnet"
    RESNET = "resnet"
    LENET = "lenet"


# to disable automatic docs, use these parameters in `FastAPI()`
# `docs_url=None, redoc_url=None, openapi_url=None`
app = FastAPI()

fake_items_db = [{"item_id": "Foo"}, {"item_id": "Bar"}, {"item_id": "Baz"}]


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}


@app.get("/users/{user_id}")
async def read_user(user_id: int, q: str | None = None):
    return {"user_id": user_id, "query": q}


@app.get("/models")
async def get_models():
    return list(ModelName)


@app.get("/models/{model_name}")
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


@app.get("/files/{file_path:path}")
async def get_file(file_path: str):
    return {"file_path": file_path}


@app.get("/items/")
async def get_items(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : (skip + limit)]


@app.get("/query_params/{required_path_param}")
async def query_params(
    required_path_param: str,
    required_query_param: str,
    optional_query_param_with_default: str = "default",
    optional_query_param: str | None = None,
):
    return {
        "required_path_param": required_path_param,
        "required_query_param": required_query_param,
        "optional_query_param_with_default": optional_query_param_with_default,
        "optional_query_param": optional_query_param,
    }
