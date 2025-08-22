from enum import Enum

from fastapi import FastAPI


class ModelName(str, Enum):
    ALEXNET = "alexnet"
    RESNET = "resnet"
    LENET = "lenet"


# to disable automatic docs, use these parameters in `FastAPI()`
# `docs_url=None, redoc_url=None, openapi_url=None`
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}

@app.get("/users/{user_id}")
async def read_user(user_id: int):
    return {"user_id": user_id}

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
