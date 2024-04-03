import time
from fastapi import Body, Depends, FastAPI, File, HTTPException, UploadFile, status, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime, timedelta, timezone
from typing import Annotated, List, Optional, Union, Any
# https://fastapi.tiangolo.com/tutorial/extra-models/
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from openai import OpenAI

from enum import Enum
from sklearn.model_selection import train_test_split
from model_trainers import logistic_regression, k_nearest_neighbours, naive_bayes, decision_tree, random_tree, svm, lasso_regression, ridge_regression, linear_regression, ann, k_means, hierarchical
from sklearn.metrics import silhouette_score, davies_bouldin_score
from evaluation_metrics.classification import accuracy_score, recall, precision, roc_auc

import pandas as pd

import os
from dotenv import load_dotenv

load_dotenv()

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

fake_users_db = {
        "johndoe": {
        "username": "johndoe",
        "full_name": "John Doe",
        "email": "johndoe@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",
        "disabled": False,
    }
    # "johndoe": {
    #     "username": "johndoe",
    #     "full_name": "John Doe",
    #     "email": "johndoe@example.com",
    #     "hashed_password": "fakehashedsecret",
    #     "disabled": False,
    # },
    # "alice": {
    #     "username": "alice",
    #     "full_name": "Alice Wonderson",
    #     "email": "alice@example.com",
    #     "hashed_password": "fakehashedsecret2",
    #     "disabled": True,
    # },
}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Union[str, None] = None

class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None

class UserInDB(User):
    hashed_password: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def fake_hash_password(password: str):
    return "fakehashed" + password

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def fake_decode_token(token):
    # This doesn't provide any security at all
    # Check the next version
    user = get_user(fake_users_db, token)
    return user

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> Token:
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return Token(access_token=access_token, token_type="bearer")

@app.get("/users/me/", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    return current_user

@app.get("/users/me/items/")
async def read_own_items(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    return [{"item_id": "Foo", "owner": current_user.username}]

class Tags(Enum):
    machineLearningAlgo = "machine learning algorithm"
    users = "users"

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = str(process_time)
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

@app.post("/machineLearningAlgo/v1/training", response_model=list[Any], status_code=status.HTTP_200_OK, tags=[Tags.machineLearningAlgo], summary="Train a ML model")
async def train_model(
        # labels: Annotated[UploadFile, File()],
        # trainingMethod: List[str],
        # trainingType: List[str]
        labels: UploadFile,
        trainingMethod: str,
        trainingType: str
):
    print("labels", labels)
    print("trainingMethod", trainingMethod)
    print("trainingType", trainingType)


    start = datetime.now()

    model = 'null'

    labels_and_paths = pd.read_csv(labels.file)
    # print("labels_and_paths", labels_and_paths)
    x = labels_and_paths.iloc[:, :-1]  # Select all columns except the last one
    y = labels_and_paths.iloc[:, -1]   # Select only the last column
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    # for ml in trainingMethod:

    match trainingType:
        case "K Nearest Neighbours":
            # print("K Nearest Neighbours")
            model = k_nearest_neighbours.train_model(X_train, y_train)
        case "Logistic Regression":
            # print("Logistic Regression")
            model = logistic_regression.train_model(X_train, y_train)
        case "Naive Bayes":
            # print("Naive Bayes")
            model = naive_bayes.train_model(X_train, y_train)
        case "Decision Tree":
            # print("Decision Tree")
            model = decision_tree.train_model(X_train, y_train)
        case "Random Forest":
            # print("Random Forest")
            model = random_tree.train_model(X_train, y_train)
        case "SVM":
            # print("SVM")
            model = svm.train_model(X_train, y_train)
        case "Lasso Regression":
            # print("Lasso Regression")
            model = lasso_regression.train_model(X_train, y_train)
        case "Ridge Regression":
            # print("Ridge Regression")
            model = ridge_regression.train_model(X_train, y_train)
        case "Linear Regression":
            # print("Linear Regression")
            model = linear_regression.train_model(X_train, y_train)
        case "ANN":
            model = ann.train_model(X_train, y_train)
        case "K Means":
            data = labels_and_paths.iloc[:, :]
            model = k_means.train_model(data)
        case "Hierarchical":
            data = labels_and_paths.iloc[:, :]
            model = hierarchical.train_model(data)
    end = datetime.now()
    time_taken_seconds = (end - start).total_seconds()
    time_taken_ms = time_taken_seconds * 1000
    print(f'time_taken: {time_taken_ms} ms')

    # ml_model = trainingMethod[0]
    # ml_model = trainingMethod
    print("here model", model)

    if trainingMethod == 'Decision Tree':
        keys = model.keys()
        metrics = []
        for key in keys:
            metrics.append({'name': key, 'accuracy': model[key].score(X_test, y_test)})
    elif trainingMethod == 'ANN':
        # print("model", model)
        metrics = "null"
    else:
        if trainingType == 'Unsupervised':
            silhouette = silhouette_score(X_test, y_test)
            davies_bouldin = davies_bouldin_score(X_test, y_test)
            metrics = {'name': trainingMethod, 'silhouette': silhouette, 'davies_bouldin': davies_bouldin}
        else:
            y_pred = model.predict(X_test)
            metrics = { 'accuracy_score': accuracy_score.calculate_metric(y_test, y_pred),
                       'recall_score': recall.calculate_metric(y_test, y_pred),
                       'precision': precision.calculate_metric(y_test, y_pred),
                       'roc_auc': roc_auc.calculate_metric(y_test, y_pred)}

    results = [{"metrics": metrics, "time_taken_ms": time_taken_ms, "trainingType": trainingType, "trainingMethod": trainingMethod }]
    print("results", results)

    return results

@app.post("/chatai")
async def chat_response(
    user_message: str
):

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    messages = [
        {
            "role": "system",
            "content": "You are a machine learning teacher at the company auto machine learning trainer. the user doesn't know much about machine learning and will ask you questions. you will your best to answer them. if user ask questions not related to machine learning then tell them you can't answer the question",
        }
    ]

    print("user_message", user_message)

    messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )

    collected_messages = []
    for chunk in stream:
        chunk_message = chunk.choices[0].delta.content or ""
        print(chunk_message, end="")
        collected_messages.append(chunk_message)

    messages.append(
        {
            "role": "system",
            "content": "".join(collected_messages)
        }
    )

    last_system_message = None

    # Iterate through messages in reverse order
    for message in reversed(messages):
        # Check if the message has role 'system'
        if message['role'] == 'system':
            last_system_message = message['content']
            break  # Stop iterating after finding the last system message

    if last_system_message is not None:
        print("Last system message:", last_system_message)
        return { "last_messages": last_system_message}
    else:
        print("No system message found.")