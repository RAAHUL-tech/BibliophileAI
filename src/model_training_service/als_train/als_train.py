import os
import logging
import pandas as pd
import numpy as np
import ray
import boto3
import pickle
from implicit.als import AlternatingLeastSquares
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO)
ray.init(address=os.getenv("RAY_ADDRESS", "local"))

s3_uri = os.environ["S3_URI"]
if not s3_uri.endswith("/"):
    s3_uri += "/"
ALS_PREFIX = os.getenv("ALS_S3_PREFIX")

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

@ray.remote
def load_events():
    mongo_uri = os.environ["MONGO_URI"]
    client = MongoClient(mongo_uri, server_api=ServerApi("1"), serverSelectionTimeoutMS=5000)

    df = pd.DataFrame(list(client["click_stream"]["events"].find()))
    df = df[df["item_id"].notna()]
    return df

@ray.remote
def preprocess(df):
    valid_event_types = ["read", "page_turn", "review", "bookmark_add"]
    df = df[df["event_type"].isin(valid_event_types)]

    df["weight"] = df["event_type"].map(
        lambda x: 3.0 if x == "review" else 2.0 if x == "read" else 1.0
    )
    df.rename(columns={"item_id": "book_id"}, inplace=True)

    df = df.groupby(["user_id", "book_id"], as_index=False)["weight"].sum()

    user_list = df["user_id"].unique().tolist()
    book_list = df["book_id"].unique().tolist()

    user_map = {u: i for i, u in enumerate(user_list)}
    book_map = {b: i for i, b in enumerate(book_list)}

    df["user_idx"] = df["user_id"].map(user_map)
    df["book_idx"] = df["book_id"].map(book_map)

    return df, user_list, book_list

@ray.remote
def train_als(df, user_list, book_list):
    mat = sp.coo_matrix(
        (df["weight"], (df["user_idx"], df["book_idx"])),
        shape=(len(user_list), len(book_list))
    ) * float(os.environ.get("ALS_ALPHA", 40.0))

    model = AlternatingLeastSquares(
        factors=int(os.environ.get("ALS_FACTORS", 64)),
        regularization=float(os.environ.get("ALS_REG", 0.1)),
        iterations=int(os.environ.get("ALS_ITER", 20)),
        calculate_training_loss=True
    )
    model.fit(mat)

    user_factors = pd.DataFrame(model.user_factors)
    user_factors["user_id"] = user_list

    book_factors = pd.DataFrame(model.item_factors)
    book_factors["book_id"] = book_list

    return model, user_factors, book_factors


# === Workflow Execution ===
df = ray.get(load_events.remote())
df, user_list, book_list = ray.get(preprocess.remote(df))
model, user_factors, book_factors = ray.get(train_als.remote(df, user_list, book_list))

# === Save Local Files ===
user_factors_file = "user_factors.parquet"
book_factors_file = "book_factors.parquet"
model_file = "als_model.pkl"

user_factors.to_parquet(user_factors_file)
book_factors.to_parquet(book_factors_file)

with open(model_file, "wb") as f:
    pickle.dump(model, f)

logging.info("Local model & features saved!")

# === Upload to S3 ===
s3 = get_s3_client()

def upload_to_s3(local_file, s3_uri):
    bucket = s3_uri.split("/")[2]
    key = "/".join(s3_uri.split("/")[3:])
    s3.upload_file(local_file, bucket, key)
    logging.info(f"Uploaded {local_file} → {s3_uri}")

S3_USER_FACTORS_PATH = f"{s3_uri.rstrip('/')}/{ALS_PREFIX}/user_factors.parquet"
S3_BOOK_FACTORS_PATH = f"{s3_uri.rstrip('/')}/{ALS_PREFIX}/book_factors.parquet"
S3_MODEL_PATH = f"{s3_uri.rstrip('/')}/{ALS_PREFIX}/als_model.pkl"
upload_to_s3(user_factors_file, S3_USER_FACTORS_PATH)
upload_to_s3(book_factors_file, S3_BOOK_FACTORS_PATH)
upload_to_s3(model_file, S3_MODEL_PATH)

logging.info("Training workflow completed & uploaded to S3 successfully!")
# ---- Stop Ray cleanly to prevent Airflow duplicate task run ----
ray.shutdown()
logging.info("Ray cluster shut down. Exiting container.")
