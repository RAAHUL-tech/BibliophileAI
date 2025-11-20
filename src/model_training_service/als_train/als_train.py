import os
import logging
import time
import pandas as pd
import numpy as np
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import scipy.sparse
import implicit

logging.basicConfig(level=logging.INFO)

def get_mongo_client():
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("Environment variable MONGO_URI is not set.")
    for i in range(5):
        try:
            client = MongoClient(mongo_uri, server_api=ServerApi('1'), serverSelectionTimeoutMS=5000)
            # Ping to check the connection
            client.admin.command('ping')
            logging.info("Connected to MongoDB Atlas cluster!")
            return client
        except Exception as e:
            logging.error(f"MongoDB connection failed (attempt {i+1}): {e}")
            time.sleep(5)
    raise RuntimeError("Failed to connect to MongoDB after 5 attempts")

mongo = get_mongo_client()
collection = mongo['click_stream']['events']
events_cursor = collection.find()
events = list(events_cursor)

if not events:
    raise RuntimeError("No events found in MongoDB - cannot train ALS model.")

# DataFrame for user-book interactions
df = pd.DataFrame(events)
if "user_id" not in df.columns or "book_id" not in df.columns or "event_type" not in df.columns:
    raise RuntimeError("MongoDB events must have 'user_id', 'book_id', and 'event_type' fields.")

# Map event_type to implicit weight (tune as needed)
df["weight"] = df["event_type"].map(lambda x: 2.0 if x.lower() == "read" else 1.0)
df = df.groupby(["user_id", "book_id"]).weight.sum().reset_index()

# Map ids to indices
user_list = df["user_id"].unique().tolist()
book_list = df["book_id"].unique().tolist()
user_map = {u: i for i, u in enumerate(user_list)}
book_map = {b: i for i, b in enumerate(book_list)}
df["user_idx"] = df["user_id"].map(user_map)
df["book_idx"] = df["book_id"].map(book_map)

# Create implicit feedback sparse matrix
mat = scipy.sparse.coo_matrix(
    (df["weight"], (df["user_idx"], df["book_idx"])),
    shape=(len(user_list), len(book_list))
)

factors = int(os.environ.get("ALS_FACTORS", 64))
reg = float(os.environ.get("ALS_REG", 0.1))
alpha = float(os.environ.get("ALS_ALPHA", 40.0))
iterations = int(os.environ.get("ALS_ITER", 20))

model = implicit.als.AlternatingLeastSquares(
    factors=factors,
    regularization=reg,
    iterations=iterations,
    calculate_training_loss=True
)
logging.info("Training ALS model...")
mat = (mat * alpha).astype("double")
model.fit(mat)
logging.info("ALS training finished.")

# Save user/book factors
user_factors = pd.DataFrame(model.user_factors)
user_factors["user_id"] = user_list
user_factors.to_parquet("user_factors.parquet")

book_factors = pd.DataFrame(model.item_factors)
book_factors["book_id"] = book_list
book_factors.to_parquet("book_factors.parquet")

logging.info("User/book latent factors written to parquet.")
