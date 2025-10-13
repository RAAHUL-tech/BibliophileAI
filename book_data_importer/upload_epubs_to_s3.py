import os
import requests
import boto3
from supabase import create_client, Client
from tqdm import tqdm

# Supabase config
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]  # Use service key for full access

# AWS S3 config
AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
AWS_S3_BUCKET = os.environ["AWS_S3_BUCKET"]
S3_PREFIX = "books-epub/"  # Folder in the S3 bucket

# Init clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

def fetch_all_books():
    # Pagination: Supabase returns max 1000 rows per call
    all_books = []
    batch_size = 1000
    offset = 0
    while True:
        response = (
            supabase.table("books")
            .select("id,download_link")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        batch = response.data
        if not batch:
            break
        all_books.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size
    return all_books

def download_and_upload_epub(book, s3_client):
    download_url = book.get('download_link')
    book_id = book.get('id')
    if not download_url:
        print(f"Book ID {book_id}: no download_link, skipped")
        return
    try:
        # Download EPUB (follow redirects)
        with requests.get(download_url, stream=True, allow_redirects=True, timeout=30) as r:
            r.raise_for_status()
            # Compose S3 key (filename)
            epub_filename = f"{book_id}.epub"
            s3_key = S3_PREFIX + epub_filename
            # Upload to S3
            s3_client.upload_fileobj(r.raw, AWS_S3_BUCKET, s3_key)
        print(f"Uploaded Book {book_id} to S3 as {s3_key}")
    except Exception as e:
        print(f"Failed for Book {book_id}: {e}")

def main():
    books = fetch_all_books()
    print(f"Fetched {len(books)} books from Supabase")
    for book in tqdm(books):
        download_and_upload_epub(book, s3_client)

if __name__ == "__main__":
    main()
