from neo4j import GraphDatabase
from typing import List, Optional
from datetime import datetime
import os

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def create_user_node(user_id: str, username: str, age: Optional[int]=None, pincode: Optional[str]=None):
    with driver.session() as session:
        session.run(
            """
            MERGE (u:User {id: $user_id})
            SET u.username = $username, u.age = $age, u.pincode = $pincode
            """,
            user_id=user_id, username=username, age=age, pincode=pincode
        )

def create_user_likes_genres(user_id: str, genres: List[str]):
    with driver.session() as session:
        for genre in genres:
            session.run("""
                MERGE (u:User {id: $user_id})
                MERGE (g:Genre {name: $genre})
                MERGE (u)-[:LIKES]->(g)
            """, user_id=user_id, genre=genre)

def create_user_follows_authors(user_id: str, author_names: List[str]):
    with driver.session() as session:
        for author in author_names:
            session.run("""
                MERGE (u:User {id: $user_id})
                MERGE (a:Author {name: $author})
                MERGE (u)-[:FOLLOWS]->(a)
            """, user_id=user_id, author=author)

def create_user_follows_users(user_id: str, follow_ids: List[str]):
    with driver.session() as session:
        for fid in follow_ids:
            session.run("""
                MERGE (u1:User {id: $user_id})
                MERGE (u2:User {id: $fid})
                MERGE (u1)-[:FOLLOWS]->(u2)
            """, user_id=user_id, fid=fid)

def create_user_read_book(user_id: str, book_id: str):
    with driver.session() as session:
        session.run("""
            MERGE (u:User {id: $user_id})
            MERGE (b:Book {id: $book_id})
            MERGE (u)-[:READ]->(b)
        """, user_id=user_id, book_id=book_id)

def create_user_bookmarked_book(user_id: str, book_id: str):
    with driver.session() as session:
        session.run("""
            MERGE (u:User {id: $user_id})
            MERGE (b:Book {id: $book_id})
            MERGE (u)-[:BOOKMARKED]->(b)
        """, user_id=user_id, book_id=book_id)

def delete_user_bookmarked_book(user_id: str, book_id: str):
    with driver.session() as session:
        session.run("""
            MATCH (u:User {id: $user_id})-[r:BOOKMARKED]->(b:Book {id: $book_id})
            DELETE r
        """, user_id=user_id, book_id=book_id)

def create_user_rated_book(user_id: str, book_id: str, score: float, timestamp: Optional[str]=None):
    if not timestamp:
        timestamp = datetime.utcnow().isoformat()
    with driver.session() as session:
        session.run("""
            MERGE (u:User {id: $user_id})
            MERGE (b:Book {id: $book_id})
            MERGE (u)-[r:RATED]->(b)
            SET r.score = $score, r.timestamp = $timestamp
        """, user_id=user_id, book_id=book_id, score=score, timestamp=timestamp)

def create_user_preferences(
    user_id: str,
    username: str,
    genres: Optional[List[str]]=None,
    authors: Optional[List[str]]=None,
    age: Optional[int]=None,
    pincode: Optional[str]=None
):
    create_user_node(user_id, username, age, pincode)
    if genres:
        create_user_likes_genres(user_id, genres)
    if authors:
        create_user_follows_authors(user_id, authors)

def patch_user_preferences(
    user_id: str,
    username: str,
    old_genres: List[str],
    new_genres: List[str],
    old_authors: List[str],
    new_authors: List[str],
    age: Optional[int]=None,
    pincode: Optional[str]=None
):
    with driver.session() as session:
        # Always ensure User node is up-to-date
        session.run(
            """
            MERGE (u:User {id: $user_id})
            SET u.username = $username, u.age = $age, u.pincode = $pincode
            """,
            user_id=user_id, username=username, age=age, pincode=pincode
        )
        
        # Remove old genre relationships not present anymore
        obsolete_genres = set(old_genres) - set(new_genres)
        for genre in obsolete_genres:
            session.run("""
                MATCH (u:User {id: $user_id})-[r:LIKES]->(g:Genre {name: $genre})
                DELETE r
            """, user_id=user_id, genre=genre)

        # Add new genre relationships
        added_genres = set(new_genres) - set(old_genres)
        for genre in added_genres:
            session.run("""
                MERGE (u:User {id: $user_id})
                MERGE (g:Genre {name: $genre})
                MERGE (u)-[:LIKES]->(g)
            """, user_id=user_id, genre=genre)

        # Remove old author FOLLOWS relationships not present anymore
        obsolete_authors = set(old_authors) - set(new_authors)
        for author in obsolete_authors:
            session.run("""
                MATCH (u:User {id: $user_id})-[r:FOLLOWS]->(a:Author {name: $author})
                DELETE r
            """, user_id=user_id, author=author)

        # Add new author FOLLOWS relationships
        added_authors = set(new_authors) - set(old_authors)
        for author in added_authors:
            session.run("""
                MERGE (u:User {id: $user_id})
                MERGE (a:Author {name: $author})
                MERGE (u)-[:FOLLOWS]->(a)
            """, user_id=user_id, author=author)

def update_user_profile_fields(user_id: str, age: int = None, pincode: str = None):
    with driver.session() as session:
        session.run(
            """
            MERGE (u:User {id: $user_id})
            SET u.age = $age, u.pincode = $pincode
            """,
            user_id=user_id, age=age, pincode=pincode
        )
