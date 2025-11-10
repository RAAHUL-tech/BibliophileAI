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

def delete_user_follows_user(user_id: str, followed_user_id: str):
    """
    Remove a FOLLOWS relationship between two users.
    """
    with driver.session() as session:
        session.run("""
            MATCH (u1:User {id: $user_id})-[r:FOLLOWS]->(u2:User {id: $followed_user_id})
            DELETE r
        """, user_id=user_id, followed_user_id=followed_user_id)

def neo4j_suggest_followers(user_id: str, limit: int = 50):
    with driver.session() as session:
        # 1. Find mutual friends (users where there is a FOLLOWS relationship both ways)
        mutual_friends_result = session.run("""
            MATCH (me:User {id: $user_id})-[:FOLLOWS]->(friend:User)-[:FOLLOWS]->(me)
            RETURN friend.id AS id, friend.username AS username
        """, user_id=user_id)
        mutual_friends = [rec["id"] for rec in mutual_friends_result]

        suggestions = {}
        exclude_ids = set(mutual_friends)
        exclude_ids.add(user_id)

        # 2. Suggest their followers & following (excluding self and already mutual friends)
        if mutual_friends:
            # Followers of mutual friends
            followers_result = session.run("""
                MATCH (f:User)-[:FOLLOWS]->(mf:User)
                WHERE mf.id IN $mutual_friends AND f.id <> $user_id
                RETURN DISTINCT f.id AS id, f.username AS username
            """, mutual_friends=mutual_friends, user_id=user_id)
            for rec in followers_result:
                if rec["id"] not in exclude_ids:
                    suggestions[rec["id"]] = dict(rec)
                    exclude_ids.add(rec["id"])

            # Following of mutual friends
            following_result = session.run("""
                MATCH (mf:User)-[:FOLLOWS]->(f:User)
                WHERE mf.id IN $mutual_friends AND f.id <> $user_id
                RETURN DISTINCT f.id AS id, f.username AS username
            """, mutual_friends=mutual_friends, user_id=user_id)
            for rec in following_result:
                if rec["id"] not in exclude_ids:
                    suggestions[rec["id"]] = dict(rec)
                    exclude_ids.add(rec["id"])

        # 3. Add users from author overlap logic (excluding self, mutual friends, any already suggested)
        author_suggestions_result = session.run("""
            MATCH (me:User {id: $user_id})-[:FOLLOWS]->(a:Author)
            WITH collect(a) as my_authors, me
            MATCH (other:User)-[:FOLLOWS]->(a2:Author)
            WHERE a2 IN my_authors AND other.id <> me.id AND NOT other.id IN $exclude_ids
            RETURN DISTINCT other.id as id, other.username as username
            ORDER BY rand()
            LIMIT $auth_limit
        """, user_id=user_id, exclude_ids=list(exclude_ids), auth_limit=limit)
        for rec in author_suggestions_result:
            if rec["id"] not in exclude_ids:
                suggestions[rec["id"]] = dict(rec)
                exclude_ids.add(rec["id"])

        # 4. If not enough, fill with random users (excluding all above)
        if len(suggestions) < limit:
            random_fill_result = session.run("""
                MATCH (me:User {id: $user_id})
                MATCH (u:User)
                WHERE u.id <> me.id AND NOT u.id IN $exclude_ids
                RETURN u.id AS id, u.username AS username
                ORDER BY rand()
                LIMIT $remaining
            """, user_id=user_id, exclude_ids=list(exclude_ids), remaining=limit-len(suggestions))
            for rec in random_fill_result:
                if rec["id"] not in exclude_ids:
                    suggestions[rec["id"]] = dict(rec)
                    exclude_ids.add(rec["id"])

        # 5. Trim to the limit and return results as a list
        return list(suggestions.values())[:limit]


def neo4j_are_users_mutually_following(user_a_id: str, user_b_id: str) -> bool:
    """
    Check if user A follows user B AND user B follows user A.
    """
    with driver.session() as session:
        result = session.run("""
            MATCH (a:User {id: $user_a}), (b:User {id: $user_b})
            RETURN EXISTS( (a)-[:FOLLOWS]->(b) ) AND EXISTS( (b)-[:FOLLOWS]->(a) ) AS mutual_follow
        """, user_a=user_a_id, user_b=user_b_id)
        record = result.single()
        return record["mutual_follow"] if record else False

def neo4j_get_followers(user_id: str):
    """
    Get list of users who follow the given user_id.
    """
    with driver.session() as session:
        result = session.run("""
            MATCH (u:User)-[:FOLLOWS]->(target:User {id: $user_id})
            RETURN u.id AS id, u.username AS username
        """, user_id=user_id)
        return [dict(record) for record in result]

def neo4j_get_following(user_id: str):
    """
    Get list of users that the given user_id is following.
    """
    with driver.session() as session:
        result = session.run("""
            MATCH (u:User {id: $user_id})-[:FOLLOWS]->(following:User)
            RETURN following.id AS id, following.username AS username
        """, user_id=user_id)
        return [dict(record) for record in result]
