import streamlit as st
from ai_agents.sql_agent.sql_agent import display_db_info, query_bot

from config.helper import create_columns

st.set_page_config(layout="wide")

def page_info():
    st.info(
        """
    Interact with the SQL Agent to query the database.

    Sample Queries

    Basic Questions:
    - What are the names of all artists in the database?
    - Can you list all the tracks in the 'Rock' genre?

    Complex Questions:
    - Show me all customers from Germany who have spent more than $50.

    Joins:
    - What are the names of tracks along with their album titles for the artist 'Metallica'?
    - Can you list all invoices with the customer’s full name and country?

    Aggregation:
    - How many tracks are there in each genre?
    - What is the total sales amount for each employee in 2023?

    Sorting and Limiting:
    - What are the top 5 most expensive tracks in the database?
    - List the 10 most recent invoices ordered by date.

    Complex Queries:
    - Which customers have purchased tracks from more than 3 different genres?
    - Show me the top 3 artists with the highest total track duration across all their albums.

    Search with Patterns:
    - Find all albums with 'Love' in their title.
    - Which tracks have names that start with 'A' and are longer than 5 minutes?

    Subqueries or Nested Queries:
    - Which customers have bought tracks from the most expensive album?
    - List all employees who have more sales than the average sales of all employees.
            
    """
    )

display_db_info()

create_columns(
    lambda: page_info(),
    lambda: query_bot(),
)