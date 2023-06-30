import duckdb

# Initialize DuckDB
conn = duckdb.connect('recordings.db')

# Create the table
conn.execute("""
    CREATE TABLE recordings (
        id BIGINT,
        inputs INT[],
        videofile STRING,
        game STRING
    )
""")    

print("Table 'recordings' created successfully.")
