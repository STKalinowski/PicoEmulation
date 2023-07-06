import duckdb

# Initialize DuckDB
conn = duckdb.connect('recordings.db')

# Create the table
res = conn.execute("""
SELECT * FROM recordings
LIMIT 20;
""")    
d = res.fetchdf()
print(d)