import sys
sys.path.append('.')

from production.database.connection import get_engine
from sqlalchemy import text

engine = get_engine()
print('Available tables:')

with engine.connect() as conn:
    result = conn.execute(text('SHOW TABLES'))
    tables = [row[0] for row in result]
    for table in tables:
        print(f"  - {table}") 