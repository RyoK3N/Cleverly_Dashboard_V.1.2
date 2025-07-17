import sqlite3
import os
from werkzeug.security import generate_password_hash

def setup_database():
    db_path = './database.db'
    
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")
    
    print("Creating new database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username VARCHAR(80) UNIQUE NOT NULL,
            password_hash VARCHAR(120) NOT NULL
        )
    ''')
    print("Created user table")
    
    password_hash = generate_password_hash('Clev@2025')
    print(f"Generated password hash: {password_hash}")
    
    cursor.execute('''
        INSERT INTO user (username, password_hash) 
        VALUES (?, ?)
    ''', ('Dharmil', password_hash))
    
    conn.commit()
    print("Inserted admin user")
    
    cursor.execute('SELECT * FROM user')
    users = cursor.fetchall()
    print(f"Users in database: {users}")
    
    conn.close()
    
    print(f"Database created successfully: {db_path}")
    print(f"File size: {os.path.getsize(db_path)} bytes")

if __name__ == "__main__":
    setup_database() 