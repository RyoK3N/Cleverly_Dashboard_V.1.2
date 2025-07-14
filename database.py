
import os
import psycopg2
from werkzeug.security import generate_password_hash

def init_db():
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    c = conn.cursor()
    
    # Drop existing table if any
    c.execute('DROP TABLE IF EXISTS admin_users')
    
    # Create admin_users table
    c.execute('''
        CREATE TABLE admin_users (
            id TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Add admin user with new credentials
    admin_id = "Dharmil"  # New username
    admin_password = "Clev@2025"  # New password
    password_hash = generate_password_hash(admin_password)
    
    c.execute("INSERT INTO admin_users (id, password_hash) VALUES (%s, %s)",
              (admin_id, password_hash))
    
    conn.commit()
    conn.close()
    print("Database initialized successfully!")

if __name__ == "__main__":
    init_db()
