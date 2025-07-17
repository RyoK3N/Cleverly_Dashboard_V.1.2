import os
import sys
from pathlib import Path

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash

app = Flask(__name__, template_folder="pages")
app.config.update(
    SECRET_KEY='your-secret-key',
    SQLALCHEMY_DATABASE_URI='sqlite:///database.db',
    SQLALCHEMY_TRACK_MODIFICATIONS=False
)

db = SQLAlchemy(app)

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

def init_database():
    print("Initializing database...")
    
    with app.app_context():
        db.drop_all()
        print("Dropped existing tables")
        
        db.create_all()
        print("Created new tables")
        
        existing_user = User.query.filter_by(username='Dharmil').first()
        if existing_user:
            print("Removing existing user...")
            db.session.delete(existing_user)
            db.session.commit()
        
        password_hash = generate_password_hash('Clev@2025')
        print(f"Generated password hash: {password_hash[:50]}...")
        
        user = User(
            username='Dharmil',
            password_hash=password_hash
        )
        
        db.session.add(user)
        db.session.commit()
        print("Created admin user successfully!")
        
        all_users = User.query.all()
        print(f"Total users in database: {len(all_users)}")
        for u in all_users:
            print(f"User: {u.username}, Hash: {u.password_hash[:50]}...")

if __name__ == "__main__":
    init_database() 