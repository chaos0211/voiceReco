import csv
import os

DB_FILE = 'voice_users.csv'

def init_db():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['username', 'embedding_path'])

def user_exists(username):
    if not os.path.exists(DB_FILE):
        return False
    with open(DB_FILE, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['username'] == username:
                return True
    return False

def add_or_replace_user(username, embedding_path):
    entries = []
    replaced = False

    if os.path.exists(DB_FILE):
        with open(DB_FILE, mode='r') as file:
            reader = csv.DictReader(file)
            entries = list(reader)

    with open(DB_FILE, mode='w', newline='') as file:
        fieldnames = ['username', 'embedding_path']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in entries:
            if row['username'] == username:
                writer.writerow({'username': username, 'embedding_path': embedding_path})
                replaced = True
            else:
                writer.writerow(row)

        if not replaced:
            writer.writerow({'username': username, 'embedding_path': embedding_path})

def get_user_embedding(username):
    if not os.path.exists(DB_FILE):
        return None
    with open(DB_FILE, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['username'] == username:
                return row['embedding_path']
    return None

def list_users():
    if not os.path.exists(DB_FILE):
        return []
    with open(DB_FILE, mode='r') as file:
        reader = csv.DictReader(file)
        return [row['username'] for row in reader]
