import csv
import os

DB_FILE = 'voice_users.csv'


class UserDatabase:
    def __init__(self, db_file='voice_users.csv'):
        self.db_file = db_file
        self._init_db()

    def _init_db(self):
        if not os.path.exists(self.db_file):
            with open(self.db_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['username', 'embedding'])

    def user_exists(self, username):
        if not os.path.exists(self.db_file):
            return False
        with open(self.db_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['username'] == username:
                    return True
        return False

    def add_or_replace_user(self, username, embedding_path):
        entries = []
        replaced = False

        if os.path.exists(self.db_file):
            with open(self.db_file, mode='r') as file:
                reader = csv.DictReader(file)
                entries = list(reader)

        with open(self.db_file, mode='w', newline='') as file:
            fieldnames = ['username', 'embedding']
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

    def get_user_embedding(self, username):
        if not os.path.exists(self.db_file):
            return None
        with open(self.db_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['username'] == username:
                    return row['embedding_path']
        return None

    def list_users(self):
        if not os.path.exists(self.db_file):
            return []
        with open(self.db_file, mode='r') as file:
            reader = csv.DictReader(file)
            return [row['username'] for row in reader]
