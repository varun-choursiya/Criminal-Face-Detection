from pathlib import Path
import sqlite3

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / 'criminal_database.db'

def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS criminals
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT NOT NULL,
                  crime TEXT NOT NULL,
                  age INTEGER,
                  location TEXT,
                  face_data BLOB NOT NULL,
                  image_path TEXT,
                  date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS detections
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  criminal_id INTEGER,
                  confidence REAL,
                  detection_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                  location TEXT,
                  FOREIGN KEY(criminal_id) REFERENCES criminals(id))''')
    conn.commit()
    conn.close()
    print(f"Initialized database at {DATABASE}")

if __name__ == '__main__':
    init_db()
