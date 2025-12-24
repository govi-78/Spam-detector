"""
Database connection and schema setup
"""
import mysql.connector
from mysql.connector import Error
from config import Config
from datetime import datetime

class Database:
    def __init__(self):
        self.connection = None
        self.connect()
        
    def connect(self):
        """Establish MySQL database connection"""
        try:
            self.connection = mysql.connector.connect(
                host=Config.MYSQL_HOST,
                user=Config.MYSQL_USER,
                password=Config.MYSQL_PASSWORD,
                database=Config.MYSQL_DB,
                port=Config.MYSQL_PORT
            )
            if self.connection.is_connected():
                print("Successfully connected to MySQL database")
        except Error as e:
            print(f"Error connecting to MySQL: {e}")
            
    def create_tables(self):
        """Create necessary database tables"""
        try:
            cursor = self.connection.cursor()
            
            # Create users table
            create_users_table = """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                email VARCHAR(100) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL,
                INDEX idx_username (username),
                INDEX idx_email (email)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            
            # Create spam_logs table
            create_logs_table = """
            CREATE TABLE IF NOT EXISTS spam_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                input_type VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                result VARCHAR(10) NOT NULL,
                confidence FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
                INDEX idx_user_id (user_id),
                INDEX idx_input_type (input_type),
                INDEX idx_result (result),
                INDEX idx_created_at (created_at)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
            """
            
            cursor.execute(create_users_table)
            cursor.execute(create_logs_table)
            self.connection.commit()
            print("Tables created successfully")
            
        except Error as e:
            print(f"Error creating tables: {e}")
        finally:
            cursor.close()
            
    def insert_log(self, input_type, content, result, confidence=None, user_id=None):
        """Insert a spam detection log entry"""
        try:
            cursor = self.connection.cursor()
            query = """
            INSERT INTO spam_logs (user_id, input_type, content, result, confidence)
            VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (user_id, input_type, content, result, confidence))
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"Error inserting log: {e}")
            return None
        finally:
            cursor.close()
    
    def create_user(self, username, email, password_hash):
        """Create a new user"""
        try:
            cursor = self.connection.cursor()
            query = """
            INSERT INTO users (username, email, password_hash)
            VALUES (%s, %s, %s)
            """
            cursor.execute(query, (username, email, password_hash))
            self.connection.commit()
            return cursor.lastrowid
        except Error as e:
            print(f"Error creating user: {e}")
            return None
        finally:
            cursor.close()
    
    def get_user_by_username(self, username):
        """Get user by username"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT * FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            user = cursor.fetchone()
            return user
        except Error as e:
            print(f"Error fetching user: {e}")
            return None
        finally:
            cursor.close()
    
    def get_user_by_email(self, email):
        """Get user by email"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT * FROM users WHERE email = %s"
            cursor.execute(query, (email,))
            user = cursor.fetchone()
            return user
        except Error as e:
            print(f"Error fetching user: {e}")
            return None
        finally:
            cursor.close()
    
    def update_last_login(self, user_id):
        """Update user's last login timestamp"""
        try:
            cursor = self.connection.cursor()
            query = "UPDATE users SET last_login = NOW() WHERE id = %s"
            cursor.execute(query, (user_id,))
            self.connection.commit()
        except Error as e:
            print(f"Error updating last login: {e}")
        finally:
            cursor.close()
            
    def get_statistics(self):
        """Get spam detection statistics"""
        try:
            cursor = self.connection.cursor(dictionary=True)
            query = """
            SELECT 
                input_type,
                COUNT(*) as total_count,
                SUM(CASE WHEN result = 'Spam' THEN 1 ELSE 0 END) as spam_count,
                SUM(CASE WHEN result = 'Not Spam' THEN 1 ELSE 0 END) as not_spam_count
            FROM spam_logs
            GROUP BY input_type
            """
            cursor.execute(query)
            results = cursor.fetchall()
            return results
        except Error as e:
            print(f"Error fetching statistics: {e}")
            return []
        finally:
            cursor.close()
            
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("MySQL connection closed")


def init_database():
    """Initialize database and create tables"""
    try:
        # First, create the database if it doesn't exist
        connection = mysql.connector.connect(
            host=Config.MYSQL_HOST,
            user=Config.MYSQL_USER,
            password=Config.MYSQL_PASSWORD,
            port=Config.MYSQL_PORT
        )
        cursor = connection.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {Config.MYSQL_DB}")
        print(f"Database '{Config.MYSQL_DB}' created or already exists")
        cursor.close()
        connection.close()
        
        # Now create tables
        db = Database()
        db.create_tables()
        db.close()
        
    except Error as e:
        print(f"Error initializing database: {e}")


if __name__ == "__main__":
    init_database()
