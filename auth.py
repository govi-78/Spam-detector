"""
Authentication Module
Handles user registration, login, and password management
"""
import hashlib
import secrets
from database import Database

class Authentication:
    def __init__(self):
        self.db = Database()
    
    def hash_password(self, password, salt=None):
        """Hash password with salt using SHA-256"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}${pwd_hash}"
    
    def verify_password(self, password, stored_hash):
        """Verify password against stored hash"""
        try:
            salt, pwd_hash = stored_hash.split('$')
            new_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return new_hash == pwd_hash
        except:
            return False
    
    def register_user(self, username, email, password):
        """
        Register a new user
        Returns: (success: bool, message: str, user_id: int|None)
        """
        # Validate inputs
        if not username or not email or not password:
            return False, "All fields are required", None
        
        if len(username) < 3:
            return False, "Username must be at least 3 characters", None
        
        if len(password) < 6:
            return False, "Password must be at least 6 characters", None
        
        if '@' not in email or '.' not in email:
            return False, "Invalid email format", None
        
        # Check if username already exists
        existing_user = self.db.get_user_by_username(username)
        if existing_user:
            return False, "Username already exists", None
        
        # Check if email already exists
        existing_email = self.db.get_user_by_email(email)
        if existing_email:
            return False, "Email already registered", None
        
        # Hash password
        password_hash = self.hash_password(password)
        
        # Create user
        user_id = self.db.create_user(username, email, password_hash)
        
        if user_id:
            return True, "Registration successful", user_id
        else:
            return False, "Registration failed. Please try again", None
    
    def login_user(self, username, password):
        """
        Login user
        Returns: (success: bool, message: str, user: dict|None)
        """
        if not username or not password:
            return False, "Username and password are required", None
        
        # Get user from database
        user = self.db.get_user_by_username(username)
        
        if not user:
            return False, "Invalid username or password", None
        
        # Verify password
        if not self.verify_password(password, user['password_hash']):
            return False, "Invalid username or password", None
        
        # Update last login
        self.db.update_last_login(user['id'])
        
        # Remove password hash from user object
        user.pop('password_hash', None)
        
        return True, "Login successful", user
    
    def validate_session(self, user_id):
        """Validate if user session is valid"""
        if not user_id:
            return False
        
        # In a production system, you would validate session token
        # For now, just check if user exists
        return True
    
    def close(self):
        """Close database connection"""
        self.db.close()
