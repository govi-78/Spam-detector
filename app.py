"""
Flask Application - Spam Detection System Backend
With User Authentication
"""
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS 
from model import SpamDetector
from database import Database
from auth import Authentication
from config import Config
import os
from functools import wraps

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = Config.SECRET_KEY  # Required for session management

# Session configuration
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['SESSION_COOKIE_HTTPONLY'] = True

# Configure CORS to allow credentials
CORS(app, supports_credentials=True, origins=['http://localhost:5000'], 
     methods=["GET", "POST", "PUT", "DELETE"], 
     allow_headers=["Content-Type", "Authorization"],
     expose_headers=["Access-Control-Allow-Credentials"])

# Initialize spam detector
detector = SpamDetector()

# Initialize database
db = Database()

# Initialize authentication
auth = Authentication()


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required', 'redirect': '/login'}), 401
        return f(*args, **kwargs)
    return decorated_function


@app.route('/')
def index():
    """Render landing page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/dashboard')
@login_required
def dashboard():
    """Render main dashboard (protected)"""
    return render_template('dashboard.html', username=session.get('username'))

@app.route('/login')
def login():
    """Render login page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')


@app.route('/register')
def register():
    """Render registration page"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('register.html')


@app.route('/api/register', methods=['POST'])
def api_register():
    """User registration endpoint"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        password = data.get('password', '')
        
        success, message, user_id = auth.register_user(username, email, password)
        
        if success:
            return jsonify({
                'success': True,
                'message': message
            })
        else:
            return jsonify({
                'success': False,
                'message': message
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/login', methods=['POST'])
def api_login():
    """User login endpoint"""
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        success, message, user = auth.login_user(username, password)
        
        if success and user is not None and isinstance(user, dict):
            # Create session
            session.clear()  # Clear any existing session
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['email'] = user['email']
            session.modified = True  # Ensure session is saved
            
            response = jsonify({
                'success': True,
                'message': message,
                'user': {
                    'username': user['username'],
                    'email': user['email']
                }
            })
            return response
        else:
            return jsonify({
                'success': False,
                'message': message or 'Invalid username or password'
            }), 401
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/logout', methods=['POST'])
def api_logout():
    """User logout endpoint"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/detect/sms', methods=['POST'])
@login_required
def detect_sms():
    """Detect spam in SMS/Text messages"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Predict
        result = detector.predict_sms(text)
        
        # Log to database with user_id
        db.insert_log(
            input_type='SMS',
            content=text,
            result=result['result'],
            confidence=result['confidence'],
            user_id=session.get('user_id')
        )
        
        return jsonify({
            'success': True,
            'result': result['result'],
            'confidence': round(result['confidence'] * 100, 2),
            'type': 'SMS'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect/url', methods=['POST'])
@login_required
def detect_url():
    """Detect spam URLs"""
    try:
        data = request.get_json()
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        # Predict
        result = detector.predict_url(url)
        
        # Log to database with user_id
        db.insert_log(
            input_type='URL',
            content=url,
            result=result['result'],
            confidence=result['confidence'],
            user_id=session.get('user_id')
        )
        
        return jsonify({
            'success': True,
            'result': result['result'],
            'confidence': round(result['confidence'] * 100, 2),
            'type': 'URL'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect/email', methods=['POST'])
@login_required
def detect_email():
    """Detect spam in emails"""
    try:
        data = request.get_json()
        subject = data.get('subject', '').strip()
        body = data.get('body', '').strip()
        
        if not subject and not body:
            return jsonify({'error': 'Email subject or body is required'}), 400
        
        # Predict
        result = detector.predict_email(subject, body)
        
        # Log to database with user_id
        email_content = f"Subject: {subject}\n\nBody: {body}"
        db.insert_log(
            input_type='Email',
            content=email_content,
            result=result['result'],
            confidence=result['confidence'],
            user_id=session.get('user_id')
        )
        
        return jsonify({
            'success': True,
            'result': result['result'],
            'confidence': round(result['confidence'] * 100, 2),
            'type': 'Email'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect/comment', methods=['POST'])
@login_required
def detect_comment():
    """Detect spam in social media comments"""
    try:
        data = request.get_json()
        comment = data.get('comment', '').strip()
        
        if not comment:
            return jsonify({'error': 'Comment is required'}), 400
        
        # Predict
        result = detector.predict_comment(comment)
        
        # Log to database with user_id
        db.insert_log(
            input_type='Comment',
            content=comment,
            result=result['result'],
            confidence=result['confidence'],
            user_id=session.get('user_id')
        )
        
        return jsonify({
            'success': True,
            'result': result['result'],
            'confidence': round(result['confidence'] * 100, 2),
            'type': 'Social Media Comment'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/statistics', methods=['GET'])
@login_required
def get_statistics():
    """Get spam detection statistics"""
    try:
        stats = db.get_statistics()
        return jsonify({
            'success': True,
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("=" * 60)
    print("🚀 Starting Finexo with Authentication")
    print("=" * 60)
    print("\nAuthentication Flow:")
    print("  1. Register: http://localhost:5000/register")
    print("  2. Login: http://localhost:5000/login")
    print("  3. Dashboard (protected): http://localhost:5000/dashboard")
    print("\nAPI Endpoints (all require authentication):")
    print("  - POST /detect/sms")
    print("  - POST /detect/url")
    print("  - POST /detect/email")
    print("  - POST /detect/comment")
    print("  - GET  /statistics")
    print("\nAuthentication Endpoints:")
    print("  - POST /api/register")
    print("  - POST /api/login")
    print("  - POST /api/logout")
    print("\n" + "=" * 60 + "\n")
    
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5000)
