from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import mysql.connector
import os
import cv2
import numpy as np
from PIL import Image
from skimage import filters, segmentation, color
import uuid
from werkzeug.utils import secure_filename
from groq import Groq
import base64
import json
import re

app = Flask(__name__)
app.secret_key = 'fruits_vegetable_adulteration_secret_key'

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'fruitsvegetablesadult_2025'
}

# Upload configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Groq client
groq_client = Groq(api_key="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")

def encode_image(image_path):
    """Encode image to base64 for Groq API"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def parse_text_response(response_text):
    """Parse non-JSON text response and extract information"""
    
    # Initialize default values
    result = {
        "type": "Unknown",
        "quality": "Unable to determine",
        "confidence": 50,
        "adulteration_signs": "Analysis incomplete",
        "safety_assessment": "Could not parse response"
    }
    
    try:
        # Try to extract type/fruit information
        type_patterns = [
            r'type["\']?\s*:\s*["\']?([^,"\'}\n]+)',
            r'fruit["\']?\s*:\s*["\']?([^,"\'}\n]+)',
            r'vegetable["\']?\s*:\s*["\']?([^,"\'}\n]+)',
            r'identified["\']?\s*:\s*["\']?([^,"\'}\n]+)'
        ]
        
        for pattern in type_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                result["type"] = match.group(1).strip()
                break
        
        # Try to extract quality
        quality_patterns = [
            r'quality["\']?\s*:\s*["\']?([^,"\'}\n]+)',
            r'assessment["\']?\s*:\s*["\']?([^,"\'}\n]+)'
        ]
        
        for pattern in quality_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                result["quality"] = match.group(1).strip()
                break
        
        # Try to extract confidence
        confidence_patterns = [
            r'confidence["\']?\s*:\s*["\']?(\d+)',
            r'(\d+)%',
            r'(\d+)\s*percent'
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                result["confidence"] = int(match.group(1))
                break
        
        # Extract adulteration signs
        adulteration_patterns = [
            r'adulteration_signs["\']?\s*:\s*["\']?([^"\'}\n]+)',
            r'signs["\']?\s*:\s*["\']?([^"\'}\n]+)',
            r'adulteration["\']?\s*:\s*["\']?([^"\'}\n]+)'
        ]
        
        for pattern in adulteration_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                result["adulteration_signs"] = match.group(1).strip()
                break
        
        # Extract safety assessment
        safety_patterns = [
            r'safety_assessment["\']?\s*:\s*["\']?([^"\'}\n]+)',
            r'safety["\']?\s*:\s*["\']?([^"\'}\n]+)',
            r'safe["\']?\s*:\s*["\']?([^"\'}\n]+)'
        ]
        
        for pattern in safety_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                result["safety_assessment"] = match.group(1).strip()
                break
        
        # If we couldn't extract much, use the full response as safety assessment
        if result["safety_assessment"] == "Could not parse response":
            result["safety_assessment"] = response_text[:200] + "..." if len(response_text) > 200 else response_text
            
    except Exception as e:
        print(f"Error parsing text response: {e}")
        result["safety_assessment"] = f"Parse error: {str(e)}"
    
    return result

def analyze_with_groq(image_path):
    """Analyze image using Groq API for adulteration detection"""
    try:
        base64_image = encode_image(image_path)
        
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": """Analyze this image of fruits or vegetables for adulteration. Please provide:
1. Type of fruit/vegetable identified
2. Quality assessment (Fresh/Good/Poor/Adulterated)
3. Confidence percentage (0-100%)
4. Signs of adulteration if any (artificial coloring, wax coating, chemical treatment, etc.)
5. Overall safety assessment for consumption

Provide the response in JSON format with keys: type, quality, confidence, adulteration_signs, safety_assessment"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        
        response_content = chat_completion.choices[0].message.content
        
        # Try to parse JSON response
        try:
            # First, try to parse as direct JSON
            analysis_result = json.loads(response_content)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON from markdown code blocks
            try:
                # Look for JSON within code blocks (```json...``` or ```...```)
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    analysis_result = json.loads(json_str)
                else:
                    # Try to find JSON without code blocks
                    json_match = re.search(r'(\{.*?\})', response_content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                        analysis_result = json.loads(json_str)
                    else:
                        raise json.JSONDecodeError("No JSON found", response_content, 0)
            except (json.JSONDecodeError, AttributeError):
                # If JSON parsing still fails, parse the text manually
                analysis_result = parse_text_response(response_content)
        
        return analysis_result
        
    except Exception as e:
        print(f"Groq API error: {e}")
        return {
            "type": "Unknown",
            "quality": "Analysis failed",
            "confidence": 0,
            "adulteration_signs": "API error occurred",
            "safety_assessment": f"Error: {str(e)}"
        }

def get_db_connection():
    """Get database connection"""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_user_upload_folder(user_id):
    """Create user-specific upload folder"""
    user_folder = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
    return user_folder

def process_image(image_path):
    """Process image to create binary, threshold, grayscale, and segmentation versions"""
    # Read the original image
    img = cv2.imread(image_path)
    original_img = img.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Binary image (using Otsu's thresholding)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Threshold image (using adaptive threshold)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Enhanced segmentation using watershed
    # Noise removal
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    # Apply watershed
    markers = cv2.watershed(img, markers)
    
    # Create a better segmentation visualization
    segmented_img = original_img.copy()
    segmented_img[markers == -1] = [0, 255, 255]  # Yellow boundaries
    
    # Create a colored segmentation mask
    segmentation_mask = np.zeros_like(original_img)
    unique_markers = np.unique(markers)
    colors = np.random.randint(0, 255, size=(len(unique_markers), 3))
    
    for i, marker in enumerate(unique_markers):
        if marker > 0:  # Skip background
            segmentation_mask[markers == marker] = colors[i]
    
    # Blend original image with segmentation mask
    blended_segmentation = cv2.addWeighted(original_img, 0.6, segmentation_mask, 0.4, 0)
    
    # Save processed images
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    folder = os.path.dirname(image_path)
    
    binary_path = os.path.join(folder, f"{base_name}_binary.jpg")
    threshold_path = os.path.join(folder, f"{base_name}_threshold.jpg")
    grayscale_path = os.path.join(folder, f"{base_name}_grayscale.jpg")
    segmentation_path = os.path.join(folder, f"{base_name}_segmentation.jpg")
    
    cv2.imwrite(binary_path, binary)
    cv2.imwrite(threshold_path, threshold)
    cv2.imwrite(grayscale_path, gray)
    cv2.imwrite(segmentation_path, blended_segmentation)
    
    return {
        'binary': binary_path,
        'threshold': threshold_path,
        'grayscale': grayscale_path,
        'segmentation': segmentation_path
    }

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Basic validation
        if not username or not email or not password:
            flash('All fields are required!', 'error')
            return render_template('register.html')
        
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                
                # Check if user already exists
                cursor.execute("SELECT id FROM users WHERE username = %s OR email = %s", (username, email))
                if cursor.fetchone():
                    flash('Username or email already exists!', 'error')
                    return render_template('register.html')
                
                # Insert new user
                cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", 
                             (username, email, password))
                connection.commit()
                flash('Registration successful! Please login.', 'success')
                return redirect(url_for('login'))
                
            except mysql.connector.Error as err:
                flash(f'Database error: {err}', 'error')
            finally:
                cursor.close()
                connection.close()
        else:
            flash('Database connection failed!', 'error')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if not username or not password:
            flash('Username and password are required!', 'error')
            return render_template('login.html')
        
        connection = get_db_connection()
        if connection:
            try:
                cursor = connection.cursor()
                cursor.execute("SELECT id, username FROM users WHERE username = %s AND password = %s", 
                             (username, password))
                user = cursor.fetchone()
                
                if user:
                    session['user_id'] = user[0]
                    session['username'] = user[1]
                    flash('Login successful!', 'success')
                    return redirect(url_for('upload'))
                else:
                    flash('Invalid username or password!', 'error')
                    
            except mysql.connector.Error as err:
                flash(f'Database error: {err}', 'error')
            finally:
                cursor.close()
                connection.close()
        else:
            flash('Database connection failed!', 'error')
    
    return render_template('login.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload page - requires login"""
    if 'user_id' not in session:
        flash('Please login to access this page!', 'error')
        return redirect(url_for('login'))
    
    processed_images = None
    original_image = None
    analysis_result = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected!', 'error')
            return render_template('upload.html')
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected!', 'error')
            return render_template('upload.html')
        
        if file and allowed_file(file.filename):
            # Create user folder
            user_folder = create_user_upload_folder(session['user_id'])
            
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            file_path = os.path.join(user_folder, unique_filename)
            
            # Save the file
            file.save(file_path)
            
            # Process the image
            try:
                processed_images = process_image(file_path)
                original_image = file_path
                
                # Analyze with Groq API
                analysis_result = analyze_with_groq(file_path)
                
                flash('Image uploaded and processed successfully!', 'success')

                # Save to database
                connection = get_db_connection()
                if connection:
                    try:
                        cursor = connection.cursor()
                        sql = """
                            INSERT INTO uploads (user_id, filename, original_path, binary_path, threshold_path, 
                                               grayscale_path, segmentation_path, fruit_type, quality_assessment, 
                                               confidence_percentage, adulteration_signs, safety_assessment)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """
                        values = (
                            session['user_id'],
                            unique_filename,
                            original_image,
                            processed_images['binary'],
                            processed_images['threshold'],
                            processed_images['grayscale'],
                            processed_images['segmentation'],
                            analysis_result.get('type', 'Unknown'),
                            analysis_result.get('quality', 'Unknown'),
                            analysis_result.get('confidence', 0),
                            analysis_result.get('adulteration_signs', ''),
                            analysis_result.get('safety_assessment', '')
                        )
                        cursor.execute(sql, values)
                        connection.commit()
                    except mysql.connector.Error as err:
                        flash(f'Database error: {err}', 'error')
                    finally:
                        cursor.close()
                        connection.close()
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'error')
        else:
            flash('Invalid file type! Please upload PNG, JPG, JPEG, or GIF files.', 'error')
    
    return render_template('upload.html', 
                         processed_images=processed_images, 
                         original_image=original_image,
                         analysis_result=analysis_result)

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out successfully!', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Create upload directory if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)
