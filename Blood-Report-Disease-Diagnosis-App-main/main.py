from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sqlite3
from PIL import Image
import google.generativeai as genai
import os
from werkzeug.utils import secure_filename
import base64
import io
import time

app = Flask(__name__)
app.secret_key = 'Hub_Zero'

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure Gemini API and model settings
GEMINI_API_KEY = "AIzaSyBSwRxJzy8Qdh0eQqo_Umsd11wo7q7d64o"
MODEL_NAME = "gemini-1.5-flash"  # Use flash model consistently
GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.8,
    "max_output_tokens": 1024,
}

# Configure API
genai.configure(api_key=GEMINI_API_KEY)

# Initialize model
try:
    print(f"Initializing Gemini model: {MODEL_NAME}")
    model = genai.GenerativeModel(MODEL_NAME)
    
    # Test the model with a simple prompt to verify it's working
    test_response = model.generate_content(
        "Respond with just the word 'OK' if you're working.",
        generation_config=GENERATION_CONFIG
    )
    if not test_response or "OK" not in test_response.text:
        raise Exception("Model response validation failed")
        
    print(f"Gemini model configured and tested successfully: {MODEL_NAME}")
except Exception as e:
    print(f"Error configuring model: {str(e)}")
    model = None

import io
import json
import time
from PIL import Image

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function to check allowed file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_cbc_values(image_path):
    """
    Use Gemini Vision API to extract CBC values from the image
    """
    print(f"Processing image: {image_path}")
    
    # Read the image
    img = Image.open(image_path)
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize image to optimal size for flash model (maintaining aspect ratio)
    target_size = (512, 512)  # Flash model prefers smaller images
    img.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Save image as bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=75)  # Lower quality for smaller size
    img_byte_arr = img_byte_arr.getvalue()

    print("Image successfully converted to bytes")
    
    if model is None:
        return None

    # Prompt for Gemini
    prompt = """
    Please analyze this CBC (Complete Blood Count) report image and extract the following values:
    - WBC (White Blood Cells)
    - RBC (Red Blood Cells)
    - HGB (Hemoglobin)
    - PLT (Platelets)
    - NEUT (Neutrophils)
    - LYMPH (Lymphocytes)
    - MONO (Monocytes)
    - EO (Eosinophils)
    - BASO (Basophils)

    Return ONLY the numerical values in a JSON format like this:
    {
        "WBC": value,
        "RBC": value,
        "HGB": value,
        "PLT": value,
        "NEUT": value,
        "LYMPH": value,
        "MONO": value,
        "EO": value,
        "BASO": value
    }
    """

    # ✅ Correct way: pass image as {mime_type, data}
    response = model.generate_content(
        [prompt, {"mime_type": "image/jpeg", "data": img_byte_arr}],
        generation_config={
            "temperature": 0.1,
            "top_p": 0.8,
            "max_output_tokens": 2048,
        }
    )

    # Extract the JSON from response
    response_text = response.text
    start_idx = response_text.find('{')
    end_idx = response_text.rfind('}') + 1

    if start_idx == -1 or end_idx == 0:
        return None
    
    json_str = response_text[start_idx:end_idx]
    values = eval(json_str)  # ⚠ Unsafe, but quick for testing. Replace with json.loads later.

    return values


# Load the main dataset
df_main = pd.read_csv("Training.csv")

# Diseases
disease = {0: 'Anemia', 1: 'Polycythemia', 2: 'Leukocytosis', 3: 'Leukopenia', 4: 'Thrombocytopenia',
           5: 'Thrombocytosis', 6: 'Neutropenia', 7: 'Neutrophilia', 8: 'Lymphocytopenia', 9: 'Lymphocytosis',
           10: 'Monocytes high', 11: 'Eosinophil high', 12: 'Basophil high', 13: 'Normal'}

# Causes
Rea = {0: [' - Anemia due to blood loss \n'
            ' - Bone marrow disorders \n'
            ' - Nutritional deficiency \n'
            ' - Chronic Kidney disease  \n'
            ' - Chronic inflammatory disease \n'],
       1: ['- Dehydration, such as from severe diarrhea \n'
           '- tumours \n'
           '- Lung diseases \n'
           '- Smoking \n'
           '- Polycythemia vera \n'],
       2: ['- Infection \n'
           '- Leukemia \n'
           '- Inflammation \n'
           '- Stress, allergies, asthma \n'],
       3: ['- Viral infection \n'
           '- Severe bacterial infection \n'
           '- Bone marrow disorders \n'
           '- Autoimmune conditions \n'
           '- Lymphoma \n'
           '- Dietary deficiencies \n'],
       4: ['- Cancer, such as leukemia or lymphoma \n'
           '- Autoimmune diseases \n'
           '- Bacterial infection \n'
           '- Viral infection like dengue \n'
           '- Chemotherapy or radiation therapy \n'
           '- Certain drugs, such as nonsteroidal anti-inflammatory drugs (NSAIDs) \n'],
       5: ['- Bone marrow disorders \n'
           '- Essential thrombocythemia \n'
           '- Anemia \n'
           '- Infection \n'
           '- Surgical removal of the spleen \n'
           '- Polycythemia vera \n'
           '- Some types of leukemia \n'],
       6: ['- Severe infection \n'
           '- Immunodeficiency \n'
           '- Autoimmune disorders \n'
           '- Dietary deficiencies \n'
           '- Reaction to drugs \n'
           '- Bone marrow damage \n'],
       7: ['- Acute bacterial infections \n'
           '- Inflammation \n'
           '- Stress, Trauma \n'
           '- Certain leukemias \n'],
       8: ['- Autoimmune disorders \n'
           '- Infections \n'
           '- Bone marrow damage \n'
           '- Corticosteroids \n'],
       9: ['- Acute viral infections \n'
           '- Certain bacterial infections \n'
           '- Chronic inflammatory disorder \n'
           '- Lymphocytic leukemia, lymphoma \n'
           '- Acute stress \n'],
       10: ['- Chronic infections \n'
            '- Infection within the heart \n'
            '- Collagen vascular diseases \n'
            '- Monocytic or myelomonocytic leukemia \n'],
       11: ['- Asthma, allergies such as hay fever \n'
            '- Drug reactions \n'
            '- Parasitic infections \n'
            '- Inflammatory disorders \n'
            '- Some cancers, leukemias or lymphomas \n'],
       12: ['- Rare allergic reactions \n'
            '- Inflammation \n'
            '- Some leukemias \n'
            '- Uremia \n'],
       13: ['- Normal \n']}

# Function to train and predict using RandomForestClassifier
def rf(W, R, H, P, N, L, M, E, B):
    # Load the dataset and split into features and target variable
    x = df_main.drop(columns=['Disease'], axis=1)
    y = df_main['Disease']

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)

    # Initialize and train the RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(x_train, y_train)

    # Prepare input data as numpy array
    t = np.array([W, R, H, P, N, L, M, E, B]).reshape(1, -1)

    # Predict using the trained model
    res = clf.predict(t)[0]
    return res

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        W = float(request.form['WBC'])
        R = float(request.form['RBC'])
        H = float(request.form['HGB'])
        P = float(request.form['PLT'])
        N = float(request.form['NEUT'])
        L = float(request.form['LYMPH'])
        M = float(request.form['MONO'])
        E = float(request.form['EO'])
        B = float(request.form['BASO'])

        # Call the RandomForestClassifier function
        result = rf(W, R, H, P, N, L, M, E, B)

        # Get the cause of the disease
        cause = Rea[result][0]

        # Render result template
        return render_template('result.html', disease=disease[result], cause=cause)

    return render_template('index.html')

# Route for handling CBC image upload
@app.route('/upload_cbc', methods=['POST'])
def upload_cbc():
    try:
        print("Upload request received")
        
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            print("No filename")
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            print(f"Invalid file type: {file.filename}")
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload a PNG, JPG, or JPEG image.'})
        
        # Check if model is initialized
        if model is None:
            return jsonify({'success': False, 'error': 'AI model not properly initialized. Please try again in a few minutes.'})
        
        # Ensure upload directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
            print(f"Created upload directory: {app.config['UPLOAD_FOLDER']}")
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"File saved to: {filepath}")
        
        # Extract CBC values from the image
        values = extract_cbc_values(filepath)
        
        # Delete the temporary file
        os.remove(filepath)
        
        if values:
            return jsonify({'success': True, 'values': values})
        else:
            return jsonify({'success': False, 'error': 'Failed to extract values from image'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Route for saving the data without displaying the result
@app.route('/save_data', methods=['POST'])
def save_data():
    if request.method == 'POST':
        W = float(request.form['WBC'])
        R = float(request.form['RBC'])
        H = float(request.form['HGB'])
        P = float(request.form['PLT'])
        N = float(request.form['NEUT'])
        L = float(request.form['LYMPH'])
        M = float(request.form['MONO'])
        E = float(request.form['EO'])
        B = float(request.form['BASO'])
        
        print("Received data:", W, R, H, P, N, L, M, E, B)  # Add this line for debugging

        # Call the RandomForestClassifier function
        result = rf(W, R, H, P, N, L, M, E, B)

        # Get the user ID of the currently logged-in user
        user_id = session.get('user_id')

        # Save the data to the user history table
        insert_user_history(user_id, W, R, H, P, N, L, M, E, B, result)

        # Show success message
        return "Data saved successfully"

# Route to render the new template with upload functionality
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('afterlogin_new.html')

# Routes for login and register
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/result')
def result():
    # Your logic for handling the result page goes here
    return render_template('result.html')


@app.route('/afterlogin', methods=['GET', 'POST'])
def afterlogin():
    if request.method == 'POST':
        # Process form submission
        W = float(request.form['WBC'])
        R = float(request.form['RBC'])
        H = float(request.form['HGB'])
        P = float(request.form['PLT'])
        N = float(request.form['NEUT'])
        L = float(request.form['LYMPH'])
        M = float(request.form['MONO'])
        E = float(request.form['EO'])
        B = float(request.form['BASO'])

        # Call the RandomForestClassifier function
        result = rf(W, R, H, P, N, L, M, E, B)

        # Get the cause of the disease
        cause = Rea[result][0]

        # Render result template
        return render_template('resultafter.html', disease=disease[result], cause=cause)  # Redirect to the desired page after form submission
    else:
        # Handle GET request
        return render_template('afterlogin.html')


# Route for logging out
@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove the username from the session
    return redirect(url_for('index'))

# Function to create a connection to the database
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

# Function to create the user table if it doesn't exist
def create_user_table():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL
        );
    ''')
    conn.commit()
    conn.close()

# Function to create the user history table if it doesn't exist
def create_user_history_table():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS user_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            WBC REAL,
            RBC REAL,
            HGB REAL,
            PLT REAL,
            NEUT REAL,
            LYMPH REAL,
            MONO REAL,
            EO REAL,
            BASO REAL,
            result INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    ''')
    conn.commit()
    conn.close()

# Check if the user table exists, if not, create it
create_user_table()
create_user_history_table()

# Function to insert user history into the user_history table
def insert_user_history(user_id, WBC, RBC, HGB, PLT, NEUT, LYMPH, MONO, EO, BASO, result):
    conn = get_db_connection()
    conn.execute('''
        INSERT INTO user_history (user_id, WBC, RBC, HGB, PLT, NEUT, LYMPH, MONO, EO, BASO, result) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, WBC, RBC, HGB, PLT, NEUT, LYMPH, MONO, EO, BASO, result))
    conn.commit()
    conn.close()

# Function to get user history based on user ID
def get_user_history(user_id):
    conn = get_db_connection()
    cursor = conn.execute('''
        SELECT * FROM user_history WHERE user_id = ?
    ''', (user_id,))
    history = cursor.fetchall()
    conn.close()
    return history

# Function to get user ID based on email
def get_user_id(email):
    conn = get_db_connection()
    cursor = conn.execute('''
        SELECT id FROM users WHERE email = ?
    ''', (email,))
    user = cursor.fetchone()
    conn.close()
    return user['id'] if user else None

    
# Route for registering a new user
@app.route('/register', methods=['GET', 'POST'])
def register_user():
    if request.method == 'POST':
        # Get form data
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        age = request.form['age']
        gender = request.form['gender']
        
        # Insert user data into the database
        conn = get_db_connection()
        conn.execute('''
            INSERT INTO users (name, email, password, age, gender)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, email, password, age, gender))
        conn.commit()
        conn.close()
        
        # Redirect to login page after successful registration
        return redirect(url_for('login')) 
    
    # Render the registration form
    return render_template('register.html')

# Route for logging in
@app.route('/login', methods=['GET', 'POST'])
def login_user():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Validate user credentials
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            # Store user information in session
            session['user_id'] = user['id']
            session['username'] = user['name']
            # Redirect to home page or any other page after successful login
            return redirect(url_for('afterlogin'))
        else:
            error_message = "Invalid email or password. Please try again."
            return render_template('login.html', error_message=error_message)

    # Render the login form
    return render_template('login.html')

# Route for viewing user history

@app.route('/view_history')
def view_history():
    # Logic to retrieve user history from the database
    # Assuming you have a function to get user history data
    history_data = get_user_history(session.get('user_id'))
    return render_template('history.html', history=history_data)

@app.route('/gemini_analysis', methods=['POST'])
def gemini_analysis():
    if request.method == 'POST':
        # Get CBC values from the form
        wbc = float(request.form['WBC'])
        rbc = float(request.form['RBC'])
        hgb = float(request.form['HGB'])
        plt = float(request.form['PLT'])
        neut = float(request.form['NEUT'])
        lymph = float(request.form['LYMPH'])
        mono = float(request.form['MONO'])
        eo = float(request.form['EO'])
        baso = float(request.form['BASO'])

        # Create a prompt for Gemini
        prompt = f"""
            You are an AI health assistant. Analyze the following Complete Blood Count (CBC) results
            in an educational and probabilistic way (not as a medical diagnosis).

            CBC Results:
            - WBC (White Blood Cells): {wbc} × 10^9/L
            - RBC (Red Blood Cells): {rbc} × 10^12/L
            - Hemoglobin: {hgb} g/dL
            - Platelets: {plt} × 10^9/L
            - Neutrophils: {neut}%
            - Lymphocytes: {lymph}%
            - Monocytes: {mono}%
            - Eosinophils: {eo}%
            - Basophils: {baso}%

            Please provide:
            1. A short explanation of what each CBC component normally indicates and in bold word.
            2. Highlight any patterns in the values that could be associated with possible conditions
            (e.g., anemia, infection, inflammation).
            3. For each possible condition, give an **estimated confidence percentage**
            (e.g., "Anemia: 75% confidence") to show likelihood, not certainty.
            4. give me the suggestion based on cbc report.
            5. End with a disclaimer: "For accurate interpretation and diagnosis, consult a qualified doctor."
            """


    try:
        if model is None:
            raise Exception("Gemini model not properly initialized")

        print("\nAttempting to generate content:")
        print("1. Model status:", "Initialized" if model else "Not initialized")
        print("2. API Key status:", "Set" if GEMINI_API_KEY else "Not set")
        
        # Use the global configuration
        response = model.generate_content(
            prompt,
            generation_config=GENERATION_CONFIG
        )
        
        print("3. Response type:", type(response))
        print("4. Response attributes:", dir(response))
        
        # Try different ways to get the response
        if hasattr(response, 'text'):
            print("5. Using response.text")
            gemini_response = response.text
        elif hasattr(response, 'parts'):
            print("5. Using response.parts")
            gemini_response = ''.join(str(part.text) for part in response.parts)
        elif isinstance(response, (str, dict)):
            print("5. Response is string or dict")
            gemini_response = str(response)
        else:
            print("5. Unknown response format")
            gemini_response = "Unable to get a proper response from the AI model."
            
        # Format the response for better readability
        gemini_response = gemini_response.replace('\n', '<br>')

    except Exception as e:
        print(f"Gemini API Error: {str(e)}")  # Log the error
        gemini_response = f"Error getting AI analysis. Please try again later. Details: {str(e)}"

        # Render the template with all values and Gemini's analysis
    return render_template('gemini_analysis.html',
                            wbc=wbc, rbc=rbc, hgb=hgb, plt=plt,
                            neut=neut, lymph=lymph, mono=mono,
                            eo=eo, baso=baso,
                            gemini_response=gemini_response)


    
# Route to upload and extract CBC values from image
@app.route('/upload_cbc_image', methods=['POST'])
def upload_cbc_image():
    if 'cbc_image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['cbc_image']
    if file.filename == '':
        return jsonify({"error": "Empty file name"}), 400

    try:
        # Save the uploaded image temporarily
        filepath = os.path.join("uploads", file.filename)
        os.makedirs("uploads", exist_ok=True)
        file.save(filepath)

        # Open the image with PIL
        cbc_image = Image.open(filepath)

        # Prompt for Gemini
        prompt = """
        You are a medical data extractor.
        Extract the following values from this CBC (Complete Blood Count) report image:
        - WBC (White Blood Cells)
        - RBC (Red Blood Cells)
        - Hemoglobin (HGB)
        - Platelets (PLT)
        - Neutrophils (%)
        - Lymphocytes (%)
        - Monocytes (%)
        - Eosinophils (%)
        - Basophils (%)

        Return ONLY JSON format with keys:
        { "WBC": "...", "RBC": "...", "HGB": "...", "PLT": "...", 
          "NEUT": "...", "LYMPH": "...", "MONO": "...", "EO": "...", "BASO": "..." }
        """

        # Send request to Gemini using global configuration
        response = model.generate_content(
            [prompt, cbc_image],
            generation_config=GENERATION_CONFIG
        )

        # Extract response text and parse JSON
        try:
            extracted_text = response.text.strip()
            # Find JSON object in response
            start_idx = extracted_text.find('{')
            end_idx = extracted_text.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                extracted_text = extracted_text[start_idx:end_idx]
            return jsonify({"extracted_data": extracted_text})
        except Exception as e:
            return jsonify({"error": f"Failed to parse response: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
