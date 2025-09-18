# Blood Report Disease Diagnosis App

This project is a Flask-based web application that uses machine learning and AI to predict blood disorders from CBC (Complete Blood Count) reports. Users can manually enter CBC values or upload a CBC report image, and the app will analyze the data using a Random Forest Classifier and Google Gemini AI Vision API.

---

## Features

- **Manual CBC Entry:** Enter WBC, RBC, Hemoglobin, Platelets, Neutrophils, Lymphocytes, Monocytes, Eosinophils, Basophils.
- **CBC Image Upload:** Upload a CBC report image; Gemini Vision extracts values automatically.
- **Disease Prediction:** Predicts disorders like Anemia, Leukocytosis, Thrombocytopenia, etc.
- **AI Explanation:** Gemini AI provides educational analysis and suggestions.
- **User Accounts:** Register, log in, and view your prediction history.
- **Save & View History:** Logged-in users can save and review past predictions.

---

## Installation & Running

### Prerequisites

- Python 3.x
- pip (Python package manager)
- SQLite (included by default)
- [Google Generative AI Python SDK](https://github.com/google/generative-ai-python)

### Steps

1. **Clone the repository:**
    ```sh
    git clone https://github.com/Rifaque/Blood-Report-Disease-Diagnosis-App.git
    cd Blood-Report-Disease-Diagnosis-App
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Ensure the training data is present:**
    - `Training.csv` should be in the project root.

4. **(Optional) Place CBC background image:**
    - Put `blo.jpg` in the `static` folder for the homepage background.

5. **Run the application:**
    ```sh
    python main.py
    ```
    - The app will run at [http://127.0.0.1:5000/](http://127.0.0.1:5000/).

---

## Usage

1. **Open your browser** and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/).
2. **Register or log in** to access full features.
3. **Enter CBC values manually** or **upload a CBC report image**.
4. **Submit** to get disease prediction and possible causes.
5. **View your history** from the dashboard.

---

## Technologies Used

- **Python** (Flask, Pandas, NumPy, scikit-learn)
- **Google Gemini AI** (Vision API for image extraction and analysis)
- **SQLite** (User and history database)
- **HTML/CSS/JS** (Frontend dashboard)

---

## Project Structure

```
Blood-Report-Disease-Diagnosis-App-main/
│
├── main.py                # Main Flask app
├── requirements.txt       # Python dependencies
├── Training.csv           # Training data for ML model
├── static/
│   └── blo.jpg            # Homepage background image
├── templates/
│   ├── index.html         # Homepage
│   ├── afterlogin.html    # Dashboard after login
│   ├── result.html        # Prediction result page
│   └── ...                # Other templates
└── README.md              # This file
```

---

## Contributing

1. Fork the repo and create a new branch.
2. Make your changes and commit.
3. Push and open a Pull Request.

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Contact

- **Authors:** Nazmus Shakib
- **Email:** bsse1452@iit.du.ac.bd

---

## Troubleshooting

- **Background image not loading?**  
  Ensure `blo.jpg` is in the `static` folder and referenced correctly in your templates.
- **Gemini API errors?**  
  Check your API key and internet connection.
- **Database issues?**  
  The app auto-creates SQLite tables on first run.

---

**Enjoy fast, AI-powered blood report analysis!**
