# Feature Engineering on Healthcare Patient Records

## Project Overview
This project applies feature engineering techniques to healthcare datasets containing patient demographics, medical history, and lab test results.  
A trained Random Forest model predicts the encoded medical condition category based on the provided input features.

---

## Workflow
1. **Training the Model**
   - Open the Google Colab notebook.
   - Upload the healthcare dataset CSV.
   - Run all cells to preprocess, engineer features, and train the model.
   - Download `healthcare_model.pkl` and `scaler.pkl`.

2. **Running the Streamlit App**
   - Place `healthcare_model.pkl` and `scaler.pkl` in the same directory as `app.py`.
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Run the app:
     ```bash
     streamlit run app.py
     ```

3. **Usage**
   - Enter patient details in the web form.
   - Click **Predict Medical Condition** to view the encoded category output.

---

## Files
- `app.py` – Streamlit frontend  
- `requirements.txt` – Dependencies  
- `README.md` – Documentation  
- `healthcare_model.pkl`, `scaler.pkl` – Trained model files from Colab
