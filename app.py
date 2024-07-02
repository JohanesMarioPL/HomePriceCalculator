from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load all models and other components
model_dt = joblib.load('model_dt.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Fitur yang diharapkan dalam urutan yang benar
expected_features = ['jumlah kamar tidur', 'jumlah kamar mandi', 'luas tanah (m2)', 'luas bangunan (m2)',
                     'carport (mobil)', 'pasokan listrik (watt)', 'Kabupaten/Kota', 'kecamatan', 'kelurahan',
                     'keamanan (ada/tidak)', 'taman (ada/tidak)', 'jarak dengan rumah sakit terdekat (km)',
                     'jarak dengan sekolah terdekat (km)', 'jarak dengan tol terdekat (km)']

def encode_json(json_data):
    """
    Function to encode JSON data into a DataFrame with the correct format for the model.
    """
    df = pd.DataFrame(json_data, index=[0])
    
    # Rename columns to match expected features
    rename_map = {
        'jumlah_kamar_tidur': 'jumlah kamar tidur',
        'jumlah_kamar_mandi': 'jumlah kamar mandi',
        'luas_tanah': 'luas tanah (m2)',
        'luas_bangunan': 'luas bangunan (m2)',
        'carport': 'carport (mobil)',
        'pasokan_listrik': 'pasokan listrik (watt)',
        'keamanan': 'keamanan (ada/tidak)',
        'taman': 'taman (ada/tidak)',
        'jarak_rumah_sakit': 'jarak dengan rumah sakit terdekat (km)',
        'jarak_sekolah': 'jarak dengan sekolah terdekat (km)',
        'jarak_tol': 'jarak dengan tol terdekat (km)'
    }
    
    df.rename(columns=rename_map, inplace=True)
    
    # Add default value for Kabupaten/Kota if not present
    if 'Kabupaten/Kota' not in df.columns:
        df['Kabupaten/Kota'] = 'Default Value'  # ganti dengan nilai default yang sesuai
    
    # Encode categorical columns
    for column in ['Kabupaten/Kota', 'kecamatan', 'kelurahan', 'keamanan (ada/tidak)', 'taman (ada/tidak)']:
        if column in df.columns:
            try:
                df[column] = label_encoders[column].transform(df[column])
            except ValueError:
                df[column] = -1  # atau nilai yang sesuai jika label tidak dikenali
    
    # Convert numerical columns to float
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    # Ensure the DataFrame has the expected order of features
    df = df[expected_features]
    
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(f"Received data: {data}")

        df = encode_json(data)

        # Memastikan semua kolom ada
        for column in expected_features:
            if column not in df.columns:
                return jsonify({'error': f'Missing column: {column}'}), 400

        # Mengubah tipe data kolom numerik menjadi float
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')

        # Memastikan urutan kolom sesuai dengan urutan yang diharapkan
        df = df[expected_features]

        # Scale data
        X_scaled = scaler.transform(df)

        # PCA
        X_pca = pca.transform(X_scaled)

        # Prediksi dari model
        pred_dt = model_dt.predict(X_pca)
        pred_dt = np.expm1(pred_dt)  # Transformasi balik prediksi

        print(f"Predicted price: {pred_dt[0]}")

        return jsonify({'predicted_price': pred_dt[0]})
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
