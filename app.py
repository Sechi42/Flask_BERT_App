from flask import Flask, render_template, request
import pandas as pd
import pickle
import re
import torch
import transformers
import numpy as np
import math
from tqdm import tqdm

app = Flask(__name__)

# Cargar el pipeline de clasificación y el modelo BERT/tokenizer solo una vez
with open('modelo.pkl', 'rb') as archivo:
    modelo_cargado = pickle.load(archivo)

tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
model = transformers.BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Función para normalizar texto
def normalize_text(text):
    pattern = r"[^a-z\s]"
    return re.sub(pattern, " ", text.lower())

# Función para convertir texto a embeddings usando BERT
def BERT_text_to_embeddings(texts, max_length=512, batch_size=100, disable_progress_bar=False):
    ids_list, attention_mask_list = [], []

    # Normalización y tokenización de los textos
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        ids_list.append(encoding['input_ids'].squeeze().tolist())
        attention_mask_list.append(encoding['attention_mask'].squeeze().tolist())

    embeddings = []

    with torch.no_grad():
        model.eval()
        for i in tqdm(range(math.ceil(len(ids_list) / batch_size)), disable=disable_progress_bar):
            ids_batch = torch.LongTensor(ids_list[batch_size * i:batch_size * (i + 1)]).to(device)
            attention_mask_batch = torch.LongTensor(attention_mask_list[batch_size * i:batch_size * (i + 1)]).to(device)

            # Obtener las embeddings del modelo BERT
            batch_embeddings = model(input_ids=ids_batch, attention_mask=attention_mask_batch)
            embeddings.append(batch_embeddings.last_hidden_state[:, 0, :].detach().cpu().numpy())

    return np.concatenate(embeddings)

# Lista de columnas esperadas
REQUIRED_COLUMNS = ['tconst', 'title_type', 'primary_title', 'original_title', 
                    'start_year', 'end_year', 'runtime_minutes', 'is_adult', 
                    'genres', 'average_rating', 'votes', 'rating', 'sp', 
                    'ds_part', 'idx', 'review']

@app.route('/')
def home():
    return render_template("homepage.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        request_data = dict(request.form)
        print("Form data received:", request_data)
        
        # Normalizar el texto de la review si está presente
        if 'review' in request_data:
            review_text = normalize_text(request_data['review'])
            # Generar embeddings BERT para la review normalizada
            try:
                review_embeddings = BERT_text_to_embeddings([review_text])
                embeddings_df = pd.DataFrame(review_embeddings)
            except Exception as e:
                print(f"Error during BERT embeddings generation: {e}")
                return render_template('homepage.html', prediction="Error in BERT embeddings generation")
        else:
            return render_template('homepage.html', prediction="No review text provided")
        
        print(f"Embeddings shape: {embeddings_df.shape}")

        # Realizar la predicción con el pipeline cargado
        try:
            pred = modelo_cargado.predict(embeddings_df)
        except Exception as e:
            print(f"Prediction error: {e}")
            return render_template('homepage.html', prediction="Error in prediction")
        
        # Interpretar el resultado de la predicción
        result = "Positive Sentiment" if int(pred[0]) == 1 else "Negative Sentiment"

        return render_template('homepage.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
