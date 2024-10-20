import streamlit as st
import pandas as pd
import os
import pickle
import re
import torch
import transformers
import numpy as np
import math
from tqdm import tqdm

# Cargar el pipeline de clasificación solo una vez
with open('modelo.pkl', 'rb') as archivo:
    modelo_cargado = pickle.load(archivo)

# Función para normalizar texto
def normalize_text(text):
    pattern = r"[^a-z\s]"
    return re.sub(pattern, " ", text.lower())

# Función para convertir texto a embeddings usando BERT
def BERT_text_to_embeddings(texts, max_length=512, batch_size=10, disable_progress_bar=False):
    # Forzar la descarga cuando sea necesario y evitar advertencias futuras
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', force_download=True)
    model = transformers.BertModel.from_pretrained('bert-base-uncased', force_download=True)
    
    ids_list = []
    attention_mask_list = []

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    embeddings = []

    # Procesamiento en lotes, reducir tamaño de batch para evitar problemas de memoria
    for i in tqdm(range(math.ceil(len(ids_list) / batch_size)), disable=disable_progress_bar):
        ids_batch = torch.LongTensor(ids_list[batch_size * i:batch_size * (i + 1)]).to(device)
        attention_mask_batch = torch.LongTensor(attention_mask_list[batch_size * i:batch_size * (i + 1)]).to(device)

        with torch.no_grad():
            model.eval()
            batch_embeddings = model(input_ids=ids_batch, attention_mask=attention_mask_batch)
        
        embeddings.append(batch_embeddings[0][:, 0, :].detach().cpu().numpy())
        
        # Limpiar memoria innecesaria
        del batch_embeddings
        torch.cuda.empty_cache()

    return np.concatenate(embeddings)

classification_pipeline = modelo_cargado

# Título de la app
st.title("Movie Review Sentiment Analysis")

# Formulario de entrada de la review
review_text = st.text_area("Enter a movie review:")

# Botón para hacer la predicción
if st.button('Predict Sentiment'):
    if review_text:
        # Normalizar el texto
        review_text_normalized = normalize_text(review_text)
        # Generar embeddings BERT para la review normalizada
        try:
            review_embeddings = BERT_text_to_embeddings([review_text_normalized])
            # Convertir los embeddings a un DataFrame para ser procesados
            embeddings_df = pd.DataFrame(review_embeddings)
        except Exception as e:
            st.error(f"Error during BERT embeddings generation: {e}")
        else:
            # Realizar la predicción con el pipeline cargado (ya en memoria)
            try:
                pred = classification_pipeline.predict(embeddings_df)
                result = "Positive Sentiment" if int(pred[0]) == 1 else "Negative Sentiment"
                st.success(f"Prediction: {result}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter a review.")
