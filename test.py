# test_azure_openai.py
import os
from openai import AzureOpenAI

# === CONFIGURACIÓN ===
endpoint = "https://invuniandesai-2.openai.azure.com/"
model_name = "gpt-4.1"          # Nombre del deployment
deployment = "gpt-4.1"          # Igual que arriba
api_version = "2024-12-01-preview"  # Versión que usas

# === OBTENER API KEY DEL ENTORNO ===
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
if not subscription_key:
    raise ValueError("Error: AZURE_OPENAI_API_KEY no está configurada. Usa: export AZURE_OPENAI_API_KEY='tu-clave'")

print(f"Clave cargada: {subscription_key[:10]}...{subscription_key[-8:]} (longitud: {len(subscription_key)})")

# === CREAR CLIENTE ===
client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

print("Cliente creado. Enviando solicitud...")

# === ENVIAR MENSAJE ===
try:
    response = client.chat.completions.create(
        model=deployment,  # Usa el nombre del deployment aquí
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "I am going to Paris, what should I see?"
            }
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
    )

    # === MOSTRAR RESPUESTA ===
    print("\nRespuesta de Azure OpenAI:")
    print("-" * 50)
    print(response.choices[0].message.content)
    print("-" * 50)

except Exception as e:
    print(f"\nError al conectar con Azure OpenAI:")
    print(e)