import os
import json
from typing import List, Dict, Any


INPUT_JSON = os.path.abspath("cybersecurity_books.json")
OUTPUT_JSON = os.path.abspath("cybersecurity_books_filtered.json")

# Palabras clave fuertes para ciberseguridad (coincidencia en título/subjects/abstract)
STRONG_KEYWORDS = [
    # EN
    "cybersecurity", "cyber security", "information security", "infosec",
    "network security", "application security", "data protection", "privacy",
    "cryptography", "encryption", "key management", "PKI", "TLS",
    "malware", "ransomware", "phishing", "intrusion", "forensics",
    "incident response", "threat intelligence", "vulnerability", "zero trust",
    "devsecops", "siem", "soc", "pentest", "penetration testing",
    "iot security", "ics security", "ot security", "kubernetes security", "cloud security",
    # ES
    "ciberseguridad", "seguridad informática", "seguridad de la información",
    "protección de datos", "criptografía", "privacidad", "forense digital",
]

# Palabras a excluir si aparecen fuertemente fuera de contexto (falsos positivos típicos)
EXCLUDE_HINTS = [
    "law", "policy", "economics", "sociology", "history", "philosophy"
]


def text_from_metadata(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(item.get("name") or "")
    for m in item.get("metadata") or []:
        key = m.get("key") or ""
        val = m.get("value") or ""
        if key in ("dc.title", "dc.description.abstract", "dc.subject.other", "dc.subject", "dc.subject.classification"):
            parts.append(val)
    return " \n ".join(parts).lower()


def is_cybersecurity_item(text: str) -> bool:
    # Debe contener al menos una keyword fuerte
    if not any(k in text for k in STRONG_KEYWORDS):
        return False
    return True


def filter_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for it in items:
        text = text_from_metadata(it)
        if is_cybersecurity_item(text):
            filtered.append(it)
    return filtered


def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("El JSON de entrada debe ser una lista")

    filtered = filter_items(data)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=4, ensure_ascii=False)

    total_pdfs = sum(1 for it in filtered for b in it.get("bitstreams", []) if b.get("mimeType") == "application/pdf")
    print(f"Items filtrados: {len(filtered)} | PDFs disponibles en filtrados: {total_pdfs}")


if __name__ == "__main__":
    main()


