import requests
import json
import time
import argparse
from typing import List, Dict, Any, Set

BASE_URL = "https://library.oapen.org/rest/search"

# Términos ampliados relacionados con ciberseguridad (EN/ES)
QUERY_TERMS = [
    # Core
    "cybersecurity", "cyber security", "information security", "infosec", "security",
    # Domains/topics
    "network security", "application security", "software security", "secure software",
    "data protection", "privacy", "GDPR", "access control", "authentication", "authorization",
    "identity management", "zero trust", "threat intelligence", "vulnerability", "risk management",
    "incident response", "intrusion detection", "intrusion prevention", "forensics", "digital forensics",
    "malware", "ransomware", "phishing", "botnet", "DDoS",
    "cryptography", "encryption", "PKI", "TLS", "key management",
    "penetration testing", "pentest", "red team", "blue team", "SOC", "SIEM",
    "cloud security", "container security", "kubernetes security",
    "devsecops", "secure coding", "security engineering",
    "IOT security", "IoT security", "OT security", "ICS security", "SCADA",
    "mobile security", "endpoint security", "supply chain security",
    "blockchain security", "privacy enhancing technologies",
    "cybercrime", "cyber law", "cyber policy",
    # Frameworks/standards
    "NIST CSF", "NIST SP 800", "ISO 27001", "CIS controls",
    # Spanish equivalents
    "ciberseguridad", "seguridad informática", "seguridad de la información",
    "protección de datos", "privacidad", "criptografía", "gestión de identidades",
    "respuesta a incidentes", "forense digital", "gestión de vulnerabilidades",
]

def build_query(terms):
    # Cada término que tenga espacios va con comillas
    def q(t):
        return f'"{t}"' if " " in t else t
    return " OR ".join(q(t) for t in terms)

def fetch_all(limit=50, max_pages=10000, start_offset=0, checkpoint_path="cybersecurity_books.json"):
    headers = {
        "Accept": "application/json",
        "User-Agent": "Datos-Tesis/1.0 (+oapen-fetch)"
    }
    query = build_query(QUERY_TERMS)
    all_items: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    # Cargar progreso previo si existe
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            prev = json.load(f)
            if isinstance(prev, list):
                for it in prev:
                    uid = it.get("uuid") or it.get("handle")
                    if uid and uid not in seen:
                        seen.add(uid)
                        all_items.append(it)
                print(f"Reanudando desde checkpoint: {len(all_items)} items previos")
    except Exception:
        pass

    offset = start_offset if start_offset else (len(all_items) // limit) * limit
    page = 0
    while page < max_pages:
        page += 1
        params = {
            "query": query,
            "expand": "metadata,bitstreams",
            "offset": offset,
            "limit": limit,
        }
        # Reintentos con backoff simple
        attempts = 0
        while True:
            attempts += 1
            try:
                resp = requests.get(BASE_URL, headers=headers, params=params, timeout=90)
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                if attempts >= 3:
                    print(f"Error página offset={offset}: {e}")
                    return all_items
                wait_s = attempts * 5
                print(f"Aviso: fallo offset={offset} intento {attempts}, reintentando en {wait_s}s...")
                time.sleep(wait_s)

        if not isinstance(data, list):
            print("Estructura inesperada (no es lista), deteniendo.")
            break
        if not data:
            break
        added = 0
        for it in data:
            uid = it.get("uuid") or it.get("handle")
            if uid and uid not in seen:
                seen.add(uid)
                all_items.append(it)
                added += 1
        print(f"Página {page} (offset {offset}): +{added} únicos (total: {len(all_items)})")

        # Guardar checkpoint por página
        try:
            with open(checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(all_items, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"No se pudo guardar checkpoint: {e}")

        if len(data) < limit:
            break
        offset += limit
        # Pausa amigable para no saturar el servicio
        time.sleep(1.0)
    return all_items

def main():
    parser = argparse.ArgumentParser(description="Fetch OAPEN cybersecurity-related items with pagination")
    parser.add_argument("--limit", type=int, default=50, help="Resultados por página (default: 50)")
    parser.add_argument("--pages", type=int, default=10000, help="Número máximo de páginas a traer (default: 10000)")
    parser.add_argument("--start-offset", type=int, default=0, help="Offset inicial (default: 0, o deducido del checkpoint)")
    args = parser.parse_args()

    items = fetch_all(limit=args.limit, max_pages=args.pages, start_offset=args.start_offset)
    out_path = "cybersecurity_books.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
    print(f"Guardado {len(items)} items en '{out_path}'")

if __name__ == "__main__":
    main()