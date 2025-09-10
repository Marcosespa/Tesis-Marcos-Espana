import os
import csv
import json
import shutil
from typing import Dict, Any, List, Optional, Tuple


INPUT_JSON = os.path.abspath("cybersecurity_books.json")
PDF_DIR = os.path.abspath("OAPEN_PDFs")
OUT_DIR_CYBER = os.path.join(PDF_DIR, "ciberseguridad")
OUT_DIR_OTHERS = os.path.join(PDF_DIR, "otros")
SUMMARY_CSV = os.path.abspath("oapen_pdfs_clasificados.csv")


CYBER_KEYWORDS = [
    # -------------------------
    # General English
    # -------------------------
    "cyber", "cybersecurity", "cyber security", "infosec", "information security",
    "security", "network security", "data security", "data protection", "privacy",
    "confidentiality", "integrity", "availability", "CIA triad",
    "cyber resilience", "risk management", "vulnerability management", "patch management",
    "secure development", "secure coding", "devsecops", "application security",
    "cloud security", "endpoint security", "OT security", "ICS security", "IoT security",
    "SCADA security", "critical infrastructure", "zero trust", "ZTNA", "trust but verify",
    "security by design", "privacy by design",

    # -------------------------
    # Threats & Attacks (English)
    # -------------------------
    "malware", "virus", "worm", "trojan", "ransomware", "locker", "crypto locker",
    "spyware", "adware", "rootkit", "keylogger", "botnet", "bot herder", "backdoor",
    "phishing", "spear phishing", "whaling", "smishing", "vishing", "business email compromise",
    "BEC", "credential stuffing", "brute force", "dictionary attack", "password spraying",
    "sql injection", "xss", "cross-site scripting", "csrf", "ssrf", "command injection",
    "remote code execution", "rce", "buffer overflow", "heap spray", "race condition",
    "zero-day", "0day", "supply chain attack", "watering hole attack", "drive-by download",
    "man-in-the-middle", "MITM", "session hijacking", "dns spoofing", "arp poisoning",
    "ddos", "denial of service", "distributed denial of service", "amplification attack",
    "side channel attack", "timing attack", "evil twin", "rogue access point",
    "clickjacking", "typosquatting", "spoofing", "deepfake", "credential theft",
    "insider threat", "cyber espionage", "cyber warfare", "hacktivism",

    # -------------------------
    # Defenses & Controls (English)
    # -------------------------
    "firewall", "next generation firewall", "ngfw", "proxy", "reverse proxy",
    "antivirus", "antimalware", "EDR", "XDR", "MDR", "NDR", "sandboxing",
    "IDS", "IPS", "SIEM", "SOAR", "UEBA", "threat intelligence", "CTI",
    "SOC", "security operations center", "incident response", "playbook", "runbook",
    "digital forensics", "forensic analysis", "incident handler", "containment",
    "eradication", "recovery", "business continuity", "disaster recovery", "BCP", "DRP",
    "identity management", "IAM", "identity and access management", "privileged access",
    "PAM", "MFA", "multi factor authentication", "2FA", "otp", "one time password",
    "certificate", "PKI", "TLS", "SSL", "VPN", "ipsec", "sase", "CASB", "secure web gateway",
    "encryption", "cryptography", "symmetric encryption", "asymmetric encryption",
    "hashing", "salting", "digital signature", "homomorphic encryption",

    # -------------------------
    # Standards & Compliance (English)
    # -------------------------
    "ISO 27001", "ISO 27002", "ISO 22301", "ISO 31000",
    "NIST", "NIST CSF", "NIST SP 800-53", "NIST SP 800-171",
    "CIS controls", "COBIT", "PCI DSS", "HIPAA", "GDPR", "CCPA",
    "FISMA", "FedRAMP", "SOC 2", "SOX", "GLBA", "Basel II",
    "OWASP", "top 10", "MITRE ATT&CK", "D3FEND", "Cyber Kill Chain",

    # -------------------------
    # Roles & Teams (English)
    # -------------------------
    "ciso", "chief information security officer", "security analyst", "threat hunter",
    "incident responder", "forensic investigator", "penetration tester", "pentester",
    "ethical hacker", "red team", "blue team", "purple team", "bug bounty",
    "white hat", "black hat", "grey hat",

    # -------------------------
    # Español General
    # -------------------------
    "ciber", "ciberseguridad", "seguridad informatica", "seguridad informática",
    "seguridad de la información", "protección de datos", "privacidad", "criptografía",
    "disponibilidad", "confidencialidad", "integridad", "resiliencia cibernética",
    "gestión de riesgos", "gestión de vulnerabilidades", "parches de seguridad",
    "desarrollo seguro", "codificación segura", "devsecops", "seguridad en la nube",
    "seguridad de aplicaciones", "seguridad de endpoints", "seguridad OT", "seguridad IoT",
    "infraestructura crítica", "confianza cero", "zero trust", "privacidad por diseño",

    # -------------------------
    # Amenazas & Ataques (Español)
    # -------------------------
    "malware", "virus", "gusano", "troyano", "ransomware", "secuestro de datos",
    "spyware", "adware", "rootkit", "registrador de teclas", "keylogger", "botnet",
    "phishing", "suplantación de identidad", "smishing", "vishing", "whaling",
    "compromiso de correo empresarial", "fuerza bruta", "ataque de diccionario",
    "inyección sql", "inyección de comandos", "ejecución remota de código",
    "desbordamiento de búfer", "día cero", "zero-day", "ataque a la cadena de suministro",
    "ataque watering hole", "descarga oculta", "hombre en el medio", "MITM",
    "secuestro de sesión", "suplantación DNS", "envenenamiento ARP",
    "denegación de servicio", "DDoS", "ataque de amplificación", "ataque de canal lateral",
    "ciberdelito", "ciberdelincuencia", "ciberespionaje", "ciberguerra", "hacktivismo",
    "amenaza interna", "amenaza persistente avanzada", "APT",

    # -------------------------
    # Defensas & Controles (Español)
    # -------------------------
    "cortafuegos", "firewall", "antivirus", "antimalware", "protección de endpoints",
    "detección de intrusos", "IDS", "prevención de intrusos", "IPS",
    "respuesta ante incidentes", "respuesta a incidentes", "análisis forense digital",
    "recuperación ante desastres", "continuidad de negocio", "plan de continuidad",
    "plan de recuperación", "gestión de identidades", "gestión de accesos",
    "autenticación multifactor", "autenticación de dos factores", "contraseña de un solo uso",
    "cifrado", "cifrado simétrico", "cifrado asimétrico", "firma digital",
    "VPN", "TLS", "SSL", "infraestructura de clave pública", "PKI",

    # -------------------------
    # Normativas & Cumplimiento (Español)
    # -------------------------
    "ISO 27001", "ISO 27002", "ISO 22301", "ISO 31000",
    "marco NIST", "controles CIS", "COBIT", "PCI DSS", "HIPAA",
    "RGPD", "GDPR", "LOPD", "SOC 2", "SOX", "GLBA",
    "OWASP", "top 10", "cadena de ataque", "matriz MITRE",

    # -------------------------
    # Roles & Equipos (Español)
    # -------------------------
    "CISO", "director de seguridad de la información", "analista de seguridad",
    "cazador de amenazas", "respuesta a incidentes", "investigador forense",
    "probador de penetración", "pentester", "hacker ético", "equipo rojo", "equipo azul",
    "equipo púrpura", "cazarrecompensas de bugs", "sombrero blanco", "sombrero negro", "sombrero gris"
]


def load_items(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("El JSON debe contener una lista de items")
    return data


def extract_text_fields(item: Dict[str, Any]) -> Tuple[str, List[str]]:
    title = ""
    subjects: List[str] = []
    for m in item.get("metadata") or []:
        key = m.get("key")
        val = m.get("value") or ""
        if key == "dc.title" and not title:
            title = val
        if key and key.startswith("dc.subject"):
            subjects.append(val)
    return title, subjects


def list_pdf_bitstreams(item: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for b in item.get("bitstreams") or []:
        if b.get("mimeType") == "application/pdf":
            name = b.get("name")
            if name:
                names.append(name)
    return names


def is_cybersecurity(title: str, subjects: List[str]) -> bool:
    haystack = " ".join([title] + subjects).lower()
    return any(keyword.lower() in haystack for keyword in CYBER_KEYWORDS)


def ensure_dirs() -> None:
    os.makedirs(OUT_DIR_CYBER, exist_ok=True)
    os.makedirs(OUT_DIR_OTHERS, exist_ok=True)


def classify_and_move() -> None:
    items = load_items(INPUT_JSON)
    ensure_dirs()

    # Construir índice nombre_pdf -> etiqueta
    mapping: Dict[str, Dict[str, Any]] = {}

    for item in items:
        title, subjects = extract_text_fields(item)
        label = "ciberseguridad" if is_cybersecurity(title, subjects) else "otros"
        handle = item.get("handle")
        bit_pdf_names = list_pdf_bitstreams(item)
        for name in bit_pdf_names:
            mapping[name] = {
                "label": label,
                "title": title,
                "handle": handle,
                "subjects": "; ".join(subjects)[:2000],  # limitar CSV
            }

    rows: List[List[str]] = []
    headers = ["file_name", "label", "title", "handle", "dest_path", "subjects", "source_path"]

    # Recorrer TODOS los archivos PDF en el directorio y subdirectorios
    moved = 0
    skipped = 0
    not_found = 0
    already_classified = 0
    moved_to_cyber = 0
    moved_to_others = 0

    for root, dirs, files in os.walk(PDF_DIR):
        # Saltar las carpetas de destino para evitar bucles infinitos
        if root == OUT_DIR_CYBER or root == OUT_DIR_OTHERS:
            continue
            
        for fname in files:
            if fname.endswith(".pdf"):
                src = os.path.join(root, fname)
                relative_path = os.path.relpath(src, PDF_DIR)
                
                # Verificar si ya está en la carpeta correcta
                if root == OUT_DIR_CYBER or root == OUT_DIR_OTHERS:
                    already_classified += 1
                    continue
                
                meta = mapping.get(fname)
                if not meta:
                    # No asociado por nombre exacto; clasificar como otros
                    dest = os.path.join(OUT_DIR_OTHERS, fname)
                    shutil.move(src, dest)
                    rows.append([fname, "otros", "", "", dest, "", relative_path])
                    moved += 1
                    moved_to_others += 1
                    continue

                label = meta["label"]
                dest_dir = OUT_DIR_CYBER if label == "ciberseguridad" else OUT_DIR_OTHERS
                dest = os.path.join(dest_dir, fname)
                shutil.move(src, dest)
                rows.append([fname, label, meta["title"], meta["handle"], dest, meta["subjects"], relative_path])
                moved += 1
                
                # Contar hacia qué carpeta va
                if label == "ciberseguridad":
                    moved_to_cyber += 1
                else:
                    moved_to_others += 1

    # Guardar CSV
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Procesados: {moved + already_classified}")
    print(f"Movidos: {moved}")
    print(f"  └─ A ciberseguridad: {moved_to_cyber}")
    print(f"  └─ A otros: {moved_to_others}")
    print(f"Ya clasificados: {already_classified}")
    print(f"CSV guardado: {SUMMARY_CSV}")


if __name__ == "__main__":
    classify_and_move()


