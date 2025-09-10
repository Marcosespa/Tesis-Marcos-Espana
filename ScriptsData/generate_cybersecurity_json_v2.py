#!/usr/bin/env python3
"""
Script para generar un JSON completo con todos los PDFs de ciberseguridad de OAPEN
Usando la API de búsqueda en lugar de paginación
"""

import requests
import json
import time
from urllib.parse import urljoin

# Configuración
OAPEN_API_BASE = "https://library.oapen.org/rest"
OUTPUT_JSON = "cybersecurity_books_complete.json"

# Palabras clave de ciberseguridad para búsqueda
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

def search_cybersecurity_items():
    """Busca items de ciberseguridad usando la API de búsqueda"""
    all_items = []
    seen_handles = set()
    
    print("🔍 Buscando items de ciberseguridad usando términos de búsqueda...")
    
    for i, term in enumerate(CYBER_KEYWORDS):
        print(f"🔎 Búsqueda {i+1}/{len(CYBER_KEYWORDS)}: '{term}'")
        
        try:
            # Usar la API de búsqueda
            search_url = f"{OAPEN_API_BASE}/items"
            params = {
                'query': term,
                'expand': 'metadata,bitstreams',
                'limit': 100  # Máximo por búsqueda
            }
            
            response = requests.get(search_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            items = data if isinstance(data, list) else []
            
            new_items = 0
            for item in items:
                handle = item.get('handle')
                if handle and handle not in seen_handles:
                    # Verificar que tenga PDFs
                    has_pdf = False
                    for bitstream in item.get('bitstreams', []):
                        if bitstream.get('mimeType') == 'application/pdf':
                            has_pdf = True
                            break
                    
                    if has_pdf:
                        all_items.append(item)
                        seen_handles.add(handle)
                        new_items += 1
                        
                        # Mostrar título si está disponible
                        title = ""
                        for meta in item.get('metadata', []):
                            if meta.get('key') == 'dc.title':
                                title = meta.get('value', '')
                                break
                        
                        print(f"  ✅ {title[:60]}...")
            
            print(f"  📊 Encontrados: {new_items} nuevos items")
            print(f"  📈 Total acumulado: {len(all_items)}")
            
            # Pausa entre búsquedas
            time.sleep(1)
            
        except requests.exceptions.RequestException as e:
            print(f"  ❌ Error en búsqueda '{term}': {e}")
            continue
        except Exception as e:
            print(f"  ❌ Error inesperado en '{term}': {e}")
            continue
    
    return all_items

def get_item_details(item_handle):
    """Obtiene detalles completos de un item específico"""
    try:
        url = f"{OAPEN_API_BASE}/items"
        params = {
            'query': f'handle:"{item_handle}"',
            'expand': 'metadata,bitstreams',
            'limit': 1
        }
        
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, list) and len(data) > 0:
            return data[0]
        
    except Exception as e:
        print(f"  ❌ Error obteniendo detalles de {item_handle}: {e}")
    
    return None

def main():
    print("🚀 Iniciando generación de JSON completo de ciberseguridad...")
    print(f"🔍 Términos de búsqueda: {len(CYBER_KEYWORDS)}")
    
    # Buscar items de ciberseguridad
    cyber_items = search_cybersecurity_items()
    
    if not cyber_items:
        print("❌ No se encontraron items de ciberseguridad")
        return
    
    print(f"\\n📊 Total de items encontrados: {len(cyber_items)}")
    
    # Obtener detalles completos de cada item
    print("\\n🔍 Obteniendo detalles completos de cada item...")
    detailed_items = []
    
    for i, item in enumerate(cyber_items):
        handle = item.get('handle')
        print(f"  📥 {i+1}/{len(cyber_items)}: {handle}")
        
        detailed_item = get_item_details(handle)
        if detailed_item:
            detailed_items.append(detailed_item)
        
        time.sleep(0.5)  # Pausa para no sobrecargar la API
    
    # Guardar en JSON
    print(f"\\n💾 Guardando {len(detailed_items)} items en {OUTPUT_JSON}...")
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(detailed_items, f, indent=2, ensure_ascii=False)
    
    print(f"✅ JSON generado exitosamente: {OUTPUT_JSON}")
    print(f"📊 Total de items de ciberseguridad: {len(detailed_items)}")
    
    # Mostrar estadísticas de PDFs
    total_pdfs = 0
    for item in detailed_items:
        for bitstream in item.get('bitstreams', []):
            if bitstream.get('mimeType') == 'application/pdf':
                total_pdfs += 1
    
    print(f"📄 Total de PDFs de ciberseguridad: {total_pdfs}")

if __name__ == "__main__":
    main()
