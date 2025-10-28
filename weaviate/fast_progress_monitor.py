#!/usr/bin/env python3
"""
Monitor rápido para verificar el progreso de la ingesta
"""

import weaviate
import time
from datetime import datetime

def connect_to_weaviate():
    """Conectar a Weaviate"""
    try:
        client = weaviate.connect_to_local()
        return client
    except Exception as e:
        print(f"❌ Error conectando a Weaviate: {e}")
        return None

def estimate_progress():
    """Estimar el progreso basado en el tiempo transcurrido"""
    try:
        client = connect_to_weaviate()
        if not client:
            return
        
        print("🚀 MONITOR RÁPIDO DE PROGRESO")
        print("=" * 40)
        print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        collections = ['NIST_SP', 'USENIX', 'MITRE', 'OWASP', 'SecurityTools', 'AISecKG', 'AnnoCTR', 'NIST_AI', 'NIST_CSWP']
        expected_counts = {
            "NIST_SP": 54973,
            "USENIX": 41875,
            "MITRE": 470,
            "OWASP": 34,
            "SecurityTools": 28,
            "AISecKG": 39,
            "AnnoCTR": 714,
            "NIST_AI": 419,
            "NIST_CSWP": 149
        }
        
        total_found = 0
        active_collections = 0
        
        for col in collections:
            try:
                collection = client.collections.get(col)
                response = collection.query.fetch_objects(limit=100)
                count = len(response.objects)
                
                if count > 0:
                    total_found += count
                    active_collections += 1
                    status = "🔄"
                    expected = expected_counts[col]
                    percent = min((count / expected) * 100, 100.0)
                    print(f"{status} {col:12}: ~{count}+/{expected} ({percent:.1f}%)")
                else:
                    print(f"⏳ {col:12}: Esperando...")
                    
            except Exception as e:
                print(f"❌ {col:12}: Error")
        
        total_expected = sum(expected_counts.values())
        total_percent = (total_found / total_expected) * 100
        
        print(f"\n📊 RESUMEN:")
        print(f"   Colecciones activas: {active_collections}/{len(collections)}")
        print(f"   Progreso estimado: {total_percent:.1f}%")
        
        if active_collections > 0:
            print(f"✅ La ingesta está funcionando correctamente")
            print(f"   Con batch size 256, debería ser más rápido que antes")
        else:
            print(f"⏳ La ingesta está comenzando...")
        
        client.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    estimate_progress()
