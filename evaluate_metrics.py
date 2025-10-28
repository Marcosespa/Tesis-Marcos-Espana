#!/usr/bin/env python3
"""
Script para calcular métricas de evaluación de respuestas RAG
Compara respuestas generadas con respuestas esperadas usando múltiples métricas
"""

import argparse
import pandas as pd
import nltk
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from textstat import flesch_kincaid_grade, gunning_fog
from nltk.translate.bleu_score import sentence_bleu
from pathlib import Path
import sys

# Descargar recursos necesarios para NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"⚠️ Warning: Could not download NLTK resources: {e}")

def calculate_metrics(reference, generated):
    """
    Calcula métricas semánticas, léxicas y de legibilidad para comparar una respuesta generada
    con una de referencia, asumiendo que ambas están en texto limpio.
    """
    try:
        # Tokenizar para BLEU
        reference_tokens = [nltk.word_tokenize(reference)]
        generated_tokens = nltk.word_tokenize(generated)

        # 1. Métricas semánticas: BERTScore
        P, R, F1 = bert_score([generated], [reference], lang="en", verbose=False)
        bertscore_f1 = F1.item()

        # 2. Métricas léxicas: BLEU y ROUGE-L
        bleu_score = sentence_bleu(reference_tokens, generated_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, generated)
        rouge_l_f1 = rouge_scores['rougeL'].fmeasure

        # 3. Métricas de legibilidad: Flesch-Kincaid y Gunning Fog
        fk_grade = flesch_kincaid_grade(generated)
        gf_index = gunning_fog(generated)

        return {
            "BERTScore_F1": bertscore_f1,
            "BLEU": bleu_score,
            "ROUGE-L_F1": rouge_l_f1,
            "Flesch-Kincaid_Grade": fk_grade,
            "Gunning_Fog_Index": gf_index
        }
    except Exception as e:
        print(f"❌ Error calculating metrics: {e}")
        return {
            "BERTScore_F1": 0.0,
            "BLEU": 0.0,
            "ROUGE-L_F1": 0.0,
            "Flesch-Kincaid_Grade": 0.0,
            "Gunning_Fog_Index": 0.0
        }

def process_excel_file(file_path):
    """
    Procesa un archivo Excel con respuestas RAG y calcula métricas
    """
    try:
        print(f"📊 Procesando archivo: {file_path}")
        
        # Leer el Excel
        df = pd.read_excel(file_path)
        
        # Verificar columnas necesarias
        # required_columns = ['rag_answer', 'expected_answer']
        required_columns = ['Answer Cleaned', 'expected_answer']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Columnas faltantes: {missing_columns}")
            print(f"📋 Columnas disponibles: {list(df.columns)}")
            return None
        
        print(f"✅ Archivo cargado: {len(df)} filas encontradas")
        
        # Calcular métricas para cada fila
        results = []
        
        for i, row in df.iterrows():
            print(f"🔄 Procesando fila {i+1}/{len(df)}...")
            
            reference = str(row['expected_answer'])
            generated = str(row['Answer Cleaned'])
            # generated = str(row['rag_answer'])
            # Calcular métricas
            metrics = calculate_metrics(reference, generated)
            
            # Agregar información de la fila
            result_row = {
                'row_index': i,
                'question': row.get('question', 'N/A'),
                'expected_answer_length': len(reference),
                'generated_answer_length': len(generated),
                **metrics
            }
            
            results.append(result_row)
        
        # Crear DataFrame con resultados
        results_df = pd.DataFrame(results)
        
        # Calcular estadísticas resumidas
        summary_stats = {
            'total_questions': len(results_df),
            'avg_bertscore_f1': results_df['BERTScore_F1'].mean(),
            'avg_bleu': results_df['BLEU'].mean(),
            'avg_rouge_l_f1': results_df['ROUGE-L_F1'].mean(),
            'avg_flesch_kincaid': results_df['Flesch-Kincaid_Grade'].mean(),
            'avg_gunning_fog': results_df['Gunning_Fog_Index'].mean(),
            'min_bertscore_f1': results_df['BERTScore_F1'].min(),
            'max_bertscore_f1': results_df['BERTScore_F1'].max(),
        }
        
        return results_df, summary_stats
        
    except Exception as e:
        print(f"❌ Error procesando archivo: {e}")
        return None, None

def save_results(results_df, summary_stats, output_file):
    """
    Guarda los resultados en un archivo Excel
    """
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Hoja con resultados detallados
            results_df.to_excel(writer, sheet_name='Métricas_Detalladas', index=False)
            
            # Hoja con estadísticas resumidas
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_excel(writer, sheet_name='Resumen_Estadísticas', index=False)
        
        print(f"✅ Resultados guardados en: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error guardando resultados: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Calcula métricas de evaluación para respuestas RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "file_path", 
        nargs='?',
        default="ExcelLimpio.xlsx",
        help="Ruta al archivo Excel con respuestas RAG (por defecto: ExcelLimpio.xlsx)"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Archivo de salida para los resultados (por defecto: metrics_results.xlsx)"
    )
    
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Ejecutar con datos de demostración"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 CALCULADORA DE MÉTRICAS DE EVALUACIÓN RAG")
    print("=" * 80)
    
    if args.demo:
        print("🎯 Ejecutando demostración con datos de ejemplo...")
        
        # Ejemplo de respuesta de referencia y generada por un LLM (en texto limpio)
        reference_text = (
            "La ciberseguridad protege sistemas, redes y datos contra accesos no autorizados y ataques. "
            "Es crucial para garantizar la confidencialidad, integridad y disponibilidad de la información."
        )
        generated_text = (
            "La ciberseguridad se enfoca en proteger sistemas y redes frente a accesos no permitidos y amenazas digitales, "
            "asegurando la confidencialidad e integridad de los datos."
        )

        # Calcular métricas
        metrics = calculate_metrics(reference_text, generated_text)

        # Imprimir resultados
        print("\n📈 Resultados de la evaluación:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return 0
    
    if not args.file_path:
        print("❌ Error: Debes proporcionar un archivo Excel o usar --demo")
        print("💡 Uso:")
        print("  python evaluate_metrics.py [archivo.xlsx]  # Por defecto usa ExcelLimpio.xlsx")
        print("  python evaluate_metrics.py --demo")
        return 1
    
    # Verificar que el archivo existe
    file_path = Path(args.file_path)
    if not file_path.exists():
        print(f"❌ Error: El archivo {file_path} no existe")
        return 1
    
    # Procesar archivo
    results_df, summary_stats = process_excel_file(file_path)
    
    if results_df is None:
        return 1
    
    # Mostrar estadísticas resumidas
    print("\n📈 ESTADÍSTICAS RESUMIDAS:")
    print("-" * 50)
    print(f"Total de preguntas: {summary_stats['total_questions']}")
    print(f"BERTScore F1 promedio: {summary_stats['avg_bertscore_f1']:.4f}")
    print(f"BLEU promedio: {summary_stats['avg_bleu']:.4f}")
    print(f"ROUGE-L F1 promedio: {summary_stats['avg_rouge_l_f1']:.4f}")
    print(f"Flesch-Kincaid promedio: {summary_stats['avg_flesch_kincaid']:.2f}")
    print(f"Gunning Fog promedio: {summary_stats['avg_gunning_fog']:.2f}")
    print(f"BERTScore F1 rango: {summary_stats['min_bertscore_f1']:.4f} - {summary_stats['max_bertscore_f1']:.4f}")
    
    # Guardar resultados
    output_file = args.output or "metrics_results.xlsx"
    if save_results(results_df, summary_stats, output_file):
        print(f"\n🎉 Evaluación completada exitosamente!")
        print(f"📁 Resultados guardados en: {output_file}")
        return 0
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
