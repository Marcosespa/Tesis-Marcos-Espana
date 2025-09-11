import PyPDF2
import os
import glob


def extraerTexto(pdf_path, documentoName):
    if not os.path.exists(pdf_path):
        print(f"Error en el archivo {pdf_path}")
        return None

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        total_pages = len(reader.pages)
        print("El total de paginas es de: ", total_pages)
        texto = ""

        for i, page in enumerate(reader.pages, 1):
            try:
                page_text = page.extractText()
                if page_text:
                    texto += f"\n ---PAGINA{i}--- \n"
                    texto += page_text + "\n"
            except Exception as e:
                print(f"Error al procesar página {i}: {e}")
                continue

        print(f"\n Extraccion completada")

    # Crear nombre del archivo de salida
    output_file = f"{documentoName}.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(texto)

    print(f"Texto guardado en: {output_file}")
    return texto


def main():
    carpeta = "/Users/marcosrodrigo/Desktop/Universidad/Sexto Semestre/Tesis/Datos/DocumentosPDF"
    
    # Verificar que la carpeta existe
    if not os.path.exists(carpeta):
        print(f"La carpeta {carpeta} no existe")
        return
    
    # Buscar todos los archivos PDF en la carpeta
    archivos_pdf = glob.glob(os.path.join(carpeta, "*.pdf"))
    
    if not archivos_pdf:
        print("No se encontraron archivos PDF en la carpeta")
        return
    
    for pdf_path in archivos_pdf:
        # Obtener solo el nombre del archivo sin la ruta
        documentoName = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"Procesando: {documentoName}")
        
        textoProcesado = extraerTexto(pdf_path, documentoName)
        
        if textoProcesado is None:
            print(f"No se pudo procesar {documentoName}")
            continue


if __name__ == "__main__":
    main()
