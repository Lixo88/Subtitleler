import os
import subprocess
import whisper
from tqdm import tqdm

def extract_audio_to_wav(input_path, output_path):
    """
    EXTRAER AUDIO A WAV:
    --------------------
    Este paso es opcional, pues Whisper puede leer directamente varios formatos (.m4a, .mp3, etc.).
    Sin embargo, a veces convertir a WAV (PCM) evita problemas de decodificación.

    - input_path: ruta del archivo de audio de entrada (por ej. "audio.m4a").
    - output_path: ruta para guardar el WAV resultante (por ej. "audio_temp.wav").
    """
    command = [
        "ffmpeg",
        "-i", input_path,       # archivo de entrada
        "-ar", "16000",         # muestreo a 16 kHz
        "-ac", "1",             # 1 canal (mono)
        "-c:a", "pcm_s16le",    # formato PCM 16 bits
        output_path,
        "-y"                    # sobrescribe si ya existía un archivo con ese nombre
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def convert_seconds_to_srt_time(seconds: float) -> str:
    """
    CONVERSIÓN DE SEGUNDOS AL FORMATO SRT:
    --------------------------------------
    Los archivos SRT requieren que el tiempo se exprese en formato HH:MM:SS,mmm
    donde 'mmm' son milisegundos.

    Este método recibe un número de segundos (float) y devuelve una cadena en ese formato.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def create_srt(segments, srt_path):
    """
    CREAR EL ARCHIVO SRT:
    ---------------------
    - segments: lista de segmentos devueltos por whisper (cada elemento tiene 'start', 'end', 'text').
    - srt_path: ruta donde se guardará el archivo .srt.

    Recorre cada segmento, formatea los tiempos y el texto, y escribe las líneas
    necesarias en formato SRT.
    """
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, start=1):
            start_time = convert_seconds_to_srt_time(segment['start'])
            end_time = convert_seconds_to_srt_time(segment['end'])
            text = segment['text'].strip()

            # Cada "bloque" de subtítulo en SRT:
            f.write(f"{i}\n")                         # índice
            f.write(f"{start_time} --> {end_time}\n") # tiempo inicio --> fin
            f.write(f"{text}\n\n")                    # texto + línea en blanco

def transcribe_audio_whisper(audio_path, model_size="small", language="es"):
    """
    USAR WHISPER PARA TRANSCIBIR:
    -----------------------------
    - audio_path: ruta al archivo de audio (wav, mp3, m4a, etc.).
    - model_size: "tiny", "base", "small", "medium", "large". 
      Cuanto más grande el modelo, mayor precisión pero más lento y más pesado.
    - language: "es" para español, "en" para inglés, etc.

    Carga el modelo y devuelve los segmentos del texto transcrito.
    """
    model = whisper.load_model(model_size)
    # transcribe() devuelve un diccionario, donde "segments" son las partes con timestamps y texto
    result = model.transcribe(audio_path, language=language)
    return result["segments"]

def main():
    """
    FUNCIÓN PRINCIPAL:
    ------------------
    1) Pide una carpeta local donde tengas tus .m4a (u otro formato).
    2) Busca todos los archivos .m4a en esa carpeta.
    3) Convierte cada uno a WAV (opcional).
    4) Lo transcribe con Whisper.
    5) Crea un .srt con el mismo nombre del archivo de audio.
    6) Limpia el archivo temporal .wav (si se creó).

    Finalmente, tendrás un .srt por cada .m4a en esa carpeta.
    """
    
    # 1) Ruta de la carpeta donde están los archivos .m4a:
    print("Ingresa la ruta local de la carpeta con tus archivos .m4a:")
    folder_path = input("Ruta de la carpeta: ").strip()

    # Verifica si la carpeta existe
    if not os.path.isdir(folder_path):
        print("La carpeta especificada no existe. Revisa la ruta e inténtalo de nuevo.")
        return

    # 2) Recolectar todos los archivos con extensión .m4a
    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.m4a')]

    if not audio_files:
        print("No se encontraron archivos .m4a en la carpeta especificada.")
        return

    print(f"Se encontraron {len(audio_files)} archivos de audio .m4a en '{folder_path}'.")

    # 3) Procesar cada archivo
    #    tqdm nos mostrará una barra de progreso para saber en qué paso estamos.
    for file in tqdm(audio_files, desc="Procesando archivos"):
        base_name, _ = os.path.splitext(file)

        # Rutas de entrada y salida
        input_path = os.path.join(folder_path, file)                   # ej: "carpeta/audio.m4a"
        temp_wav_path = os.path.join(folder_path, f"{base_name}_temp.wav")  # ej: "carpeta/audio_temp.wav"
        srt_path = os.path.join(folder_path, f"{base_name}.srt")            # ej: "carpeta/audio.srt"

        # (Opcional) 4) Convertir a WAV
        extract_audio_to_wav(input_path, temp_wav_path)

        # 5) Transcribir con Whisper
        segments = transcribe_audio_whisper(
            audio_path=temp_wav_path,   # O podrías usar `input_path` directo si quieres saltarte el paso WAV
            model_size="small",         # "tiny", "base", "small", etc.
            language="es"               # Fuerza el idioma español
        )

        # 6) Crear SRT con el mismo nombre
        create_srt(segments, srt_path)

        # 7) Borrar archivo temporal WAV
        os.remove(temp_wav_path)

        print(f"[OK] Generado: {srt_path}")

if __name__ == "__main__":
    main()
