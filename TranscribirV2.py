import os
import subprocess
import whisper
from tqdm import tqdm

def extract_audio_to_wav(input_path, output_path):
    """
    Convierte a WAV (16kHz, mono) con ffmpeg. Paso opcional pero útil para evitar
    problemas de decodificación.
    """
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        output_path,
        "-y"
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def convert_seconds_to_srt_time(seconds: float) -> str:
    """
    Convierte un número de segundos en formato 'HH:MM:SS,mmm'.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def create_srt_with_natural_breaks(result,
                                   srt_path,
                                   max_chars=60,
                                   max_duration=5.0,
                                   silence_threshold=0.6,
                                   punctuation_breaks=['.', '!', '?']):
    """
    CREA SRT USANDO TIMESTAMPS DE PALABRA, DETECCIÓN DE SILENCIO Y PUNTUACIÓN.
    -------------------------------------------------------------------------
    - 'result': diccionario devuelto por Whisper con 'word_timestamps=True'.
    - 'srt_path': ruta del archivo .srt de salida.
    - 'max_chars': máx. de caracteres por subtítulo.
    - 'max_duration': máx. de segundos por subtítulo (p. ej. 5).
    - 'silence_threshold': si hay un silencio mayor (p. ej. 0.6s), se corta.
    - 'punctuation_breaks': lista de caracteres que, si se detectan al final
      de una palabra, se realiza un corte.

    LÓGICA PRINCIPAL:
    -----------------
    1) Unifica todas las "words" en una sola lista (words_list).
    2) Recorre secuencialmente, acumulando palabras en un "chunk" (subtítulo actual).
       - Corte si:
         a) superamos max_chars
         b) superamos max_duration
         c) hay un silencio > silence_threshold
         d) al final de una palabra aparece un signo de puntuación "fuerte"
            ('.', '!', '?'), en cuyo caso cortamos ahí si ya llevamos
            varias palabras en el chunk.
    3) Se guarda cada chunk con su start y end (primer y última palabra).
    """
    # 1) Unificamos las "words" de todos los segmentos
    words_list = []
    for seg in result["segments"]:
        if "words" in seg:
            words_list.extend(seg["words"])

    if not words_list:
        print("No se encontraron 'word timestamps' en el resultado. "
              "Asegúrate de usar word_timestamps=True.")
        return

    srt_index = 1
    subtitles = []

    # Variables para el chunk en construcción
    chunk_start_time = None
    chunk_end_time = None
    chunk_text = ""

    # Recorremos la lista palabra por palabra
    prev_word_end = 0.0  # guardamos el 'end' de la palabra anterior para detectar silencios

    for i, w in enumerate(words_list):
        word_text = w["word"]  # ej. "Hola "
        word_start = w["start"]
        word_end = w["end"]
        trimmed_word = word_text.strip()

        # Si el chunk está vacío (primera palabra) iniciamos
        if chunk_start_time is None:
            chunk_start_time = word_start
            chunk_end_time = word_end
            chunk_text = trimmed_word
            prev_word_end = word_end
            continue

        # Si no está vacío, checamos condiciones de corte
        proposed_text = (chunk_text + " " + trimmed_word).strip()
        proposed_duration = word_end - chunk_start_time

        # a) si se supera el límite de caracteres
        if len(proposed_text) > max_chars:
            # cerramos chunk anterior
            subtitles.append({
                "start": chunk_start_time,
                "end": chunk_end_time,
                "text": chunk_text.strip()
            })
            # iniciamos chunk nuevo con la palabra actual
            chunk_start_time = word_start
            chunk_end_time = word_end
            chunk_text = trimmed_word

        # b) si se supera la duración máxima
        elif proposed_duration > max_duration:
            subtitles.append({
                "start": chunk_start_time,
                "end": chunk_end_time,
                "text": chunk_text.strip()
            })
            chunk_start_time = word_start
            chunk_end_time = word_end
            chunk_text = trimmed_word

        # c) si hay un silencio largo entre la palabra anterior y la actual
        elif (word_start - prev_word_end) >= silence_threshold and len(chunk_text) > 0:
            # cerramos chunk por silencio
            subtitles.append({
                "start": chunk_start_time,
                "end": chunk_end_time,
                "text": chunk_text.strip()
            })
            chunk_start_time = word_start
            chunk_end_time = word_end
            chunk_text = trimmed_word

        else:
            # si no se superan los límites, añadimos la palabra
            chunk_end_time = word_end
            chunk_text = proposed_text

            # d) si la palabra termina en puntuación "fuerte"
            last_char = trimmed_word[-1] if trimmed_word else ""
            if last_char in punctuation_breaks and i < len(words_list) - 1:
                # cerramos chunk en este punto
                subtitles.append({
                    "start": chunk_start_time,
                    "end": chunk_end_time,
                    "text": chunk_text.strip()
                })
                # comenzamos un nuevo chunk pero SÓLO si no es la última palabra
                chunk_start_time = None
                chunk_end_time = None
                chunk_text = ""

        prev_word_end = word_end

    # Agregamos el último chunk si queda algo
    if chunk_text.strip():
        subtitles.append({
            "start": chunk_start_time,
            "end": chunk_end_time,
            "text": chunk_text.strip()
        })

    # Ahora escribimos el archivo SRT
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, sub in enumerate(subtitles, start=1):
            start_srt = convert_seconds_to_srt_time(sub["start"])
            end_srt = convert_seconds_to_srt_time(sub["end"])
            text = sub["text"]

            f.write(f"{i}\n")
            f.write(f"{start_srt} --> {end_srt}\n")
            f.write(f"{text}\n\n")

def transcribe_audio_whisper(audio_path, model_size="small", language="es"):
    """
    Usa Whisper con word_timestamps=True para obtener timestamps a nivel palabra.
    """
    model = whisper.load_model(model_size)
    result = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        verbose=False
    )
    return result

def main():
    print("Ingresa la ruta local de la carpeta con tus archivos .m4a:")
    folder_path = input("Ruta de la carpeta: ").strip()

    if not os.path.isdir(folder_path):
        print("La carpeta especificada no existe. Revisa la ruta e inténtalo de nuevo.")
        return

    audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.m4a')]
    if not audio_files:
        print("No se encontraron archivos .m4a en la carpeta especificada.")
        return

    print(f"Se encontraron {len(audio_files)} archivos de audio .m4a.")

    for file in tqdm(audio_files, desc="Procesando archivos"):
        base_name, _ = os.path.splitext(file)
        input_path = os.path.join(folder_path, file)
        temp_wav_path = os.path.join(folder_path, f"{base_name}_temp.wav")
        srt_path = os.path.join(folder_path, f"{base_name}.srt")

        # Convertir a WAV (opcional, si .m4a funciona sin problemas, puedes omitir)
        extract_audio_to_wav(input_path, temp_wav_path)

        # Transcribir con word timestamps
        result = transcribe_audio_whisper(
            audio_path=temp_wav_path,
            model_size="small",
            language="es"
        )

        # Crear SRT con silencios y puntuación
        create_srt_with_natural_breaks(
            result=result,
            srt_path=srt_path,
            max_chars=60,
            max_duration=5.0,
            silence_threshold=0.6,     # Ajusta si quieres ser más o menos sensible al silencio
            punctuation_breaks=['.', '!', '?']  # Podrías incluir ',' si gustas
        )

        os.remove(temp_wav_path)
        print(f"[OK] Generado: {srt_path}")

if __name__ == "__main__":
    main()
