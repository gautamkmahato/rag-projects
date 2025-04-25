import os
from dotenv import load_dotenv
import yt_dlp
import whisper
import re
import time


load_dotenv() 


# Configuration
YOUTUBE_URL = "https://www.youtube.com/watch?v=FQCTzomz6bw"


def download_audio(video_url, output_path="audio"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])


def generate_transcript():
    # Load Whisper model
    model = whisper.load_model("base")

    # Transcribe with word-level timestamps
    result = model.transcribe("audio.mp3", word_timestamps=True, verbose=False)

    sentences = []
    current_sentence = []
    start_time = None

    for segment in result["segments"]:
        for word in segment["words"]:
            word_text = word["word"]
            if start_time is None:
                start_time = word["start"]
            
            current_sentence.append(word)
            
            if re.search(r"[.!?]$", word_text.strip()):  # End of sentence
                end_time = word["end"]
                text = "".join(w["word"] for w in current_sentence).strip()
                sentences.append({
                    "text": text,
                    "start": start_time,
                    "end": end_time
                })
                current_sentence = []
                start_time = None


    # write transcript to the text file
    file_name = f"transcript_{int(time.time())}.txt"

    with open(file_name, "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(f"[{s['start']:.2f}s - {s['end']:.2f}s] {s['text']}\n")


# Example usage
download_audio(YOUTUBE_URL)
generate_transcript()
