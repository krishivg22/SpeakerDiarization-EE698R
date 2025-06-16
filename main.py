import assemblyai as aai
import datetime
import numpy as np
import torchaudio
import torch
import os
import wave
import threading
import pyaudio
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax, entr
from pyannote.audio import Inference, Model
from torch.cuda.amp import autocast
from langdetect import detect, DetectorFactory
from pydub import AudioSegment

DetectorFactory.seed = 0  # for consistent language results

HF_TOKEN = "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"
ASSEMBLY_AI_KEY = "82840b44091848d3a78b52c7966fd380"
RECORDED_FILE = "mic_recorded.mp3"
PRE_RECORDED_FILE = "./data/spread.mp3"  
RECORD_SECONDS = 999
SAMPLE_RATE = 16000

def record_audio(filename, stop_event):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=1024)
    frames = []
    print(" Recording... (press ENTER to stop)\n")
    while not stop_event.is_set():
        data = stream.read(1024)
        frames.append(data)
    print(" Recording stopped. Saving file...\n")
    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def run_diarization(audio_path):
    aai.settings.api_key = ASSEMBLY_AI_KEY
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(audio_path)

    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to("cuda" if torch.cuda.is_available() else "cpu")

    embedding_model = Inference(
        Model.from_pretrained("pyannote/embedding", use_auth_token=HF_TOKEN).to(waveform.device),
        window="whole"
    )

    aligned_embeddings = []
    aligned_utterances = []

    for utt in transcript.utterances:
        start_time = utt.start / 1000
        end_time = utt.end / 1000
        start_idx = int(start_time * sample_rate)
        end_idx = int(end_time * sample_rate)
        segment = waveform[:, start_idx:end_idx].cpu()

        with torch.no_grad(), autocast():
            embedding = embedding_model({"waveform": segment, "sample_rate": sample_rate})

        if not isinstance(embedding, np.ndarray):
            embedding = embedding.numpy()
        embedding = embedding / np.linalg.norm(embedding)
        aligned_embeddings.append(embedding)
        aligned_utterances.append(utt)

    X = np.vstack(aligned_embeddings)

    best_k, best_score = 1, -1
    if len(X) <= 2:
        best_k = len(X)
    else:
        for k in range(2, min(6, len(X))):
            kmeans = KMeans(n_clusters=k, n_init=10).fit(X)
            score = silhouette_score(X, kmeans.labels_)
            if score > best_score:
                best_score = score
                best_k = k

    kmeans = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
    cluster_labels = kmeans.predict(X)

    utterance_info = []
    for i, utt in enumerate(aligned_utterances):
        emb = aligned_embeddings[i].reshape(1, -1)
        similarities = cosine_similarity(emb, kmeans.cluster_centers_).flatten()
        probs = softmax(similarities)
        confidence = np.max(probs)
        uncertainty = entr(probs).sum()

        try:
            pred_lang = detect(utt.text.strip())
        except:
            pred_lang = "unknown"

        utterance_info.append({
            "start": str(datetime.timedelta(milliseconds=utt.start)).split(".")[0],
            "speaker": cluster_labels[i],
            "text": utt.text,
            "confidence": confidence,
            "uncertainty": uncertainty,
            "language": pred_lang
        })

    print(f"\n Estimated Number of Speakers: {best_k}\n")
    for utt in utterance_info:
        print(f"{utt['start']} | Speaker {utt['speaker']} | "
              f"Conf: {utt['confidence']:.2f} | Uncert: {utt['uncertainty']:.2f} ")
        print(f"{utt['text']}\n")
        
print("Choose mode:\n1. Real-time Mic Recording\n2. Pre-recorded File")
choice = input("Enter 1 or 2: ").strip()

if choice == '1':
    stop_event = threading.Event()
    thread = threading.Thread(target=record_audio, args=("mic_recorded.wav", stop_event))
    thread.start()
    input(" Press ENTER when done speaking...\n")
    stop_event.set()
    thread.join()
    sound = AudioSegment.from_wav("mic_recorded.wav")

    # Export as MP3
    sound.export(RECORDED_FILE, format="mp3")
    run_diarization(RECORDED_FILE)

elif choice == '2':
    if not os.path.exists(PRE_RECORDED_FILE):
        print(f"❌ File not found: {PRE_RECORDED_FILE}")
    else:
        run_diarization(PRE_RECORDED_FILE)
else:
    print("❌ Invalid choice. Exiting.")