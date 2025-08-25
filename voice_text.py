import os
import pvporcupine
import traceback
import re
import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue
import threading
import librosa

from smolagents import CodeAgent, LiteLLMModel, tool
from piper import PiperVoice
# from transformers import pipeline
import smtplib
from email.mime.text import MIMEText

import traceback

from dotenv import load_dotenv

load_dotenv()
# =================================================================================
# --- Configuration ---
YOUR_DEVICE_ID = "Rapoo Camera: USB Audio"
DEVICE_SAMPLE_RATE = 44100
WHISPER_SAMPLE_RATE = 16000
MODEL_TYPE = "base.en"  # Can be tiny.en, base.en, small.en, medium.en, large-v2

# --- VAD Configuration ---
ENERGY_THRESHOLD = 0.1
SILENCE_DURATION = 2
CONVO_END_DURATION = 10
CHUNK_DURATION = 0.2
# =================================================================================

audio_queue = queue.Queue()
is_recording = threading.Event()

# limellm to connect my local ollama
model_llm = LiteLLMModel(
    model_id="ollama_chat/gemma3",
    api_base="http://localhost:11434",
)


@tool
def send_markdown_email(content: str, to_email: str, subject: str = "AI Tooling Briefing") -> str:
    """Sends an email with markdown content using credentials from environment variables.

    Args:
        content (str): The markdown-formatted content of the email body.
        to_email (str): The recipient's email address.
        subject (str): The subject line of the email.
    """
    # Load credentials securely from environment variables
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")

    if not all([smtp_server, smtp_user, smtp_password]):
        return "Error: SMTP environment variables not set. Please set SMTP_SERVER, SMTP_USER, and SMTP_PASSWORD."

    # Prepare and send the email
    msg = MIMEText(content, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, [to_email], msg.as_string())
        return f"Email sent successfully to {to_email}."
    except Exception as e:
        return f"Failed to send email: {e}"



# code agent framework from smolagents
chat_history = []
agent = CodeAgent(tools=[send_markdown_email], add_base_tools=True, model=model_llm, stream_outputs=True)

voice = PiperVoice.load("./en_US-kusal-medium.onnx")

# wake up porcupine
porcupine = pvporcupine.create(
    access_key=os.getenv("PRCPINE_KEY"),
    keyword_paths=["./hey-jarvis_en_linux_v3_0_0/hey-jarvis_en_linux_v3_0_0.ppn"],
)

# print("Loading zero-shot classification model...")
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
# print("Classifier loaded.")

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    resampled_chunk = librosa.resample(
        y=indata.flatten(), orig_sr=DEVICE_SAMPLE_RATE, target_sr=WHISPER_SAMPLE_RATE
    )
    audio_queue.put(resampled_chunk)

# def get_intent(text: str) -> str:
#     """
#     Classifies intent using a fast, specialized zero-shot model.
#     """
#     print("Classifying intent (zero-shot)...")

#     # Use more descriptive labels for better accuracy
#     candidate_labels = ["conversation", "computer task"]

#     try:
#         # The model returns scores for each label
#         result = classifier(text, candidate_labels)

#         # The label with the highest score is our intent
#         top_label = result['labels'][0]
#         print(f"Top label: '{top_label}' with score {result['scores'][0]:.2f}")

#         if top_label == "computer task":
#             return "task"
#         return "chat"

#     except Exception as e:
#         print("getting intent failed ", e)
#         print(traceback.format_exc())
#         return "task"
def get_intent(text: str) -> str:
    print("Classifying intent...")
    try:
        system_prompt = """
                    You are a precise intent classifier. Your job is to determine if the user's request is a 'task' or a 'chat'.

                    ## Definitions
                    - A 'task' is a specific, direct command that requires performing an action like searching the web, executing code, or manipulating files. A task is also required for any query that needs **real-time, up-to-the-minute information** (e.g., current stock prices, live weather, currency exchange rates).
                    - A 'chat' is conversational. It includes asking for general information, requesting explanations, seeking opinions, or asking for creative text generation like poems or stories.

                    ## Instructions
                    - If the user is giving a command to DO something or asking for LIVE data, classify it as 'task'.
                    - If the user is asking for information that is not time-sensitive, classify it as 'chat', no matter how complex the topic is.
                    - Respond with only a single word: 'chat' or 'task'.

                    ## Examples
                    User: Hello there!
                    Assistant: chat

                    User: Explain quantum computing to me in simple terms.
                    Assistant: chat

                    User: What's the exchange rate for USD to GBP right now?
                    Assistant: task
            
                    User: Write a python script to download a youtube video.
                    Assistant: task

                    User: Search the web for the latest news on AI.
                    Assistant: task

                    User: Tell me a story about a dragon.
                    Assistant: chat

                    User: List the files in my current directory.
                    Assistant: task
                """
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": text}]},
        ]

        intent = model_llm(messages, max_tokens=5).content

        if "task" in intent:
            return "task"
        return "chat"

    except Exception as e:
        print(traceback.format_exc())
        print(
            f"Intent classification failed: {e}. Defaulting to 'task'."
        )
        return "task"


def process_audio_for_transcription(audio_frames, model):
    is_recording.set()
    print("Processing phrase...")
    full_audio = np.concatenate(audio_frames)

    print("Transcribing with faster-whisper...")
    segments, info = model.transcribe(full_audio, beam_size=5)

    # Concatenate the transcribed segments
    transcribed_text = "".join(segment.text for segment in segments).strip()

    full_response = ""
    if transcribed_text:
        print("\n--- You Said ---")
        print(transcribed_text)
        print("----------------\n")
        intent = get_intent(transcribed_text)
        if intent == "task":
            full_response = str(agent.run(transcribed_text))
        else:
            # TODO: merge the two memories from normal chat and the agent run
            current_chat = {
                "role": "user",
                "content": [{"type": "text", "text": transcribed_text}],
            }
            chat_history.append(current_chat)
            full_response = model_llm(chat_history, max_tokens=500).content
            chat_history.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": full_response}],
                }
            )
        print("----assistant----")
        print(full_response)
        # Synthesize audio from the full response text
        audio_generator = voice.synthesize(re.sub(r"[\*_`#]", "", full_response))

        audio_arrays = [chunk.audio_float_array for chunk in audio_generator]

        #  Concatenate all the small arrays into one big array
        if audio_arrays:
            full_audio_array = np.concatenate(audio_arrays)

            # Convert the float audio to the int16 format required by sounddevice
            # The values are in [-1, 1], so we scale them to the int16 range
            audio_int16 = (full_audio_array * 32767).astype(np.int16)

            # Play the final audio
            sd.play(audio_int16, samplerate=voice.config.sample_rate)
            sd.wait()
    else:
        print("No speech detected in the last phrase.")

    with audio_queue.mutex:
        audio_queue.queue.clear()

    is_recording.clear()


def process_voice(model):
    try:
        chunk_frames_device = int(CHUNK_DURATION * DEVICE_SAMPLE_RATE)
        phrase_audio = []
        silence_chunks = 0
        convo_end_chunks = 0
        max_silence_chunks = int(SILENCE_DURATION / CHUNK_DURATION)
        max_convo_end_chunks = int(CONVO_END_DURATION / CHUNK_DURATION)
        is_speaking = False

        with audio_queue.mutex:
            audio_queue.queue.clear()

        with sd.InputStream(
            samplerate=DEVICE_SAMPLE_RATE,
            device=YOUR_DEVICE_ID,
            channels=1,
            dtype="float32",
            blocksize=chunk_frames_device,
            callback=audio_callback,
        ):
            while True:
                if is_recording.is_set():
                    continue
                audio_chunk = audio_queue.get()
                rms = np.sqrt(np.mean(audio_chunk**2))
                if rms < ENERGY_THRESHOLD:
                    convo_end_chunks += 1
                    if convo_end_chunks > max_convo_end_chunks:
                        break
                if is_speaking:
                    phrase_audio.append(audio_chunk)
                    if rms < ENERGY_THRESHOLD:
                        silence_chunks += 1
                        if silence_chunks > max_silence_chunks:
                            is_speaking = False
                            # Pass the model object to the processing thread
                            threading.Thread(
                                target=process_audio_for_transcription,
                                args=(phrase_audio.copy(), model),
                            ).start()
                            phrase_audio = []
                    else:
                        silence_chunks = 0
                elif rms > ENERGY_THRESHOLD:
                    print("▶️  Speech detected...")
                    is_speaking = True
                    silence_chunks = 0
                    convo_end_chunks = 0
                    phrase_audio.append(audio_chunk)
    except Exception as e:
        print(e)
        print(traceback.format_exc())


##TODO: use threading to listen and if there is any stop or cancel current process then stop the process and run the new task with updated chat history

if __name__ == "__main__":
    print("Loading faster-whisper model...")
    # For CPU:
    # model = WhisperModel(MODEL_TYPE, device="cpu", compute_type="int8")
    # For GPU:
    model = WhisperModel(MODEL_TYPE, device="cuda", compute_type="float16")

    print("Model loaded. Listening for your voice...")

    print("Initializing Porcupine wake word engine...")
    try:
        # This InputStream is for Porcupine, which requires a different sample rate and data type

        def listen_for_wake_word():
            with sd.InputStream(
                samplerate=porcupine.sample_rate,
                device=YOUR_DEVICE_ID,
                channels=1,
                dtype="int16",  # Porcupine requires int16 audio
                blocksize=porcupine.frame_length,
                callback=lambda indata, frames, time, status: audio_queue.put(
                    indata.flatten()
                ),
            ):
                with audio_queue.mutex:
                    audio_queue.queue.clear()
                print("\nListening for 'Hey Jarvis'...")
                while True:
                    pcm = audio_queue.get()
                    result = porcupine.process(pcm)

                    if result >= 0:
                        print("✅ Wake word detected!")
                        break

        while True:
            listen_for_wake_word()
            process_voice(model)

    except KeyboardInterrupt:
        print("\nExiting program.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print(traceback.format_exc())
