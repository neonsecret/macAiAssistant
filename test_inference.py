import json
import os
import tempfile
import wave
from datetime import datetime

import pyaudio
import sounddevice as sd
import soundfile as sf
import torch
from TTS.api import TTS
from datasets import load_dataset
from llama_cpp import Llama
from playsound import playsound
from pywhispercpp.model import Model

whisper_model = Model('base', n_threads=6)
llm = Llama.from_pretrained(
    repo_id="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
    filename="*Q8.gguf",
    verbose=False,
    n_gpu_layers=-1
)
device = torch.device("cpu")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)


def transcribe_directly():
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav")

    sample_rate = 16000
    bits_per_sample = 16
    chunk_size = 1024
    audio_format = pyaudio.paInt16
    channels = 1

    def callback(in_data, frame_count, time_info, status):
        wav_file.writeframes(in_data)
        return None, pyaudio.paContinue

    wav_file = wave.open(temp_file.name, 'wb')
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(bits_per_sample // 8)
    wav_file.setframerate(sample_rate)
    audio = pyaudio.PyAudio()

    stream = audio.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size,
                        stream_callback=callback)

    input("Press Enter to stop recording...")
    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Close the wave file
    wav_file.close()
    result = whisper_model.transcribe(temp_file.name)
    temp_file.close()

    return "".join([x.text for x in result])


def answer_speech(prompt, default_formatting=True):
    print("Infer start")
    output = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. "
                           "If you are asked about the weather, only say one word: weather; "
                           "If you are asked about time, only return one word: time; "
                           "If you are asked about today's date, only say: date;"
                           "If you are asked to list the current folder, say: list.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    # print(output)
    # print(json.dumps(output["choices"][0]["message"], indent=2))
    out = output["choices"][0]["message"]
    if default_formatting:
        out = out["content"]
    return out


def parse_output(response):
    match response.lower().strip():
        case "weather":
            return "It's sunny in Plzen"  # TODO Get weather
        case "date":
            return datetime.now().strftime("It's %A the %d, %B %Y")
        case "time":
            return datetime.now().strftime("It's %H %M")
        case "list":
            return " ,".join(os.listdir())  # run a command
        case _:
            return response


def voice_speech(prompt, playback=True):
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav")
    tts.tts_to_file(text=prompt, speaker="Ana Florence", language="en", file_path=temp_file.name)
    if playback:
        print("Playing back")
        playsound(temp_file.name)
    temp_file.close()


if __name__ == '__main__':
    prompt = transcribe_directly()
    print(prompt)
    # prompt = "List the current folder"
    out = answer_speech(prompt)
    print(out)
    out = parse_output(out)
    print(out)
    voice_speech(out)
