import os
import tempfile
import threading
import wave
from datetime import datetime

import pyaudio
import rumps
import torch
from TTS.api import TTS
from llama_cpp import Llama
from playsound import playsound
from pywhispercpp.model import Model


class AssistantModelsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_recorded_result = None
        self.llm = Llama.from_pretrained(
            repo_id="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
            filename="*Q8.gguf",
            verbose=False,
            n_gpu_layers=-1
        )
        self.device = torch.device("cpu")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.whisper_model = Model('base', n_threads=6)

    def answer_speech(self, prompt, default_formatting=True):
        print("Infer start")
        output = self.llm.create_chat_completion(
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

    @staticmethod
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

    def voice_speech(self, prompt, playback=True):
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav")
        self.tts.tts_to_file(text=prompt, speaker="Ana Florence", language="en", file_path=temp_file.name)
        if playback:
            print("Playing back")
            playsound(temp_file.name)
        temp_file.close()

    def process_answer(self, last_recorded_result=None):
        if last_recorded_result is None:
            last_recorded_result = self.last_recorded_result
        helper_answer = self.answer_speech(last_recorded_result)
        helper_answer = self.parse_output(helper_answer)
        self.voice_speech(helper_answer)


class NeonAssistant(AssistantModelsMixin, rumps.App):
    def __init__(self):
        super().__init__("NeonAssistant", icon="icon_pytorch.jpg")
        self.menu = ["Start Recording", "Stop Recording"]
        # Adding menu items
        self.audio = None
        self.wav_file = None
        self.temp_file = None

        # Initialize a flag to manage recording state
        self.is_recording = False
        self.recording_thread = None
        self.stream = None
        self.last_recorded_result = None

    @rumps.clicked("Start Recording")
    def start_recording(self, _):
        if not self.is_recording:
            self.is_recording = True
            self.start_recording_thread()
        else:
            rumps.alert(title="Already Recording", message="Recording is already in progress.")

    def start_recording_thread(self):
        # Start the recording process in a separate thread
        self.recording_thread = threading.Thread(target=self.start_recording_process)
        self.recording_thread.start()

    def start_recording_process(self):
        if self.is_recording:
            print("Started recording.")
            self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav")
            sample_rate = 16000
            bits_per_sample = 16
            chunk_size = 1024
            audio_format = pyaudio.paInt16
            channels = 1

            def callback(in_data, frame_count, time_info, status):
                self.wav_file.writeframes(in_data)
                return None, pyaudio.paContinue

            self.wav_file = wave.open(self.temp_file.name, 'wb')
            self.wav_file.setnchannels(channels)
            self.wav_file.setsampwidth(bits_per_sample // 8)
            self.wav_file.setframerate(sample_rate)
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(format=audio_format,
                                          channels=channels,
                                          rate=sample_rate,
                                          input=True,
                                          frames_per_buffer=chunk_size,
                                          stream_callback=callback)
        # while
        #     print("Recording...")  # Replace with actual recording logic
        #     time.sleep(1)  # Simulate time taken to record

    @rumps.clicked("Stop Recording")
    def stop_recording(self, _):
        if self.is_recording:
            self.is_recording = False
            self.stop_recording_thread()
        else:
            rumps.alert(title="Not Recording", message="No recording is in progress.")

    def stop_recording_thread(self):
        if self.recording_thread:  # and self.recording_thread.is_alive():
            self.recording_thread.join()
            # Stop and close the audio stream
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            self.wav_file.close()
            self.last_recorded_result = self.whisper_model.transcribe(self.temp_file.name)
            print(self.last_recorded_result)
            self.temp_file.close()
            self.recording_thread = None
            self.process_answer(self.last_recorded_result)
            print("Recording stopped.")  # Replace with any cleanup logic if necessary


if __name__ == "__main__":
    NeonAssistant().run()
