import base64
import io
import os
import tempfile
import threading
import wave
from datetime import datetime

import pyaudio
import rumps
import torch
from PIL import ImageGrab
from TTS.api import TTS
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler, Llava16ChatHandler, MoondreamChatHandler
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
from playsound import playsound
from pywhispercpp.model import Model

import custom_functions
from custom_functions import convert_func_args


class AssistantModelsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_recorded_result = None
        self.chat_handler = None
        self.use_vision = False
        if not self.use_vision:
            # self.llm = Llama.from_pretrained(
            #     repo_id="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
            #     filename="*F16*",
            #     verbose=False,
            #     n_gpu_layers=-1,
            #     n_ctx=2048,
            #     chat_format="chatml-function-calling"
            # )
            self.llm = Llama.from_pretrained(
                repo_id="meetkai/functionary-small-v2.2-GGUF",
                filename="functionary-small-v2.2.q4_0.gguf",
                chat_format="functionary-v2",
                tokenizer=LlamaHFTokenizer.from_pretrained("meetkai/functionary-small-v2.2-GGUF"),
                n_gpu_layers=-1,
                verbose=False
            )
        else:
            self.chat_handler = MoondreamChatHandler.from_pretrained(
                "vikhyatk/moondream2",
                filename="*mmproj*")
            self.llm = Llama.from_pretrained(
                repo_id="vikhyatk/moondream2",
                filename="*text-model*",
                verbose=False,
                chat_handler=self.chat_handler,
                n_gpu_layers=-1,
                n_ctx=2048
            )

        self.device = torch.device("cpu")
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        self.whisper_model = Model('base', n_threads=6)
        self.functions = custom_functions.AssistantFunctions()

    def answer_speech(self, prompt, default_formatting=True):
        print("Infer start")
        tools, tool_choice = convert_func_args()
        sys_prompt = ("You are a helpful assistant. You give helpful, detailed, and polite answers to the user's "
                      "questions. You may call functions with appropriate input when necessary.")
        output = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=0.5,
        )
        print(output)
        if "function_call" in output["choices"][0]["message"]:
            print(output["choices"][0]["message"]["function_call"])
            func_name = output["choices"][0]["message"]["function_call"]["name"].replace(":", "")
            func_args = output["choices"][0]["message"]["function_call"]["arguments"].strip()
            func_args = {} if func_args == "{}" else func_args
            print(func_args)
            actual_func = getattr(self.functions, func_name)
            out = actual_func(*func_args)
        else:
            out = output["choices"][0]["message"]
            if default_formatting:
                out = out["content"]
        return out

    def answer_speech_vision(self, prompt, img=None, default_formatting=True):  # vision
        assert self.use_vision
        print("Vision Infer start")
        msg_content = [
            {"type": "text", "text": prompt}
        ]
        if img is not None:
            msg_content += [{"type": "image_url", "image_url": {"url": img}}]

        output = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful computer assistant who also perfectly describes images. "
                               "If you are asked about time, only return one word: time; "
                               "If you are asked about the screen of the computer, describe what you see on the screen"
                },
                {"role": "user",
                 "content": msg_content},
            ],
            temperature=0.5,
        )
        # print(output)
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
            case "screen":
                return "SCREEN"
            case _:
                return response

    def voice_speech(self, prompt, playback=True):
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav")
        self.tts.tts_to_file(text=prompt, speaker="Ana Florence", language="en", file_path=temp_file.name)
        if playback:
            print("Playing back")
            playsound(temp_file.name)
        temp_file.close()

    def process_answer(self, last_recorded_result=None, voice=True):
        if last_recorded_result is None:
            last_recorded_result = self.last_recorded_result
        if self.use_vision:
            print("Grabbing a screenshot..")
            img = self.screenshot()
            helper_answer = self.answer_speech_vision(last_recorded_result, img=img)
            helper_answer = self.parse_output(helper_answer)
        else:
            helper_answer = self.answer_speech(last_recorded_result)
            helper_answer = self.parse_output(helper_answer)
        if voice:
            self.voice_speech(helper_answer)
        else:
            print("Printing silently:")
            print(helper_answer)

    @staticmethod
    def screenshot():
        img = ImageGrab.grab()
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        base64_data = base64.b64encode(img_byte_arr).decode('utf-8')
        return f"data:image/png;base64,{base64_data}"

    @staticmethod
    def image_to_base64_data_uri(file_path):
        with open(file_path, "rb") as img_file:
            base64_data = base64.b64encode(img_file.read()).decode('utf-8')
            return f"data:image/png;base64,{base64_data}"


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

    def debug_run(self, prompt):
        print("debug run")
        self.process_answer(prompt, voice=False)


if __name__ == "__main__":
    debug = True

    assistant = NeonAssistant()
    if debug:
        assistant.debug_run("What operation system am I using?")
    else:
        assistant.run()
