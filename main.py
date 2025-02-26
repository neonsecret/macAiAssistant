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
from custom_functions import convert_func_args_llama
import xml.etree.ElementTree as ET

LLM_SYS_PROMPT = f"""
You are an AI assistant with tool access. Follow these steps:

1. Analyze user request
2. Choose appropriate tool if needed
3. Format tool call as XML
4. Process tool response
5. Deliver final answer

Available functions to call:
{custom_functions.class_to_r1_function_schema(custom_functions.AssistantFunctions)}

Output format:
<tool_name>
<parameter>value</parameter>
</tool_name>
"""


class AssistantModelsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_recorded_result = None
        self.chat_handler = None
        self.use_vision = False
        if not self.use_vision:  # just LLM
            # self.llm = Llama.from_pretrained(
            #     repo_id="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
            #     filename="*F16*",
            #     verbose=False,
            #     n_gpu_layers=-1,
            #     n_ctx=2048,
            #     chat_format="chatml-function-calling"
            # )
            self.llm = Llama.from_pretrained(
                repo_id="bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
                filename="*Q4_K_M*",
                n_ctx=8192,
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

    def dialogue_call(self, user_input: str, default_formatting: bool = True):
        def execute_tool(tool_xml):
            try:
                root = ET.fromstring(tool_xml)
                function_name = root.text  # should be tag actually
                params = {child.tag: child.text for child in root}
                return function_name, params

            except ET.ParseError:
                return "Invalid XML format"

        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": LLM_SYS_PROMPT,
                },
                {"role": "user", "content": user_input},
            ],
            temperature=0.5,
            stop=["<|im_end|>"]
        )
        full_response = response["choices"][0]["message"]["content"]
        print(full_response)

        # thought = full_response.split("<think>")[1].split("</think>")[0]
        # print(f"Model Reasoning: {thought}")

        tool_call = full_response.split("</think>")[1].strip()
        print(tool_call)
        func_name, params = execute_tool(tool_call)
        print("Tool: ", func_name, params)
        method = getattr(custom_functions.AssistantFunctions, func_name)
        output = method(**params)
        print("Output: ", output)

        return "hehe"

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
            # helper_answer = self.parse_output(helper_answer)
        else:
            helper_answer = self.dialogue_call(last_recorded_result)
        if voice:
            self.voice_speech(helper_answer)
        else:
            print("Printing silently helper answer:")
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
        assistant.debug_run("Find out the current year with an online search.")
    else:
        assistant.run()
