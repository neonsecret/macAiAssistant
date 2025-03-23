import asyncio
import base64
import datetime
import io
import os
import subprocess
import sys
import sys
import tempfile
import threading
import time
import wave
from contextlib import contextmanager

import pyaudio
import torch
import webrtcvad
from PIL import ImageGrab
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler
from modules.custom_functions import execute_function_call, create_tool_schema, AssistantFunctions
from playsound import playsound
from pywhispercpp.model import Model
from transformers import AutoTokenizer

os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS

DEBUG = True
LIGHT_RUN = False

sys.path.append("functionary")
LLM_TOOLS = create_tool_schema(AssistantFunctions)
TODAY_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
LLM_SYS_PROMPT = f"""You are a system management assistant with access to functions.
Current date: {TODAY_DATE}.
If multiple functions could apply, choose the one that best addresses the query’s intent.
If you require to use the command prompt to run a command or retrieve some information there is no function for, 
use the run_terminal_command function.
Always ensure that your responses and function calls are safe, accurate, and contextually relevant."""


@contextmanager
def change_dir(destination):
    origin = os.getcwd()
    os.chdir(destination)
    try:
        yield
    finally:
        os.chdir(origin)


try:
    with change_dir("functionary"):
        from functionary.prompt_template import get_prompt_template_from_tokenizer
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "functionary"])
    with change_dir("functionary"):
        from functionary.prompt_template import get_prompt_template_from_tokenizer


class AssistantModelsMixin:
    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.last_recorded_result = None
        self.chat_handler = None
        self.use_vision = False
        if not LIGHT_RUN:
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
                    repo_id="meetkai/functionary-small-v2.5-GGUF",
                    filename="*Q4*",
                    n_ctx=8192,
                    n_gpu_layers=-1,
                    verbose=False
                )
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "meetkai/functionary-small-v2.5-GGUF", legacy=True
                )
                with change_dir("functionary"):
                    self.prompt_template = get_prompt_template_from_tokenizer(self.tokenizer)
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
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)
        self.functions = AssistantFunctions()

    def dialogue_call(self, user_input: str = None, messages: list = None, is_function_call: bool = True):
        # Firstly, the user prompts the llm for a function call
        # After the function call, llm receives the function output and provides a textual interpretation

        if not messages:  # this is a function call
            messages = [
                {"role": "system", "content": LLM_SYS_PROMPT},
                {'role': 'user', 'content': user_input},
                {'role': 'assistant'}
            ]
        if is_function_call:
            prompt_str = self.prompt_template.get_prompt_from_messages(messages, LLM_TOOLS)
            token_ids = self.tokenizer.encode(prompt_str)

            gen_tokens = []
            # Get list of stop_tokens
            stop_token_ids = [
                self.tokenizer.encode(token)[-1]
                for token in self.prompt_template.get_stop_tokens_for_generation()
            ]
            for token_id in self.llm.generate(token_ids, temp=0.8):
                if token_id in stop_token_ids:
                    break
                gen_tokens.append(token_id)

            llm_output = self.tokenizer.decode(gen_tokens)
            initial_result = self.prompt_template.parse_assistant_response(llm_output)
            if 'tool_calls' in initial_result:
                for tool_call in initial_result['tool_calls']:
                    result = execute_function_call(tool_call)
                    print(f"Function {tool_call['function']['name']} returned: {result}")
            else:
                raise NotImplementedError(str(initial_result))
        else:
            print(messages)
            result = self.llm.create_chat_completion(
                messages,
                # temperature=0.7
            )
            result = result["choices"][0]["message"]["content"]

        print("\n", is_function_call, result)
        if is_function_call:
            messages = [
                {"role": "system", "content": LLM_SYS_PROMPT},
                {'role': 'user', 'content': user_input},
                initial_result,
                {'role': 'function', "name": tool_call['function']['name'], 'content': result},
            ]
            return self.dialogue_call(messages=messages, is_function_call=False)
        return result

    def answer_speech_vision(self, prompt, img=None, default_formatting=True):  # vision
        assert self.use_vision
        # print("Vision Infer start")
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
            # print("Playing back")
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


class NeonAssistant(AssistantModelsMixin):
    def __init__(self, manager, loop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ready = False
        self.audio = None
        self.wav_file = None
        self.temp_file = None
        self.temp_dir = tempfile.TemporaryDirectory()

        # Initialize flags to manage recording state
        self.is_recording = False
        self.recording_thread = None
        self.stream = None
        self.last_recorded_result = None
        self.confidence_counter = None
        self.user_talking = None

        self.stop_event = threading.Event()

        self.loop = loop
        self.manager = manager
        asyncio.run_coroutine_threadsafe(
            self.manager.broadcast("NeonAssistant initialized"),
            self.loop
        )
        self.ready = True

    def start_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.start_recording_thread()

            asyncio.run_coroutine_threadsafe(
                self.manager.broadcast("Recording started"),
                self.loop
            )

    def start_recording_thread(self):
        self.recording_thread = threading.Thread(target=self.start_recording_process)
        self.recording_thread.start()

    def start_recording_process(self):
        if self.is_recording:
            print("Started recording.")
            self.temp_file = os.path.join(self.temp_dir.name, 'recording_process.wav')
            sample_rate = 16000
            bits_per_sample = 16
            chunk_size = 480
            audio_format = pyaudio.paInt16
            channels = 1
            self.confidence_counter = 0
            self.user_talking = False

            self.stop_event.clear()

            def callback(in_data, frame_count, time_info, status):
                self.wav_file.writeframes(in_data)
                # audio = np.frombuffer(in_data, dtype=np.int16
                is_speech = self.vad.is_speech(in_data, sample_rate)
                if not self.user_talking and is_speech:
                    self.confidence_counter += 1
                    if self.confidence_counter >= 5:
                        self.user_talking = True
                        self.confidence_counter = 0
                        print("user talks now")
                        asyncio.run_coroutine_threadsafe(
                            self.manager.broadcast("User talks now."),
                            self.loop
                        )
                if self.user_talking and not is_speech:
                    self.confidence_counter += 1
                    if self.confidence_counter >= 5:
                        print("user stopped talking")
                        asyncio.run_coroutine_threadsafe(
                            self.manager.broadcast("User stopped talking."),
                            self.loop
                        )
                        self.user_talking = False
                        self.confidence_counter = 0

                        self.stop_event.set()
                return None, pyaudio.paContinue

            self.wav_file = wave.open(self.temp_file, 'wb')
            self.wav_file.setnchannels(channels)
            self.wav_file.setsampwidth(bits_per_sample // 8)
            self.wav_file.setframerate(sample_rate)
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(format=audio_format,
                                          channels=channels,
                                          rate=sample_rate,
                                          input=True,
                                          frames_per_buffer=chunk_size,
                                          stream_callback=callback,
                                          input_device_index=0)
            self.stream.start_stream()
            print("started stream..")

            # Main loop: wait until the stop event is signaled.
            while not self.stop_event.is_set():
                time.sleep(0.1)

            self.cleanup_recording()

    def cleanup_recording(self):
        print("stopping rec thread")
        # self.recording_thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        self.wav_file.close()
        self.last_recorded_result = self.whisper_model.transcribe(self.temp_file)[0].text
        print(self.last_recorded_result)

        self.is_recording = False
        self.recording_thread = None
        self.process_answer(self.last_recorded_result)
        print("Recording stopped.")
        asyncio.run_coroutine_threadsafe(
            self.manager.broadcast("Recording stopped."),
            self.loop
        )

    def debug_run(self, prompt):
        print("debug run")
        self.process_answer(prompt, voice=False)

    def __del__(self):  # destructor
        try:
            self.temp_dir.cleanup()
            self.llm._sampler.close()
            self.llm.close()
        except:
            pass


if __name__ == "__main__":
    assistant = NeonAssistant()
    assistant.start_recording()
    # assistant.debug_run("How old is adrien brody?")
    del assistant  # cleanup
