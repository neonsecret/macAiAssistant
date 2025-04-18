import asyncio
import datetime
import os
import subprocess
import sys
import tempfile
import threading
import time
import wave
from contextlib import contextmanager

import pyaudio
import torch
import webrtcvad
from llama_cpp import Llama
from playsound import playsound
from pywhispercpp.model import Model
from transformers import AutoTokenizer

from modules.custom_functions import execute_function_call, create_tool_schema, AssistantFunctions

os.environ["COQUI_TOS_AGREED"] = "1"
from TTS.api import TTS

DEBUG = True
LIGHT_RUN = False

sys.path.append("functionary")
LLM_TOOLS = create_tool_schema(AssistantFunctions)
TODAY_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
LLM_SYS_PROMPT = f"""You are a helpful system management assistant with access to functions.
Current date: {TODAY_DATE}.
Your goal is to assist the user with their request by utilizing the available tools.
Think step-by-step. First, understand the user's request. Then, determine if any tools are needed to fulfill the request.
You can call multiple tools in sequence if necessary (e.g., search online, then add a calendar event).
If you need to run a command in the terminal, use the 'run_terminal_command' function, but be mindful of security.
If an app you need to open is not installed, try searching for a similar app, or open a website of the app, for example for youtube.
Once you have gathered all the necessary information and performed the required actions, you MUST call the 'speak' function.
The 'speak' function is your way to communicate the final answer or confirmation back to the user.
Do NOT provide a response directly in text format. Always use the 'speak' function for your final output to the user.
If the user's request doesn't require any tools, call the 'speak' function directly with your response.
If a function call failed, conclude with 'speak' and explain what happened.
Always ensure your function calls are accurate and safe. Call 'speak' to conclude."""


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
        self.last_recorded_result = None
        if not LIGHT_RUN:
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

            self.device = torch.device("cpu")
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            self.whisper_model = Model('base', n_threads=6)
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)
        self.functions = AssistantFunctions()

    def dialogue_loop(self, user_input: str) -> str:
        messages = [
            {"role": "system", "content": LLM_SYS_PROMPT},
            {'role': 'user', 'content': user_input},
        ]
        max_turns = 10  # Add a limit to prevent infinite loops
        turn_count = 0

        while turn_count < max_turns:
            turn_count += 1
            messages.append({'role': 'assistant', 'content': None})
            prompt_str = self.prompt_template.get_prompt_from_messages(messages, LLM_TOOLS)
            token_ids = self.tokenizer.encode(prompt_str)

            stop_token_ids = [
                self.tokenizer.encode(token)[-1]
                for token in self.prompt_template.get_stop_tokens_for_generation()
            ]
            output_token_ids = self.llm.generate(token_ids, temp=0.7)  # Adjust temp as needed
            filtered_tokens = []
            for token_id in output_token_ids:
                if token_id in stop_token_ids:
                    break
                filtered_tokens.append(token_id)

            llm_output_text = self.tokenizer.decode(filtered_tokens)
            messages.pop()

            if not llm_output_text.strip():
                return "The assistant did not provide a response."

            try:
                parsed_response = self.prompt_template.parse_assistant_response(llm_output_text)
                messages.append(parsed_response)
            except:
                if "tool_calls" not in llm_output_text and "\"function\":" not in llm_output_text:
                    # Assume it's a direct answer if parsing fails and no function call syntax found
                    print("No function call detected.", llm_output_text)
                    return "No function call detected."
                else:
                    return f"Error understanding the assistant's response format. Raw: {llm_output_text}"

            if 'tool_calls' in parsed_response and parsed_response['tool_calls']:
                tool_call = parsed_response['tool_calls'][0]  # Process one call per turn
                func_name = tool_call['function']['name']
                print("Calling", tool_call)
                try:
                    function_result = execute_function_call(tool_call)
                except:
                    function_result = "Function call failed."
                print(f"Function call returned: {function_result}")
                if isinstance(function_result, bool):
                    function_result = "Success." if function_result else "Function failed."

                if func_name == "speak":
                    return function_result

                messages.append({
                    'role': 'function',
                    "name": func_name,
                    'content': function_result
                })
            elif 'content' in parsed_response and parsed_response['content']:
                print("No function call only speech.")
                content = parsed_response['content'].strip()
                try:
                    content = content.split("speak")[1].strip().replace("\n", "")
                    content = list(eval(content).values())[0]
                except Exception as e:
                    print(content, e)
                return content
            else:
                return "Assistant response format was unclear. No action taken."

        return "Assistant reached maximum turns without providing a final answer."

    def voice_speech(self, prompt, playback=True):
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav")
        self.tts.tts_to_file(text=prompt, speaker="Ana Florence", language="en", file_path=temp_file.name)
        if playback:
            playsound(temp_file.name)
        temp_file.close()

    def process_answer(self, last_recorded_result=None, voice=True):
        if last_recorded_result is None:
            last_recorded_result = self.last_recorded_result

        helper_answer = self.dialogue_loop(last_recorded_result)

        if voice:
            self.voice_speech(helper_answer)
        else:
            print("Printing silently helper answer:")
            print(helper_answer)


class NeonAssistant(AssistantModelsMixin):
    def __init__(self, manager, loop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ready = False
        self.audio = None
        self.wav_file = None
        self.temp_file = None
        self.temp_dir = tempfile.TemporaryDirectory()

        self.is_recording = False
        self.recording_thread = None
        self.stream = None
        self.last_recorded_result = None
        self.confidence_counter = None
        self.user_talking = None

        self.stop_event = threading.Event()

        self.loop = loop
        self.manager = manager
        if manager is not None and loop is not None:
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
                is_speech = self.vad.is_speech(in_data, sample_rate)
                if not self.user_talking and is_speech:
                    self.confidence_counter += 1
                    if self.confidence_counter >= 5:
                        self.user_talking = True
                        self.confidence_counter = 0
                        print("user talks now")
                        asyncio.run_coroutine_threadsafe(
                            self.manager.broadcast("Speak, please."),
                            self.loop
                        )
                if self.user_talking and not is_speech:
                    self.confidence_counter += 1
                    if self.confidence_counter >= 20:
                        print("user stopped talking")
                        asyncio.run_coroutine_threadsafe(
                            self.manager.broadcast("You stopped talking."),
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

            while not self.stop_event.is_set():
                time.sleep(0.1)

            self.cleanup_recording()

    def cleanup_recording(self):
        print("stopping rec thread")
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
        # asyncio.run_coroutine_threadsafe(
        #     self.manager.broadcast("Recording stopped."),
        #     self.loop
        # )

    def debug_run(self, prompt):
        print("debug run")
        self.process_answer(prompt, voice=False)

    def __del__(self):
        try:
            self.temp_dir.cleanup()
            # Check if llm object and _sampler exist before trying to close
            if hasattr(self, 'llm') and self.llm is not None:
                if hasattr(self.llm, '_sampler') and self.llm._sampler is not None:
                    self.llm._sampler.close()
                self.llm.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def voice_speech(self, prompt, playback=True):
        super().voice_speech(prompt, playback=playback)
        if len(prompt) > 100:
            prompt = prompt[:100] + ".."
        time.sleep(2)
        asyncio.run_coroutine_threadsafe(
            self.manager.broadcast(prompt),
            self.loop
        )


if __name__ == "__main__":
    assistant = NeonAssistant(None, None)
    # assistant.start_recording()
    assistant.debug_run("Test cue")
    del assistant  # cleanup
