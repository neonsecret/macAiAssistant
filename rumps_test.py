from llama_cpp import Llama
import rumps


class AwesomeStatusBarApp(rumps.App):
    def __init__(self):
        super(AwesomeStatusBarApp, self).__init__("infer", icon="icon_pytorch.jpg")
        self.menu = ["Run"]
        self.prompt = rumps.Window("Enter your prompt").run().text

    @rumps.clicked("Run")
    def prefs(self, _):
        out = infer(self.prompt)
        rumps.alert(out)


def infer(prompt="hehe"):
    print("Loading")
    llm = Llama.from_pretrained(
        repo_id="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF",
        filename="*Q8.gguf",
        verbose=False,
        n_gpu_layers=-1
    )
    print("Infer start")
    output = llm(
        prompt,  # Prompt
        max_tokens=256,  # Generate up to 32 tokens, set to None to generate up to the end of the context window
        stop=["Q:", "\n"],  # Stop generating just before the model would generate a new question
        echo=True  # Echo the prompt back in the output
    )  # Generate a completion, can also call create_completion
    print(output)
    print(output["choices"][0]["text"])
    return output["choices"][0]["text"]


if __name__ == "__main__":
    AwesomeStatusBarApp().run()
