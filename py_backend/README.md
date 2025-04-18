Firstly install requirements: `pip install -r requirements.txt`\
To install llama-cpp-python follow instructions here: https://github.com/abetlen/llama-cpp-python\
To run using python: `python main.py`\
To build a standalone app: `python setup.py py2app`\
To run the app in console: `./dist/main.app/Contents/MacOS/main`\

To use the app: an icon will appear on the top right panel, press start recording, say something, then press stop
recording,
and the app will respond. For now only english is supported. The model is able to tell the time and current date, or
list the files in the local directory.