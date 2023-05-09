# Chatbot README

This is a chatbot program that uses natural language processing to interact with users through text or voice. The program is built on the following frameworks:

- OpenAI
- Azure Cognitive Services Speech
- Gradio
- Langchain

## Installation

Before running the program, make sure to install the required packages by running:

```pip
pip3 install -r requirements.txt
```

You will also need to set up Azure Cognitive Services Speech by creating a Speech resource in the Azure Portal and obtaining a subscription key and region.

## Usage

To launch the chatbot, run the following code:

```python
from chatbot import Chatbot

chatbot = Chatbot()
chatbot.launch()
```

This will open a Gradio interface where users can input text or speak into the microphone. The chatbot will respond with text and voice.

The Chatbot class has the following methods:

- run(question): Takes a text input and returns the chatbot's response as a string.
- predict(input, history=[]): Takes a text input and returns a list of tuples containing the input and output messages, as well as the chat history as a list.
- process_audio(audio, history=[]): Takes a path to an audio file and returns the same output as predict.
- transcribe(audio): Takes a path to an audio file and returns the transcribed text using OpenAI's audio transcription API.
- launch(): Starts the Gradio interface.

The play_voice(text) function is used to play the chatbot's responses using Azure Cognitive Services Speech.

## Credits

This program was built by Langchain.
