import openai
import os
import azure.cognitiveservices.speech as speechsdk
import gradio as gr

from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

class Chatbot:
    def __init__(self, tools=None):
        self.verbose = True
        self.tools = tools or []
        self.history = []
        self.agent_chain = initialize_agent(
            tools=self.tools, 
            llm=ChatOpenAI(temperature=0), 
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
            verbose=self.verbose, 
            memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
            max_iterations=2, 
            max_turns=2
        )

    def run(self, question):
        return self.agent_chain.run(input=question)

    def predict(self, input, history=[]):
        self.history.append(input)
        response = self.run(input)
        self.history.append(response)
        play_voice(response)
        responses = [(u,b) for u,b in zip(self.history[::2], self.history[1::2])]
        return responses, self.history
    
    def process_audio(self, audio, history=[]):
        text = self.transcribe(audio)
        return self.predict(text, history)

    def transcribe(self, audio):
        os.rename(audio, audio + '.wav')
        with open(audio + '.wav', "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            return transcript['text']

    def launch(self):
        with gr.Blocks(css="#chatbot{height:600px} .overflow-y-auto{height:600px}") as demo:
            chatbot = gr.Chatbot(elem_id="chatbot")
            state = gr.State([])

            with gr.Row():
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)

            with gr.Row():
                audio = gr.Audio(source="microphone", type="filepath")

            txt.submit(self.predict, [txt, state], [chatbot, state])
            audio.change(self.process_audio, [audio, state], [chatbot, state])

        demo.launch()

def play_voice(text):
    speech_config = speechsdk.SpeechConfig(subscription=os.environ.get('AZURE_SPEECH_KEY'), region=os.environ.get('AZURE_SPEECH_REGION'))
    audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    # The language of the voice that speaks.
    speech_config.speech_synthesis_language='zh-CN'
    speech_config.speech_synthesis_voice_name='zh-CN-XiaohanNeural'

    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    speech_synthesizer.speak_text_async(text)
