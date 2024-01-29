import torch
import streamlit as st
from configuration.models import *


@st.cache_data  # type: ignore
def load_sst_model(sst_type):
    if sst_type == "whisper":
        # Setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32 

        # Load Whisper model
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            WHISPER_MODEL, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        model.to(device)

        processor = AutoProcessor.from_pretrained(WHISPER_MODEL)

        model = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
        return model
    

from streamlit_mic_recorder import mic_recorder
import streamlit as st
import io
from openai import OpenAI
import dotenv
import os

# https://github.com/B4PT0R/streamlit-mic-recorder
def WhisperSTT(openai_api_key=None,start_prompt="Start - ⏺️",stop_prompt="Stop - ⏹️",just_once=False,use_container_width=False,language=None,callback=None,args=(),kwargs={},key=None):
    if not 'openai_client' in st.session_state:
        dotenv.load_dotenv()
        st.session_state.openai_client=OpenAI(api_key = os.getenv('OPENAI_API_KEY'))
    if not '_last_speech_to_text_transcript_id' in st.session_state:
        st.session_state._last_speech_to_text_transcript_id=0
    if not '_last_speech_to_text_transcript' in st.session_state:
        st.session_state._last_speech_to_text_transcript=None
    if key and not key+'_output' in st.session_state:
        st.session_state[key+'_output']=None
    audio = mic_recorder(start_prompt=start_prompt,stop_prompt=stop_prompt,just_once=just_once,use_container_width=use_container_width,key=key)
    new_output=False
    if audio is None:
        output=None
    else:
        id=audio['id']
        new_output=(id>st.session_state._last_speech_to_text_transcript_id)
        if new_output:
            output=None
            st.session_state._last_speech_to_text_transcript_id=id
            audio_BIO = io.BytesIO(audio['bytes'])
            audio_BIO.name='audio.mp3'
            success=False
            err=0
            while not success and err<3: #Retry up to 3 times in case of OpenAI server error.
                try:
                    transcript = st.session_state.openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_BIO,
                        language=language
                    )
                except Exception as e:
                    print(str(e)) # log the exception in the terminal
                    err+=1
                else:
                    success=True
                    output=transcript.text
                    st.session_state._last_speech_to_text_transcript=output
        elif not just_once:
            output=st.session_state._last_speech_to_text_transcript
        else:
            output=None

    if key:
        st.session_state[key+'_output']=output
    if new_output and callback:
        callback(*args,**kwargs)
    return output