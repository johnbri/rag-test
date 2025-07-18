import os
import streamlit as st
import tempfile
import uuid
import logging
import speech_recognition as sr
import shutil
import atexit
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from elevenlabs.client import ElevenLabs
import pygame
import threading
import time
import base64

# Set page config FIRST
st.set_page_config(page_title="Voice RAG Chatbot", page_icon="🎤")

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ElevenLabs client
@st.cache_resource
def init_elevenlabs():
    return ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

client = init_elevenlabs()

class VoiceRAGChatbot:
    def __init__(self):
        self.vectorstore = None
        self.retriever = None
        self.persist_directory = None
        self.llm = Ollama(model="llama3.1:8b")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize pygame mixer
        pygame.mixer.init()
        
        # Calibrate microphone
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
    
    def cleanup_vectorstore(self):
        """Clean up existing vectorstore and temporary files"""
        if self.persist_directory and os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                logger.info(f"Cleaned up vectorstore directory: {self.persist_directory}")
            except Exception as e:
                logger.warning(f"Could not clean up vectorstore directory: {e}")
        
        self.vectorstore = None
        self.retriever = None
        self.persist_directory = None
    
    def reset_session(self):
        """Reset all session data including documents, memory, and vectorstore"""
        # Clear vectorstore and documents
        self.cleanup_vectorstore()
        
        # Clear conversation memory
        self.memory.clear()
        
        logger.info("Session reset - all documents and conversation history cleared")
            
    def load_documents(self, uploaded_files):
        documents = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            try:
                if uploaded_file.type == "application/pdf":
                    loader = PyPDFLoader(tmp_file_path)
                elif uploaded_file.type == "text/plain":
                    loader = TextLoader(tmp_file_path)
                else:
                    st.error(f"Unsupported file type: {uploaded_file.type}")
                    continue
                
                docs = loader.load()
                documents.extend(docs)
            finally:
                os.unlink(tmp_file_path)
        
        return documents
    
    def setup_vectorstore(self, documents):
        # Clean up any existing vectorstore first
        self.cleanup_vectorstore()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        split_docs = text_splitter.split_documents(documents)
        
        # Create a fresh vectorstore with unique collection name and temp directory
        collection_name = f"documents_{uuid.uuid4().hex[:8]}"
        
        # Create a temporary directory for ChromaDB
        self.persist_directory = tempfile.mkdtemp()
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                collection_name=collection_name,
                persist_directory=self.persist_directory
            )
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
            
            logger.info(f"Successfully created vectorstore with {len(split_docs)} chunks")
            return len(split_docs)
            
        except Exception as e:
            logger.error(f"Error creating vectorstore: {e}")
            self.cleanup_vectorstore()
            raise e
    
    def create_chain(self):
        if not self.retriever:
            raise ValueError("Vectorstore not initialized. Please upload documents first.")
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            verbose=True
        )
        
        return chain
    
    def chat(self, question):
        chain = self.create_chain()
        response = chain({"question": question})
        return response["answer"], response["source_documents"]
    
    def listen_for_speech(self, duration=5):
        """Listen for speech input and return transcribed text"""
        try:
            with self.microphone as source:
                audio = self.recognizer.listen(source, timeout=duration, phrase_time_limit=10)
                
            text = self.recognizer.recognize_google(audio)
            return text
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            st.error(f"Error with speech recognition: {e}")
            return None
    
    def generate_speech(self, text: str):
        """Generate speech audio from text using ElevenLabs"""
        try:
            audio_generator = client.text_to_speech.convert(
                text=text,
                voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice ID
                model_id="eleven_monolingual_v1"
            )
            
            # Convert generator to bytes
            audio_bytes = b"".join(audio_generator)
            return audio_bytes
            
        except Exception as e:
            logger.error(f"Error with text-to-speech: {e}")
            return None

def play_audio(audio_bytes):
    """Play audio using pygame with interruption support"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        pygame.mixer.music.load(tmp_file_path)
        pygame.mixer.music.play()
        
        # Set audio playing state after starting playback
        st.session_state.audio_playing = True
        st.session_state.stop_audio = False
        
        # Wait for playback to finish or be interrupted
        while pygame.mixer.music.get_busy():
            if st.session_state.stop_audio:
                pygame.mixer.music.stop()
                logger.info("Audio playback stopped by user")
                break
            pygame.time.wait(100)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
    finally:
        # Reset audio state
        st.session_state.audio_playing = False
        st.session_state.stop_audio = False

def stop_audio():
    """Stop currently playing audio"""
    st.session_state.stop_audio = True
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            logger.info("Audio stopped by user")
    except pygame.error:
        logger.warning("Could not stop audio - pygame not initialized")
    st.session_state.audio_playing = False

def audio_to_base64(audio_bytes):
    """Convert audio bytes to base64 for HTML audio player"""
    return base64.b64encode(audio_bytes).decode()

def handle_voice_input(voice_enabled):
    """Handle voice input processing"""
    # Stop any currently playing audio when starting new input
    if st.session_state.audio_playing or pygame.mixer.music.get_busy():
        stop_audio()
    
    with st.spinner("Listening... Speak now!"):
        st.session_state.is_listening = True
        user_input = st.session_state.chatbot.listen_for_speech(duration=8)
        st.session_state.is_listening = False
        
        if user_input:
            st.success(f"You said: {user_input}")
            
            # Check for conversation exit commands
            if st.session_state.conversation_mode and user_input.lower() in ['stop conversation', 'goodbye', 'end conversation', 'quit']:
                st.session_state.conversation_mode = False
                st.session_state.conversation_active = False
                st.info("Conversation mode ended")
                st.rerun()
                return
            
            # Process the voice input
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            with st.spinner("Thinking..."):
                response, sources = st.session_state.chatbot.chat(user_input)
                
                # Generate speech for response
                audio_bytes = None
                if voice_enabled:
                    audio_bytes = st.session_state.chatbot.generate_speech(response)
                
                # Add assistant response
                message_data = {"role": "assistant", "content": response}
                if audio_bytes:
                    message_data["audio"] = audio_bytes
                
                st.session_state.messages.append(message_data)
                
                # Play audio
                if audio_bytes and voice_enabled:
                    threading.Thread(target=play_audio, args=(audio_bytes,)).start()
                
                # Show sources
                if sources:
                    source_text = "**Sources:**\n"
                    for i, source in enumerate(sources):
                        source_text += f"- Source {i+1}: {source.page_content[:100]}...\n"
                    st.info(source_text)
            
            st.rerun()
        else:
            st.warning("Could not understand audio. Please try again.")

def cleanup_on_exit():
    """Clean up session data when the app exits"""
    if hasattr(st.session_state, 'chatbot'):
        st.session_state.chatbot.cleanup_vectorstore()
        logger.info("Cleaned up vectorstore on app exit")

def main():
    st.title("🎤 Voice RAG Chatbot with Document Upload")
    st.markdown("Upload documents and interact with them using voice and text")
    
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = VoiceRAGChatbot()
        # Register cleanup function for when the app exits
        atexit.register(cleanup_on_exit)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "is_listening" not in st.session_state:
        st.session_state.is_listening = False
    
    if "conversation_mode" not in st.session_state:
        st.session_state.conversation_mode = False
    
    if "conversation_active" not in st.session_state:
        st.session_state.conversation_active = False
    
    if "audio_playing" not in st.session_state:
        st.session_state.audio_playing = False
    
    if "stop_audio" not in st.session_state:
        st.session_state.stop_audio = False
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            accept_multiple_files=True,
            type=['pdf', 'txt']
        )
        
        if uploaded_files and st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                documents = st.session_state.chatbot.load_documents(uploaded_files)
                num_chunks = st.session_state.chatbot.setup_vectorstore(documents)
                st.success(f"Processed {len(documents)} documents into {num_chunks} chunks")
        
        # Session status
        st.subheader("📊 Session Status")
        if st.session_state.chatbot.retriever:
            st.success("✅ Documents loaded and ready for questions")
        else:
            st.info("ℹ️ No documents loaded - upload documents to get started")
        
        # Clear session button
        if st.button("🗑️ Clear Session", help="Remove all uploaded documents and conversation history"):
            # Stop any playing audio
            if st.session_state.audio_playing:
                stop_audio()
            
            st.session_state.chatbot.reset_session()
            st.session_state.messages = []
            st.session_state.conversation_mode = False
            st.session_state.conversation_active = False
            st.session_state.is_listening = False
            st.session_state.audio_playing = False
            st.session_state.stop_audio = False
            st.success("Session cleared! All documents and conversation history removed.")
            st.rerun()
        
        st.divider()
        
        # Voice settings
        st.header("🎙️ Voice Settings")
        voice_enabled = st.toggle("Enable Voice Response", value=True)
        
        if voice_enabled:
            st.info("Voice responses are enabled using ElevenLabs")
        
        st.divider()
        
        # Conversation mode
        st.header("💬 Conversation Mode")
        st.markdown("Switch between single questions and continuous conversation")
        
        if not st.session_state.conversation_active:
            if st.button("🎤 Start Conversation Mode", type="primary"):
                st.session_state.conversation_mode = True
                st.session_state.conversation_active = True
                st.rerun()
        else:
            if st.button("🛑 Stop Conversation Mode", type="secondary"):
                st.session_state.conversation_mode = False
                st.session_state.conversation_active = False
                st.session_state.is_listening = False
                st.rerun()
            
            st.success("🎤 Conversation mode is active")
            st.info("The chatbot will continuously listen for your voice input")
    
    # Main chat interface
    if st.session_state.conversation_mode:
        # Conversation mode - continuous voice chat
        st.markdown("### 💬 Conversation Mode Active")
        
        # Display chat messages (no audio controls in conversation mode)
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Conversation mode interface
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            if st.session_state.conversation_active and not st.session_state.is_listening:
                if st.button("🎤 Speak Now", type="primary", use_container_width=True):
                    if not st.session_state.chatbot.retriever:
                        st.error("Please upload and process documents first!")
                    else:
                        handle_voice_input(voice_enabled)
            
            if st.session_state.is_listening:
                st.markdown("🔴 **Listening... Speak now!**")
                st.info("Say 'stop conversation' or 'goodbye' to end the conversation")
            
            # Audio status indicator
            if st.session_state.audio_playing or pygame.mixer.music.get_busy():
                st.markdown("🔊 **Audio playing...**")
                st.info("Click 'Stop Audio' to interrupt")
        
        with col2:
            # Stop audio button - check both session state and pygame state
            audio_is_playing = st.session_state.audio_playing or pygame.mixer.music.get_busy()
            
            if audio_is_playing:
                if st.button("🔇 Stop Audio", type="secondary", use_container_width=True):
                    stop_audio()
                    st.rerun()
            else:
                st.button("🔇 Stop Audio", disabled=True, use_container_width=True, help="No audio playing")
        
        with col3:
            if st.button("💬 Chat Mode", use_container_width=True):
                st.session_state.conversation_mode = False
                st.session_state.conversation_active = False
                st.session_state.is_listening = False
                # Stop any playing audio when switching modes
                if st.session_state.audio_playing:
                    stop_audio()
                st.rerun()
    
    else:
        # Regular chat mode
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Play audio if it exists
                    if message.get("audio") and voice_enabled:
                        audio_b64 = audio_to_base64(message["audio"])
                        st.markdown(f"""
                        <audio controls>
                            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                            Your browser does not support the audio element.
                        </audio>
                        """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 🎤 Voice Input")
            
            # Single voice input button
            if st.button("🎙️ Record Voice", disabled=st.session_state.is_listening):
                if not st.session_state.chatbot.retriever:
                    st.error("Please upload and process documents first!")
                else:
                    handle_voice_input(voice_enabled)
            
            if st.session_state.is_listening:
                st.markdown("🔴 **Listening...**")
    
    # Text input (only show in chat mode)
    if not st.session_state.conversation_mode:
        if prompt := st.chat_input("Ask a question about your documents"):
            if not st.session_state.chatbot.retriever:
                st.error("Please upload and process documents first!")
            else:
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response, sources = st.session_state.chatbot.chat(prompt)
                        st.markdown(response)
                        
                        # Generate speech for response
                        audio_bytes = None
                        if voice_enabled:
                            audio_bytes = st.session_state.chatbot.generate_speech(response)
                        
                        # Add assistant response
                        message_data = {"role": "assistant", "content": response}
                        if audio_bytes:
                            message_data["audio"] = audio_bytes
                            
                            # Play audio
                            threading.Thread(target=play_audio, args=(audio_bytes,)).start()
                            
                            # Show audio player
                            audio_b64 = audio_to_base64(audio_bytes)
                            st.markdown(f"""
                            <audio controls>
                                <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
                                Your browser does not support the audio element.
                            </audio>
                            """, unsafe_allow_html=True)
                        
                        st.session_state.messages.append(message_data)
                        
                        # Show sources
                        if sources:
                            with st.expander("Source Documents"):
                                for i, source in enumerate(sources):
                                    st.markdown(f"**Source {i+1}:**")
                                    st.markdown(source.page_content[:500] + "...")
                                    st.markdown(f"*File: {source.metadata.get('source', 'Unknown')}*")
                                    st.markdown("---")
    else:
        # In conversation mode, show instructions
        st.info("💬 **Conversation Mode Active** - Use the 'Speak Now' button above to continue the conversation, or switch back to Chat Mode to type messages.")

if __name__ == "__main__":
    main()