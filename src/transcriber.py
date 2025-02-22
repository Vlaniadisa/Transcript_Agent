import os
from pathlib import Path
import json
import openai
from datetime import datetime
import time
from pydub import AudioSegment

class AudioTranscriber:
    def __init__(self, api_key: str):
        """Initialize the transcriber with OpenAI API key"""
        print("Initializing OpenAI client...")
        if not api_key:
            raise ValueError("OpenAI API key is missing!")
        self.client = openai.OpenAI(api_key=api_key)
    
    def _split_audio(self, audio_path: str):
        """Split audio file into chunks under 25MB"""
        print("Splitting audio into chunks...")
        
        audio = AudioSegment.from_file(audio_path)
        # Split into 2-minute chunks (120 seconds) to stay well under 25MB
        chunk_length = 120 * 1000  # milliseconds
        chunks = []
        
        for i, start in enumerate(range(0, len(audio), chunk_length)):
            chunk = audio[start:start + chunk_length]
            chunk_path = f"chunk_{i}.wav"
            # Export as mono audio to reduce file size
            chunk = chunk.set_channels(1)
            chunk.export(chunk_path, format="wav", parameters=["-ac", "1"])
            
            # Verify chunk size
            size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            print(f"Chunk {i+1}: {size_mb:.1f}MB, {len(chunk)/1000:.1f} seconds")
            chunks.append(chunk_path)
            
        return chunks
    
    def transcribe(self, audio_path: str, output_path: str = None) -> dict:
        """Transcribe audio file using OpenAI Whisper API"""
        try:
            start_time = time.time()
            chunks = self._split_audio(audio_path)
            all_text = []
            
            for i, chunk_path in enumerate(chunks, 1):
                print(f"\nProcessing chunk {i} of {len(chunks)}...")
                
                with open(chunk_path, 'rb') as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-1",
                        language="en"
                    )
                    all_text.append(transcript.text)
                
                os.remove(chunk_path)
                print(f"Processed and removed chunk {i}")
            
            # Combine results
            result = {
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "filename": Path(audio_path).name,
                    "chunks_processed": len(chunks),
                    "processing_time": f"{time.time() - start_time:.1f} seconds"
                },
                "text": " ".join(all_text)
            }
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                print(f"\nTranscript saved to {output_path}")
            
            return result
            
        except Exception as e:
            print(f"Error: {str(e)}")
            # Clean up any remaining chunks
            for file in Path().glob("chunk_*.wav"):
                file.unlink()
            raise
