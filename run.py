import os
from src.transcriber import AudioTranscriber
from config import OPENAI_API_KEY

def main():
    print("=== Starting Transcription Process ===")
    
    # Initialize the transcriber
    transcriber = AudioTranscriber(OPENAI_API_KEY)
    
    # Audio file path
    audio_file_path = r"D:\02-consolidated.wav"  # Update this path
    
    try:
        # Verify file exists
        if not os.path.exists(audio_file_path):
            print(f"Error: Audio file not found at {audio_file_path}")
            return
            
        # Transcribe the audio
        result = transcriber.transcribe(
            audio_path=audio_file_path,
            output_path="transcript.json"
        )
        
        # Print preview
        print("\nTranscription preview:")
        print(result["text"][:500] + "...")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 