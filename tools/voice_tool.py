from main import process_voice_input

class VoiceTool:
    def process_voice_input(self, file_path: str, language: str = 'en-US'):
        return process_voice_input(file_path, language)
    
    def convert_to_speech(self, text: str, voice: str = 'default'):
        # Implement text-to-speech conversion
        return {"status": "success", "message": "Text converted to speech"}

voice_tool = VoiceTool() 