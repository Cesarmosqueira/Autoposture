from gtts import gTTS
import subprocess

# Text you want to convert to speech
text_good = "Muy bien"
text_bad = "Posture!"

# Create a gTTS object
tts_good = gTTS(text_good)
tts_bad = gTTS(text_bad)

def generate_audios(status):
    audio_file = f"assets/audios/{status}.mp3"
    if status == 'good':
        tts_good.save(audio_file)
    if status == 'bad':
        tts_bad.save(audio_file)


def play_audio(status):
    audio_file = f"assets/audios/{status}.mp3"
    subprocess.run(["mpv", audio_file])

