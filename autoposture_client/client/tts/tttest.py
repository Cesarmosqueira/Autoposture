from gtts import gTTS
import subprocess


def generate_audios(status, text):
    try:
        audio_file = f"assets/audios/{status}.mp3"
        gTTS(text).save(audio_file)
        return True
    except:
        return False


def play_audio(status):
    audio_file = f"assets/audios/{status}.mp3"
    print(audio_file)
    subprocess.run(["mpv", audio_file])
