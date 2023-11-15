import os
import threading
#os.environ["KIVY_NO_CONSOLELOG"] = "1"
from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from internal.frame_process import model_initialization, on_update
from internal.tts.tttest import generate_audios, play_audio
from kivy.core.text import Label
from kivy.graphics import Line, Rectangle, Color
from kivy.uix.widget import Widget
from kivy.clock import Clock
from vendor.kvconfigs import KV
import cv2
from notifypy import Notify

# Set the application icon

notification = Notify()

class AutopostureApp(MDApp):
    def build(self):
        generate_audios('bad', 'Sit correctly!')
        self.recently_alerted = False
        self.theme_cls.theme_style_switch_animation = True
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"
        self.icon = 'assets/ergonomic.png'

        self.prediction_scores = []  # List to store prediction scores
        return Builder.load_string(KV)

    def on_start(self):
        self.video_running = False
        self.capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
        model_initialization('0', 'src_models/yolov7-w6-pose.pt')

    def load_video_button(self):
        if not self.video_running:
            self.root.ids.start_autoposture.text = "PAUSE AUTOPOSTURE"
            self.video_running = True
            Clock.schedule_interval(self.load_video, 1.0 / 30.0)
        else:
            self.root.ids.start_autoposture.text = "RESTART AUTOPOSTURE"
            self.video_running = False
            Clock.unschedule(self.load_video)

    def switch_theme_style(self):
        self.theme_cls.primary_palette = (
            "Blue" if self.theme_cls.primary_palette == "Blue" else "Blue"
        )
        self.theme_cls.theme_style = (
            "Dark" if self.theme_cls.theme_style == "Light" else "Light"
        )

    def alert_user(self):
        notification.title = "Bad Posture"
        notification.message = "You are in a bad posture. Please correct your sitting."
        notification.icon = "assets/ergonomic.png"
        notification.send()
        play_audio('bad')

    def run_alert(self):
        if not self.recently_alerted:
            self.recently_alerted = False
            threading.Thread(target=self.alert_user).start()

    def trigger_alert(self, dt):
        self.alert_user()
        self.recently_alerted = False


    def load_video(self, *args):
        if not self.video_running:
            return

        ret, frame = self.capture.read()
        frame, posture, prediction_score, should_update = on_update(frame, self.recently_alerted, 0.7)
        if should_update:
            self.run_alert()
            self.recently_alerted = True
        else:
            self.recently_alerted = False

        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]),
            colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')

        video_image = self.root.ids.video_image
        video_image.texture = texture

        # Update posture label and set the text color
        posture_label = self.root.ids.posture_label
        posture_label.text = f"Posture: {posture}"
        posture_label.theme_text_color = "Error" if posture == "bad" else "Primary"

        prediction_label = self.root.ids.prediction_label
        prediction_label.text = f"Prediction Score: {int(prediction_score * 100)}%"
        
        self.prediction_scores.append(prediction_score)


    def show_average_popup(self):
        if self.prediction_scores:
            average = int(sum(self.prediction_scores) / len(self.prediction_scores) * 100)
            content = MDBoxLayout(orientation='vertical', padding=10)

            progress_label = MDLabel(
                text=f"Average Prediction Score: {average}%",
                theme_text_color="Secondary",
                font_style="H6",
            )
            content.add_widget(progress_label)

            popup = Popup(
                title="Average Score",
                content=content,
                size_hint=(None, None),
                size=("350dp", "200dp"),
            )

            # progress_label.theme_text_color = "Primary"
            if self.theme_cls.theme_style == "Light":
            # If the theme is "Light," set the background color and font color accordingly
                popup.background_color = self.theme_cls.bg_dark

            popup.open()

        # Clear the prediction scores list
            self.prediction_scores = []
        else:
        # Handle the case where there are no prediction scores.
            content = MDLabel(text="No prediction scores yet")
            popup = Popup(
                title="Average Score",
                content=content,
                size_hint=(None, None),
                size=("350dp", "200dp"),  # Adjust the size here to make it wider
            )

            if self.theme_cls.theme_style == "Light":
            # If the theme is "Light," set the background color and font color accordingly
                popup.background_color = self.theme_cls.bg_dark
                content.theme_text_color = "Primary"

            popup.open()

if __name__ == '__main__':
    AutopostureApp().run()
