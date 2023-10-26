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
from kivy.core.text import Label
from kivy.graphics import Line, Rectangle, Color
from kivy.uix.widget import Widget
from kivy.clock import Clock
import cv2

KV = '''
RelativeLayout:
    BoxLayout:
        orientation: 'vertical'
        size_hint: None, None
        size: self.minimum_size
        pos_hint: {'center_x': 0.5, 'center_y': 0.45}

        Image:
            id: video_image
            size_hint_x: None
            width: 640
            size_hint_y: None
            height: 480

        RelativeLayout:
            id: labels_layout
            size_hint_y: None
            height: "40dp"
            padding: 10

            MDLabel:
                id: posture_label
                text: "Posture:"
                font_size: '24sp'
                bold: True
                theme_text_color: "Secondary"
                pos_hint: {'x': 0.02}

            MDLabel:
                id: prediction_label
                text: "Prediction Score: 0"
                font_size: '24sp'
                bold: True
                pos_hint: {'x': 0.6}

    MDRaisedButton:
        icon: 'lightbulb-multiple-outline'
        on_release: app.switch_theme_style()
        pos_hint: {"x": -.03}

    AnchorLayout:
        anchor_x: 'center'
        anchor_y: 'top'
        pos_hint: {'center_x': 0.8}
        BoxLayout:
            orientation: 'horizontal'
            size_hint: 1, None
            height: "60dp"
            spacing: "10dp"
            pos_hint: {'center_x': 0.5}

            MDRaisedButton:
                id: start_autoposture
                text: "START AUTOPOSTURE"
                size_hint_x: None
                width: 200
                pos_hint: {'center_x': 0.5}
                on_release: app.load_video_button()

            MDRaisedButton:
                id: show_average_button
                text: "Show Average"
                size_hint_x: None
                width: 200
                on_release: app.show_average_popup()
                disabled: start_autoposture.state == 'down'
'''

class AutopostureApp(MDApp):
    def build(self):
        self.theme_cls.theme_style_switch_animation = True
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"
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

    def load_video(self, *args):
        if not self.video_running:
            return

        ret, frame = self.capture.read()
        frame, posture, prediction_score = on_update(frame)

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