from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from internal.frame_process import model_initialization, on_update
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
            width: 640  # Set the width to match the video frame width
            size_hint_y: None
            height: 480  # Adjust the height according to your video feed

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
        pos_hint: {'center_x': 0.5, 'center_y': 0.47}
        MDRaisedButton:
            id: start_autoposture
            text: "START AUTOPOSTURE"
            size_hint_x: None
            width: 800 
            size_hint_y: None
            height: 60  
            on_release: app.load_video_button()
'''

class AutopostureApp(MDApp):
    def build(self):
        self.theme_cls.theme_style_switch_animation = True
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "Blue"
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

if __name__ == '__main__':
    AutopostureApp().run()