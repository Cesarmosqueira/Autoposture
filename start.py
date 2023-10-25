from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from internal.frame_process import model_initialization, on_update
import cv2

class AutopostureApp(MDApp):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        
        # Create a button and bind the load_video function to its on_release event
        self.video_running = False
        self.button = MDRaisedButton(
            text="START AUTOPOSTURE",
            size_hint=(1, None))
        self.button.bind(on_release=self.load_video_button)
        
        # Create a BoxLayout for the video display
        self.video_layout = BoxLayout(orientation='horizontal')
        self.image = Image()
        self.video_layout.add_widget(self.image)
        
        # Create a BoxLayout for posture and prediction labels
        self.label_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
        self.posture_label = MDLabel(text="Posture:", font_size='24sp', bold=True, theme_text_color="Secondary")
        self.prediction_label = MDLabel(text="Prediction Score: 0", font_size='24sp', bold=True)
        self.label_layout.add_widget(self.posture_label)
        self.label_layout.add_widget(self.prediction_label)
        
        layout.add_widget(self.video_layout)
        layout.add_widget(self.button)
        layout.add_widget(self.label_layout)
        
        # Specify the backend as V4L2 by setting the API preference
        model_initialization('0', 'src_models/yolov7-w6-pose.pt')
        self.capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        return layout

    def load_video_button(self, instance):
        if not self.video_running:
            self.button.text = "PAUSE AUTOPOSTURE"
            self.video_running = True
            Clock.schedule_interval(self.load_video, 1.0 / 30.0)
        else:
            self.button.text = "RESTART AUTOPOSTURE"
            self.video_running = False
            Clock.unschedule(self.load_video)

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
        self.image.texture = texture
        
        # Update posture label and set the text color
        self.posture_label.text = f"Posture: {posture}"
        if posture == "bad":
            self.posture_label.theme_text_color = "Error"
        else:
            self.posture_label.theme_text_color = "Primary"
        self.prediction_label.text = f"Prediction Score: {int(prediction_score * 100)}%"

if __name__ == '__main__':
    AutopostureApp().run()