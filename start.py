from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDRaisedButton
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from internal.frame_process import model_initialization, on_update
import cv2

class AutopostureApp(MDApp):
    def build(self):
        layout = MDBoxLayout(orientation='vertical')
        self.image = Image()
        layout.add_widget(self.image)
        layout.add_widget(MDRaisedButton(
            text="THIS IS A BUTTON",
            pos_hint={'center_x': .5, 'center_y': .5},
            size_hint=(None, None))
        )
        # Specify the backend as V4L2 by setting the API preference
        model_initialization('0', 'src_models/yolov7-w6-pose.pt')
        self.capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
        Clock.schedule_interval(self.load_video, 1.0 / 30.0)


        return layout

    def load_video(self, *args):
        ret, frame = self.capture.read()
        
        frame = on_update(frame)

        buffer = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]),
            colorfmt='bgr')
        texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
        self.image.texture = texture

if __name__ == '__main__':
    AutopostureApp().run()

