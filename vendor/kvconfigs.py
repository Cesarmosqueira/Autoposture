
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
        pos_hint: {'center_x': 0.7}
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

            MDRaisedButton:
                id: another_button
                text: "Change Threshold"
                size_hint_x: None
                width: 200
                on_release: app.toggle_input_field()

            MDTextField:
                id: threshold_input
                hint_text: 'Threshold (0.7)'
                input_filter: 'float'
                size_hint_x: None
                width: 200
                on_text_validate: app.change_threshold(threshold_input.text)
'''
