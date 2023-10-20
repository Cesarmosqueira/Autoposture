from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

from kivy.garden.modernmenu import ModernMenu
from kivy.garden.mapview import MapView

class GardenVisualizationApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        menu = ModernMenu()
        map_view = MapView()
        layout.add_widget(menu)
        layout.add_widget(map_view)
        return layout

if __name__ == '__main__':
    GardenVisualizationApp().run()

