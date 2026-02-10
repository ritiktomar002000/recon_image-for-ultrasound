import os
import numpy as np

from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.slider import Slider
from kivy.uix.scrollview import ScrollView

from recon_core import run_from_raw_files


# ---------- image -> texture with percentile window ----------
def to_tex(img, lo=1, hi=99):
    img = np.asarray(img, dtype=np.float32)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

    vmin, vmax = np.percentile(img, [lo, hi])
    if vmax <= vmin:
        vmax = vmin + 1e-6

    img = (img - vmin) / (vmax - vmin)
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    rgb = np.dstack([img, img, img])
    tex = Texture.create(size=(rgb.shape[1], rgb.shape[0]), colorfmt="rgb")
    tex.blit_buffer(rgb.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
    tex.flip_vertical()
    return tex


class ZoomPopup(Popup):
    def __init__(self, title_text, tex, **kwargs):
        super().__init__(title=title_text, size_hint=(0.95, 0.95), **kwargs)
        root = BoxLayout(orientation="vertical", spacing=8, padding=8)

        sv = ScrollView(do_scroll_x=True, do_scroll_y=True)
        img = Image(texture=tex, allow_stretch=True, keep_ratio=True, size_hint=(None, None))
        img.size = (tex.size[0] * 2, tex.size[1] * 2)
        sv.add_widget(img)

        root.add_widget(sv)
        close_btn = Button(text="Close", size_hint=(1, None), height=44)
        close_btn.bind(on_press=self.dismiss)
        root.add_widget(close_btn)
        self.content = root


class ImagePanel(BoxLayout):
    """
    Panel with:
      - Title
      - Depth slider (per panel)
      - Contrast slider (per panel)
      - Image view (double-tap to zoom)
    """
    def __init__(self, title, on_change_callback, **kwargs):
        super().__init__(orientation="vertical", spacing=6, padding=4, size_hint_y=None, **kwargs)
        self.bind(minimum_height=self.setter("height"))

        self.title = title
        self.on_change_callback = on_change_callback

        self.add_widget(Label(text=title, size_hint=(1, None), height=22))

        # Depth
        self.depth_lbl = Label(text="Depth: 100%", size_hint=(1, None), height=18)
        self.add_widget(self.depth_lbl)
        self.depth_slider = Slider(min=10, max=100, value=100, step=1, size_hint=(1, None), height=30)
        self.depth_slider.bind(value=self._changed)
        self.add_widget(self.depth_slider)

        # Contrast
        self.contrast_lbl = Label(text="Contrast: 98% (≈ prctile [1,99])", size_hint=(1, None), height=18)
        self.add_widget(self.contrast_lbl)
        self.contrast_slider = Slider(min=50, max=99, value=98, step=1, size_hint=(1, None), height=30)
        self.contrast_slider.bind(value=self._changed)
        self.add_widget(self.contrast_slider)

        # Image
        self.img = Image(size_hint=(1, None), height=380, allow_stretch=True, keep_ratio=True)
        self.img.bind(on_touch_down=self._maybe_zoom)
        self.add_widget(self.img)

        # spacer
        self.height = 22 + 18 + 30 + 18 + 30 + 380 + 10

    def _changed(self, *_):
        self.depth_lbl.text = f"Depth: {int(self.depth_slider.value)}%"
        self.contrast_lbl.text = f"Contrast: {int(self.contrast_slider.value)}% (≈ prctile)"
        self.on_change_callback()

    def _maybe_zoom(self, widget, touch):
        if widget.collide_point(*touch.pos) and widget.texture is not None and touch.is_double_tap:
            ZoomPopup(self.title, widget.texture).open()
            return True
        return False

    def get_params(self):
        depth_pct = int(self.depth_slider.value)
        contrast_pct = int(self.contrast_slider.value)
        low = (100 - contrast_pct) / 2
        high = 100 - low
        return depth_pct, low, high

    def set_texture(self, tex):
        self.img.texture = tex


class Root(BoxLayout):
    def __init__(self, **kw):
        super().__init__(orientation="vertical", spacing=8, padding=8, **kw)

        self.env_path = None
        self.rf_path = None

        # Cached images (full resolution)
        self.us_img = None
        self.rf_img = None
        self.rec_img = None

        self.status = Label(
            text="1) Select ENV file → Set as ENV | 2) Select RF file → Set as RF | 3) Run",
            size_hint=(1, None),
            height=28,
        )
        self.add_widget(self.status)

        start_path = os.path.join(os.path.expanduser("~"), "Desktop")

        # ---- chooser ----
        self.fc = FileChooserIconView(filters=["*.raw"], path=start_path, size_hint=(1, 0.28))
        self.add_widget(self.fc)

        # ---- file assign buttons ----
        pick_row = BoxLayout(size_hint=(1, None), height=44, spacing=8)
        self.pick_env_btn = Button(text="Set as ENV (_env.raw)", size_hint=(1, 1))
        self.pick_rf_btn = Button(text="Set as RF (_rf.raw)", size_hint=(1, 1))
        self.pick_env_btn.bind(on_press=self.set_env)
        self.pick_rf_btn.bind(on_press=self.set_rf)
        pick_row.add_widget(self.pick_env_btn)
        pick_row.add_widget(self.pick_rf_btn)
        self.add_widget(pick_row)

        self.file_lbl = Label(text="ENV: (not set) | RF: (not set)", size_hint=(1, None), height=24)
        self.add_widget(self.file_lbl)

        # ---- run controls ----
        controls = BoxLayout(size_hint=(1, None), height=44, spacing=8)
        controls.add_widget(Label(text="Navg:", size_hint=(None, 1), width=55))
        self.navg_in = TextInput(text="10", multiline=False, input_filter="int", size_hint=(None, 1), width=90)
        controls.add_widget(self.navg_in)

        self.run_btn = Button(text="Run Reconstruction", size_hint=(1, 1))
        self.run_btn.bind(on_press=self.on_run)
        controls.add_widget(self.run_btn)
        self.add_widget(controls)

        # ---- panels in scrollview ----
        sv = ScrollView(size_hint=(1, 1), do_scroll_y=True)
        self.panels = BoxLayout(orientation="vertical", spacing=12, padding=6, size_hint_y=None)
        self.panels.bind(minimum_height=self.panels.setter("height"))

        self.panel_us = ImagePanel("US ENV (averaged)", self.refresh_all)
        self.panel_rf = ImagePanel("PA RF (averaged)", self.refresh_all)
        self.panel_rec = ImagePanel("PA Reconstruction", self.refresh_all)

        self.panels.add_widget(self.panel_us)
        self.panels.add_widget(self.panel_rf)
        self.panels.add_widget(self.panel_rec)

        sv.add_widget(self.panels)
        self.add_widget(sv)

    def popup(self, title, text):
        box = BoxLayout(orientation="vertical", padding=10, spacing=10)
        box.add_widget(Label(text=text))
        btn = Button(text="OK", size_hint=(1, None), height=44)
        box.add_widget(btn)
        pop = Popup(title=title, content=box, size_hint=(0.9, 0.45))
        btn.bind(on_press=pop.dismiss)
        pop.open()

    def _selected_file(self):
        return self.fc.selection[0] if self.fc.selection else None

    def set_env(self, *_):
        f = self._selected_file()
        if not f:
            self.popup("No file", "Select a .raw file first.")
            return
        self.env_path = f
        self._update_file_label()

    def set_rf(self, *_):
        f = self._selected_file()
        if not f:
            self.popup("No file", "Select a .raw file first.")
            return
        self.rf_path = f
        self._update_file_label()

    def _update_file_label(self):
        self.file_lbl.text = (
            f"ENV: {os.path.basename(self.env_path) if self.env_path else '(not set)'} | "
            f"RF: {os.path.basename(self.rf_path) if self.rf_path else '(not set)'}"
        )

    def on_run(self, *_):
        if not self.env_path or not self.rf_path:
            self.popup("Missing files", "Please set BOTH ENV and RF files.")
            return

        try:
            navg = int(self.navg_in.text.strip())
        except Exception:
            navg = 1

        self.status.text = f"Running… (Navg={navg})"
        self.run_btn.disabled = True
        Clock.schedule_once(lambda dt: self._run_job(navg), 0.05)

    def _run_job(self, navg):
        try:
            # order: rf_path first, env_path second (matches your earlier call)
            us, rf_img, rec_img, *_ = run_from_raw_files(self.rf_path, self.env_path, navg)

            self.us_img = us
            self.rf_img = rf_img
            self.rec_img = rec_img

            self.status.text = "Done. Each image has its own depth/contrast sliders. Double-tap to zoom."
            self.refresh_all()

        except Exception as e:
            self.popup("Error", str(e))
            self.status.text = "Failed."
        finally:
            self.run_btn.disabled = False

    def refresh_all(self):
        # only refresh if we have data
        if self.us_img is None or self.rf_img is None or self.rec_img is None:
            return

        def crop(img, depth_pct):
            n = img.shape[0]
            keep = max(1, int(n * depth_pct / 100.0))
            return img[:keep, :]

        # US
        d_us, lo_us, hi_us = self.panel_us.get_params()
        us_disp = crop(self.us_img, d_us)
        self.panel_us.set_texture(to_tex(us_disp, lo=lo_us, hi=hi_us))

        # RF
        d_rf, lo_rf, hi_rf = self.panel_rf.get_params()
        rf_disp = crop(self.rf_img, d_rf)
        self.panel_rf.set_texture(to_tex(rf_disp, lo=lo_rf, hi=hi_rf))

        # Recon
        d_re, lo_re, hi_re = self.panel_rec.get_params()
        rec_disp = crop(self.rec_img, d_re)
        self.panel_rec.set_texture(to_tex(rec_disp, lo=lo_re, hi=hi_re))


class AppMain(App):
    def build(self):
        return Root()


if __name__ == "__main__":
    AppMain().run()
