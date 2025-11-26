import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
import time
import logging
import pickle
import os
from gtts import gTTS
from PIL import Image, ImageTk, ImageOps, ImageDraw
import customtkinter as ctk

# --- Configuration & Aesthetics ---
APP_NAME = "SignAssist"
# Color Palette (Minimalist Dark Theme)
COLOR_BG = "#121212"       # Very dark grey/black background
COLOR_SIDEBAR = "#1E1E1E"  # Slightly lighter for sidebar
COLOR_ACCENT = "#3B8ED0"   # Primary Blue
COLOR_ACCENT_HOVER = "#36719F"
COLOR_TEXT = "#FFFFFF"
COLOR_TEXT_SEC = "#A0A0A0" # Secondary text
COLOR_CARD = "#252525"     # Card background for video/text areas

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False, 
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# --- Model Loading ---
model = None
labels_dict = {0: '0', 1: '1', 2: '2', 3: 'L', 4: 'ok', 5: 'not ok', 6: 'Peace'} 

if os.path.exists('model.p'):
    try:
        model_dict = pickle.load(open('model.p', 'rb'))
        model = model_dict['model']
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
else:
    logging.warning("'model.p' not found. Running in demo mode.")

class SignAssistApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Setup
        self.geometry("1300x850")
        self.title(APP_NAME)
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("blue")
        
        # Configure Grid Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # State Variables
        self.current_language = 'en'
        self.gesture_history = []
        self.last_detected_char = None
        self.last_time = time.time()
        self.delay_buffer = 0
        self.thumbs_up_popup = None
        self.thumbs_down_popup = None
        self.history_visible = False

        # Custom Fonts
        self.font_header = ("Segoe UI Display", 28, "bold")
        self.font_subheader = ("Segoe UI", 18)
        self.font_body = ("Segoe UI", 14)
        self.font_btn = ("Segoe UI", 13, "bold")
        self.font_mono = ("Consolas", 12) # For techy feel

        # --- UI Construction ---
        self.setup_sidebar()
        self.setup_main_area()
        
        # --- Camera Setup ---
        self.cap = cv2.VideoCapture(0)
        self.update_camera()
        
        self.after(1000, self.show_tutorial)

    def setup_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0, fg_color=COLOR_SIDEBAR)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(8, weight=1) # Spacer

        # Logo Area
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, 
            text=APP_NAME, 
            font=self.font_header,
            text_color=COLOR_ACCENT
        )
        self.logo_label.grid(row=0, column=0, padx=30, pady=(40, 20), sticky="w")

        # Theme Switch
        self.theme_switch = ctk.CTkSwitch(
            self.sidebar_frame, 
            text="Dark Mode", 
            command=self.toggle_theme, 
            onvalue="Dark", 
            offvalue="Light",
            font=self.font_body,
            progress_color=COLOR_ACCENT
        )
        self.theme_switch.select() 
        self.theme_switch.grid(row=1, column=0, padx=30, pady=15, sticky="w")

        # Divider
        self.create_divider(self.sidebar_frame, row=2)

        # Language Section
        self.lbl_lang = ctk.CTkLabel(self.sidebar_frame, text="TRANSLATION TARGET", font=("Segoe UI", 11, "bold"), text_color=COLOR_TEXT_SEC)
        self.lbl_lang.grid(row=3, column=0, padx=30, pady=(20, 5), sticky="w")
        
        self.lang_var = ctk.StringVar(value="English")
        self.lang_menu = ctk.CTkOptionMenu(
            self.sidebar_frame, 
            variable=self.lang_var, 
            values=["English", "Hindi", "Spanish", "French"], 
            command=self.set_language,
            fg_color=COLOR_CARD,
            button_color=COLOR_ACCENT,
            button_hover_color=COLOR_ACCENT_HOVER,
            text_color=COLOR_TEXT,
            font=self.font_body,
            corner_radius=8
        )
        self.lang_menu.grid(row=4, column=0, padx=30, pady=10, sticky="ew")

        # Tools Section
        self.lbl_tools = ctk.CTkLabel(self.sidebar_frame, text="TOOLS", font=("Segoe UI", 11, "bold"), text_color=COLOR_TEXT_SEC)
        self.lbl_tools.grid(row=5, column=0, padx=30, pady=(30, 5), sticky="w")

        self.btn_history = self.create_sidebar_btn("Show History", self.toggle_history, row=6)
        self.btn_feedback = self.create_sidebar_btn("Send Feedback", self.open_feedback_window, row=7)

        # Footer / Exit
        self.btn_exit = ctk.CTkButton(
            self.sidebar_frame, 
            text="Exit Application", 
            command=self.exit_app, 
            fg_color="transparent", 
            border_color="#E74C3C",
            border_width=1,
            hover_color="#331818", 
            text_color="#E74C3C",
            font=self.font_btn,
            height=40,
            corner_radius=8
        )
        self.btn_exit.grid(row=9, column=0, padx=30, pady=30, sticky="ew")

    def setup_main_area(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=COLOR_BG)
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)

        # 1. Video Container (The "Stage")
        self.video_wrapper = ctk.CTkFrame(self.main_frame, corner_radius=20, fg_color=COLOR_CARD, border_width=0)
        self.video_wrapper.grid(row=0, column=0, sticky="nsew", padx=30, pady=(30, 15))
        self.video_wrapper.grid_columnconfigure(0, weight=1)
        self.video_wrapper.grid_rowconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(self.video_wrapper, text="", corner_radius=20)
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Floating Status Pill
        self.status_pill = ctk.CTkFrame(self.video_wrapper, fg_color="#000000", corner_radius=20, height=35, width=100, bg_color="transparent")
        self.status_pill.place(x=20, y=20)
        
        self.live_dot = ctk.CTkLabel(self.status_pill, text="●", text_color="#e74c3c", font=("Arial", 16))
        self.live_dot.place(x=10, y=2)
        self.live_text = ctk.CTkLabel(self.status_pill, text="LIVE", text_color="white", font=("Segoe UI", 11, "bold"))
        self.live_text.place(x=30, y=5)

        # 2. Controls & Output Area
        self.bottom_section = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.bottom_section.grid(row=1, column=0, sticky="ew", padx=30, pady=(0, 30))
        self.bottom_section.grid_columnconfigure(0, weight=1) # Output
        self.bottom_section.grid_columnconfigure(1, weight=0) # Controls

        # Left: Text Output Box
        self.output_frame = ctk.CTkFrame(self.bottom_section, corner_radius=15, fg_color=COLOR_CARD, height=120)
        self.output_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        
        self.lbl_transcript = ctk.CTkLabel(self.output_frame, text="TRANSCRIPT", font=("Segoe UI", 10, "bold"), text_color=COLOR_TEXT_SEC)
        self.lbl_transcript.pack(anchor="w", padx=20, pady=(15, 0))

        self.text_field = ctk.CTkTextbox(
            self.output_frame, 
            height=70, 
            font=("Segoe UI", 20), 
            fg_color="transparent", 
            text_color=COLOR_TEXT,
            wrap="word"
        )
        self.text_field.pack(padx=15, pady=(5, 15), fill="both", expand=True)

        # Right: Control Buttons
        self.controls_frame = ctk.CTkFrame(self.bottom_section, corner_radius=15, fg_color=COLOR_CARD, height=120)
        self.controls_frame.grid(row=0, column=1, sticky="nsew")

        self.create_control_btn("undo", "Undo", self.undo_text, 0)
        self.create_control_btn("trash", "Clear", self.clear_text, 1)
        self.create_control_btn("volume-2", "Speak", self.speak_text, 2, is_primary=True)

        # Feedback Toast (Floating)
        self.feedback_label = ctk.CTkLabel(
            self.main_frame, 
            text="Ready", 
            font=("Segoe UI", 12), 
            text_color=COLOR_TEXT_SEC, 
            fg_color=COLOR_CARD,
            corner_radius=10,
            height=30,
            width=200
        )
        self.feedback_label.place(relx=0.5, rely=0.05, anchor="n")

    # --- Helper UI Functions ---
    def create_divider(self, parent, row):
        line = ctk.CTkFrame(parent, height=1, fg_color="#333333")
        line.grid(row=row, column=0, sticky="ew", padx=30, pady=10)

    def create_sidebar_btn(self, text, command, row):
        btn = ctk.CTkButton(
            self.sidebar_frame, 
            text=text, 
            command=command, 
            font=self.font_body, 
            fg_color="transparent", 
            hover_color="#2B2B2B", 
            anchor="w",
            corner_radius=8,
            height=40,
            text_color=COLOR_TEXT
        )
        btn.grid(row=row, column=0, padx=20, pady=5, sticky="ew")
        return btn

    def create_control_btn(self, icon_name, text, command, col, is_primary=False):
        # Note: In a real deployment, use CTkImage for icons. Here we use text for simplicity.
        color = COLOR_ACCENT if is_primary else "#333333"
        hover = COLOR_ACCENT_HOVER if is_primary else "#444444"
        
        btn = ctk.CTkButton(
            self.controls_frame,
            text=text,
            command=command,
            width=90,
            height=90, # Square buttons
            corner_radius=12,
            fg_color=color,
            hover_color=hover,
            font=self.font_btn
        )
        btn.pack(side="left", padx=15, pady=15)

    # --- Logic Functions ---

    def update_feedback(self, message):
        self.feedback_label.configure(text=message)
        logging.info(f"Feedback: {message}")

    def update_camera(self):
        ret, frame = self.cap.read()
        if ret:
            # 1. Basic Image Prep
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            
            predicted_character = None

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Minimalist Drawing: Thinner lines, specific colors
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(59, 142, 208), thickness=2, circle_radius=2), # Blue dots
                        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)  # White lines
                    )

                    # Model Prediction logic remains same
                    if model:
                        try:
                            data_aux = []
                            x_ = [lm.x for lm in hand_landmarks.landmark]
                            y_ = [lm.y for lm in hand_landmarks.landmark]

                            for lm in hand_landmarks.landmark:
                                data_aux.append(lm.x - min(x_))
                                data_aux.append(lm.y - min(y_))

                            prediction = model.predict([np.asarray(data_aux)])
                            predicted_character = labels_dict[int(prediction[0])]
                        except Exception:
                            pass

            # Update Logic
            if predicted_character:
                self.update_feedback(f"Detected: {predicted_character}")
                self.handle_special_gestures(predicted_character)
                
                # Debounce Logic
                current_time = time.time()
                if predicted_character == self.last_detected_char:
                    if (current_time - self.last_time) >= 1.0:
                        if self.delay_buffer == 0:
                            self.add_text_to_field(predicted_character)
                            self.delay_buffer = 1
                else:
                    self.last_detected_char = predicted_character
                    self.last_time = current_time
                    self.delay_buffer = 0

            # 2. Image Processing for UI (Corner Radius & Resize)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Smart Resize logic
            container_w = self.video_wrapper.winfo_width()
            container_h = self.video_wrapper.winfo_height()
            
            if container_w > 10 and container_h > 10:
                # Cover method (like CSS object-fit: cover)
                img_ratio = img.width / img.height
                container_ratio = container_w / container_h
                
                if container_ratio > img_ratio:
                    resize_w = container_w
                    resize_h = int(container_w / img_ratio)
                else:
                    resize_w = int(container_h * img_ratio)
                    resize_h = container_h
                
                img = img.resize((resize_w, resize_h), Image.Resampling.LANCZOS)
                
                # Center crop
                left = (resize_w - container_w) / 2
                top = (resize_h - container_h) / 2
                right = (resize_w + container_w) / 2
                bottom = (resize_h + container_h) / 2
                img = img.crop((left, top, right, bottom))
            else:
                img = img.resize((640, 480)) # Fallback

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.after(10, self.update_camera)

    def handle_special_gestures(self, char):
        if char == 'ok' and self.thumbs_up_popup is None:
            self.show_popup("👍 Great Job!", "thumbs_up")
        elif char == 'not ok' and self.thumbs_down_popup is None:
            self.show_popup("👎 Try Again!", "thumbs_down")
            
        if char != 'ok' and self.thumbs_up_popup:
            self.thumbs_up_popup.destroy()
            self.thumbs_up_popup = None
        if char != 'not ok' and self.thumbs_down_popup:
            self.thumbs_down_popup.destroy()
            self.thumbs_down_popup = None

    def show_popup(self, message, type_key):
        popup = ctk.CTkToplevel(self)
        popup.geometry("300x100")
        popup.overrideredirect(True) # Remove window borders for cleaner look
        
        # Position popup relative to main window
        x = self.winfo_x() + (self.winfo_width() // 2) - 150
        y = self.winfo_y() + (self.winfo_height() // 2) - 50
        popup.geometry(f"+{x}+{y}")
        
        frame = ctk.CTkFrame(popup, fg_color=COLOR_ACCENT, corner_radius=15)
        frame.pack(fill="both", expand=True)
        
        lbl = ctk.CTkLabel(frame, text=message, font=("Segoe UI", 18, "bold"), text_color="white")
        lbl.pack(expand=True)
        
        if type_key == "thumbs_up":
            self.thumbs_up_popup = popup
        else:
            self.thumbs_down_popup = popup

    def add_text_to_field(self, text):
        self.text_field.insert('end', text)
        self.gesture_history.append(text)
        if hasattr(self, 'history_text_box'):
            self.update_history_display()
        self.update_feedback(f"Added '{text}'")

    # --- Action Functions ---
    def clear_text(self):
        self.text_field.delete('1.0', 'end')
        self.update_feedback("Transcript cleared.")

    def undo_text(self):
        content = self.text_field.get('1.0', 'end').strip()
        if content:
            new_content = content[:-1]
            self.text_field.delete('1.0', 'end')
            self.text_field.insert('1.0', new_content)
            if self.gesture_history:
                self.gesture_history.pop()
                if hasattr(self, 'history_text_box'):
                    self.update_history_display()

    def speak_text(self):
        text = self.text_field.get('1.0', 'end').strip()
        if text:
            try:
                tts = gTTS(text=text, lang='en') 
                tts.save("output.mp3")
                if os.name == "nt":
                    os.system("start output.mp3")
                else:
                    os.system("xdg-open output.mp3")
            except Exception as e:
                logging.error(f"TTS Error: {e}")

    def toggle_history(self):
        if self.history_visible:
            if hasattr(self, 'history_panel'):
                self.history_panel.destroy()
            self.btn_history.configure(text="Show History", fg_color="transparent")
            self.history_visible = False
        else:
            self.history_panel = ctk.CTkFrame(self.main_frame, width=250, corner_radius=20, fg_color=COLOR_CARD)
            self.history_panel.place(relx=0.97, rely=0.03, anchor="ne", relheight=0.5)
            
            header = ctk.CTkLabel(self.history_panel, text="HISTORY", font=("Segoe UI", 11, "bold"), text_color=COLOR_TEXT_SEC)
            header.pack(pady=(20, 10), padx=20, anchor="w")
            
            self.history_text_box = ctk.CTkTextbox(self.history_panel, width=200, fg_color="transparent", font=("Segoe UI", 14))
            self.history_text_box.pack(pady=5, padx=20, fill="both", expand=True)
            self.update_history_display()
            
            btn = ctk.CTkButton(self.history_panel, text="Clear All", command=self.clear_history, width=100, height=30, fg_color="#333", hover_color="#444")
            btn.pack(pady=20)

            self.btn_history.configure(text="Hide History", fg_color=COLOR_ACCENT)
            self.history_visible = True

    def update_history_display(self):
        self.history_text_box.delete('1.0', 'end')
        self.history_text_box.insert('end', '\n'.join(self.gesture_history)) # Vertical list looks better

    def clear_history(self):
        self.gesture_history = []
        self.update_history_display()

    def toggle_theme(self):
        if self.theme_switch.get() == "Dark":
            ctk.set_appearance_mode("Dark")
        else:
            ctk.set_appearance_mode("Light")

    def set_language(self, choice):
        self.current_language = choice
        self.update_feedback(f"Target Language: {choice}")

    def open_feedback_window(self):
        fb_win = ctk.CTkToplevel(self)
        fb_win.title("Feedback")
        fb_win.geometry("400x350")
        fb_win.configure(fg_color=COLOR_BG)
        
        ctk.CTkLabel(fb_win, text="Your thoughts matter.", font=("Segoe UI", 20, "bold")).pack(pady=(30, 10))
        txt = ctk.CTkTextbox(fb_win, height=150, width=300, corner_radius=10, fg_color=COLOR_CARD)
        txt.pack(pady=10)
        
        def save():
            with open("feedback.txt", "a") as f:
                f.write(f"[{time.ctime()}] {txt.get('1.0', 'end').strip()}\n")
            fb_win.destroy()
            self.update_feedback("Feedback sent. Thank you!")
            
        ctk.CTkButton(fb_win, text="Submit Feedback", command=save, fg_color=COLOR_ACCENT, width=300, height=40).pack(pady=10)

    def show_tutorial(self):
        self.update_feedback("Welcome! Show hand gestures to begin.")

    def exit_app(self):
        self.cap.release()
        self.quit()

if __name__ == "__main__":
    app = SignAssistApp()
    app.mainloop()