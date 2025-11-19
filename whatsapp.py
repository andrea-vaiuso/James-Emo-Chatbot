import tkinter as tk
from tkinter import Canvas, Frame, Label, Entry, Button, simpledialog
import threading, random
from datetime import datetime
import torch
from PIL import Image, ImageTk, ImageDraw
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from utilities import play_system_sound, SOUNDS
from LLM.llm import LLMModel

# =========================
# THEME / BEHAVIOR SETTINGS
# =========================
THEME = {
    "bg": "#111B21",
    "bubble_user_bg": "#005C4B",
    "bubble_bot_bg": "#1F2C34",
    "bubble_sys_bg": "#182229",
    "header_height": 64,
    "user_name_color": "#9EE9BB",
    "assistant_name_color": "#8FD1FF",
    "system_name_color": "#F4EBD0",
    "user_text_color": "#E9F5EB",
    "assistant_text_color": "#DDE6EA",
    "system_text_color": "#F8F4E7",
    "typing_color": "#8696A0",
    "timestamp_color": "#8696A0",
    "wrap": 520,      # px wrap-length for message text
    "radius": 22,     # bubble corner radius
    "pad_in": (10, 4), # inner padding (x, y) - slimmer bubbles
    "input_bar_bg": "#1F2C34",
    "input_bg": "#0B141A",
    "input_fg": "#E9F5EB",
    "input_border": "#0B845E",
    "button_bg": "#25D366",
    "button_fg": "#052D1A",
    "button_active_bg": "#1DA851",
    "button_active_fg": "#FFFFFF",
    "button_icon": "\u27A4"
}
FONTS = {
    "name": ("Segoe UI", 12, "bold"),
    "header_name": ("Segoe UI", 14, "bold"),
    "text": ("Segoe UI", 12, "normal"),
    "input": ("Segoe UI", 12, "normal"),
    "button": ("Segoe UI", 12, "bold"),
    "typing": ("Segoe UI", 10, "italic"),
    "timestamp": ("Segoe UI", 8, "normal")
}
TIME_FMT = "%H:%M"  # NEW

TYPING_DELAY_RANGE = (1500, 2500)  # ms, randomized, does not block LLM

# ===========================================
#  BUBBLE CHAT UI (TKINTER) + ROUNDED BUBBLES
# ===========================================
class BubbleChatApp:
    def __init__(self, root: tk.Tk, user_name: str, ai_username: str, llm: LLMModel, max_history_length=8000):
        self.llm_obj = llm
        self.root = root
        self.root.title("chat")
        self.root.configure(bg=THEME["bg"])
        self.root.resizable(True, True)

        self.max_history_length = max_history_length

        self._build_header(THEME["header_height"])

        # Container for canvas + scrollbar
        self.top_container = Frame(self.root, bg=THEME["bg"])
        self.top_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Canvas + inner frame
        self.canvas = tk.Canvas(self.top_container, bg=THEME["bg"], highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.top_container, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.chat_frame = Frame(self.canvas, bg=THEME["bg"])
        self.window_id = self.canvas.create_window((0, 0), window=self.chat_frame, anchor="nw")
        self._bubble_registry = {}
        self._current_wrap = THEME["wrap"]

        # Bindings to keep width and scroll working correctly
        self.chat_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel(self.canvas)

        # Bottom input area
        self.bottom_frame = Frame(
            self.root,
            bg=THEME["input_bar_bg"],
            highlightbackground=THEME["input_border"],
            highlightthickness=1
        )
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.user_input = Entry(
            self.bottom_frame,
            font=FONTS["input"],
            bg=THEME["input_bg"],
            fg=THEME["input_fg"],
            insertbackground=THEME["input_fg"],
            relief="flat",
            bd=0,
            highlightbackground=THEME["input_border"],
            highlightcolor=THEME["input_border"],
            highlightthickness=1
        )
        self.user_input.pack(side=tk.LEFT, padx=(6, 6), pady=6, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.on_send)

        self.send_button = Button(
            self.bottom_frame,
            text=f"{THEME['button_icon']} Send",
            font=FONTS["button"],
            command=self.on_send,
            bg=THEME["button_bg"],
            fg=THEME["button_fg"],
            activebackground=THEME["button_active_bg"],
            activeforeground=THEME["button_active_fg"],
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=16,
            pady=8
        )
        self.send_button.pack(side=tk.RIGHT, padx=6, pady=6)

        # State
        self.user_name = user_name
        self.ai_username = ai_username
        self.conversation_history = self.llm_obj.init_conversation(self.user_name)
        self.typing_after_id = None
        self.typing_visible = False
        self._typing_frame = None

        # Initial system greeting
        self.add_message_bubble("System", f"Hello {self.user_name}, welcome to the chat with {self.ai_username}!\nType 'exit' or 'quit' to leave.", role="system")

    # ---- scrolling fixes ----
    def _on_frame_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.canvas.yview_moveto(1.0)  # keep autoscroll

    def _on_canvas_configure(self, event):
        # Match inner frame width to canvas width
        self.canvas.itemconfig(self.window_id, width=event.width)
        self._update_bubble_sizes(event.width)

    def _bind_mousewheel(self, widget):
        # Windows and Linux
        widget.bind_all("<MouseWheel>", self._on_mousewheel)
        # macOS
        widget.bind_all("<Button-4>", self._on_mousewheel)
        widget.bind_all("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        if event.num == 4:  # macOS up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:  # macOS down
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _compute_wrap_length(self, container_width=None):
        width = container_width if container_width and container_width > 1 else self.canvas.winfo_width()
        if width <= 1:
            width = self.canvas.winfo_reqwidth()
        if width and width > 1:
            usable = max(240, width - 120)
            return int(usable)
        return THEME["wrap"]

    def _rounded_rect_points(self, width, height, radius):
        r = min(radius, width / 2, height / 2)
        x0, y0 = 0, 0
        x1, y1 = width, height
        return [
            x0 + r, y0,
            x1 - r, y0,
            x1, y0,
            x1, y0 + r,
            x1, y1 - r,
            x1, y1,
            x1 - r, y1,
            x0 + r, y1,
            x0, y1,
            x0, y1 - r,
            x0, y0 + r,
            x0, y0,
        ]

    def _layout_single_bubble(self, meta, wraplength):
        meta["text_label"].configure(wraplength=wraplength)
        inner = meta["inner"]
        inner.update_idletasks()
        width = inner.winfo_reqwidth() + meta["pad_x"] * 2
        height = inner.winfo_reqheight() + meta["pad_y"] * 2
        points = self._rounded_rect_points(width, height, THEME["radius"])
        canvas = meta["canvas"]
        canvas.coords(meta["polygon"], *points)
        canvas.configure(width=width, height=height)

    def _update_bubble_sizes(self, container_width=None, force=False):
        wraplength = self._compute_wrap_length(container_width)
        if not force and wraplength == self._current_wrap:
            return
        self._current_wrap = wraplength
        stale = []
        for outer, meta in list(self._bubble_registry.items()):
            if not meta["inner"].winfo_exists():
                stale.append(outer)
                continue
            self._layout_single_bubble(meta, wraplength)
        for outer in stale:
            self._bubble_registry.pop(outer, None)

    def _unregister_bubble(self, outer):
        self._bubble_registry.pop(outer, None)

    # ---- message send flow ----
    def on_send(self, event=None):
        user_text = self.user_input.get().strip()
        if not user_text:
            return

        self.user_input.delete(0, tk.END)
        self.add_message_bubble(self.user_name, user_text, role="user")
        play_system_sound(SOUNDS["send"])

        if user_text.lower() in ["exit", "quit"]:
            self.add_message_bubble("System", "Exiting the chat. Goodbye!", role="system")
            self.root.after(800, self.root.destroy)
            return

        # Update conversation and start reply thread
        self.conversation_history += f"<|start_header_id|>{self.user_name}<|end_header_id|>\n{user_text} <|eot_id|>\n"
        self.llm_obj.memory_chunks.append(f"{self.user_name}: {user_text}\n")
        threading.Thread(target=self._llm_reply_thread, daemon=True).start()

    def _llm_reply_thread(self):
        # Schedule showing a typing bubble after a small random delay.
        delay_ms = random.randint(*TYPING_DELAY_RANGE)
        self.typing_after_id = self.root.after(delay_ms, lambda: self._show_typing_bubble(f"{self.llm_obj.ai_username} is writing..."))

        # Run generation in this thread
        response = self.llm_obj.generate_response(self.conversation_history)

        # Cancel pending typing if not yet shown, remove if shown
        def finalize_ui():
            if self.typing_after_id is not None:
                # If typing bubble was not shown yet, this cancels its appearance
                try:
                    self.root.after_cancel(self.typing_after_id)
                except Exception:
                    pass
                self.typing_after_id = None
            if self.typing_visible:
                self._remove_typing_bubble()

            # Append to history and render
            self.conversation_history += f"<|start_header_id|>{self.llm_obj.ai_username}<|end_header_id|>\n{response}\n<|eot_id|>\n"
            self.llm_obj.memory_chunks.append(f"{self.llm_obj.ai_username}: {response}\n")
            self.conversation_history = self.conversation_history[-self.max_history_length:]
            self.add_message_bubble(self.llm_obj.ai_username, response, role="bot")
            play_system_sound(SOUNDS["recv"])

        # Marshal UI ops back to main thread
        self.root.after(0, finalize_ui)

    # ---- typing bubble helpers ----
    def _show_typing_bubble(self, text):
        if self.typing_visible:
            return
        self.typing_visible = True
        self._typing_frame = self._create_bubble(
            name=self.llm_obj.ai_username,
            text=text,
            role="typing"
        )

    def _remove_typing_bubble(self):
        if self._typing_frame:
            self._typing_frame.destroy()
            self._unregister_bubble(self._typing_frame)
            self._typing_frame = None
        self.typing_visible = False

    # ---- drawing rounded bubbles on a Canvas ----
    def _create_bubble(self, name, text, role="bot", ts=None):
        """
        Creates a rounded bubble composed of a Canvas with a rounded rect and two labels:
        first line bold name, second line the message text.
        """
        outer = Frame(self.chat_frame, bg=THEME["bg"])
        outer.pack(fill=tk.X, pady=3, padx=10)

        # alignment
        align = "e" if role == "user" else "w"

        # colors
        if role == "user":
            bb = THEME["bubble_user_bg"]
            name_color = THEME["user_name_color"]
            msg_color = THEME["user_text_color"]
        elif role == "system":
            bb = THEME["bubble_sys_bg"]
            name_color = THEME["system_name_color"]
            msg_color = THEME["system_text_color"]
        elif role == "typing":
            bb = THEME["bubble_bot_bg"]
            name_color = THEME["assistant_name_color"]
            msg_color = THEME["assistant_text_color"]
        else:
            bb = THEME["bubble_bot_bg"]
            name_color = THEME["assistant_name_color"]
            msg_color = THEME["assistant_text_color"]

        # Canvas-based bubble
        c = Canvas(outer, bg=THEME["bg"], highlightthickness=0)
        c.pack(anchor=align, padx=0)

        # A frame inside canvas to hold labels
        inner = Frame(c, bg=bb)
        # Labels: name on first line (bold), text on second line
        name_lbl = Label(inner, text=name, fg=name_color, bg=bb, font=FONTS["name"], anchor="w", justify="left")
        name_lbl.pack(anchor="w", padx=(THEME["pad_in"][0], THEME["pad_in"][0]), pady=(THEME["pad_in"][1], 0))

        msg_font = FONTS["typing"] if role == "typing" else FONTS["text"]
        if role == "typing":
            msg_color = THEME["typing_color"]
        text_lbl = Label(
            inner,
            text=text,
            fg=msg_color,
            bg=bb,
            font=msg_font,
            wraplength=self._current_wrap,
            justify="left",
            anchor="w"
        )
        text_lbl.pack(anchor="w", padx=(THEME["pad_in"][0], THEME["pad_in"][0]), pady=(1, THEME["pad_in"][1]))

        if ts is None:
            ts = datetime.now().strftime(TIME_FMT)
        meta_row = Frame(inner, bg=bb)
        meta_row.pack(anchor="e",
                    fill="x",
                    padx=(THEME["pad_in"][0], THEME["pad_in"][0]),
                    pady=(0, THEME["pad_in"][1]))
        ts_lbl = Label(meta_row, text=ts, fg=THEME["timestamp_color"],
                    bg=bb, font=FONTS["timestamp"])
        ts_lbl.pack(side="right")

        # Materialize sizes to draw rounded rect behind
        inner.update_idletasks()
        pad_x = 3
        pad_y = 1
        width = inner.winfo_reqwidth() + pad_x * 2
        height = inner.winfo_reqheight() + pad_y * 2
        points = self._rounded_rect_points(width, height, THEME["radius"])
        polygon_id = c.create_polygon(points, smooth=True, splinesteps=20, fill=bb, outline=bb)

        # Place inner frame inside canvas as a window
        c.create_window(pad_x, pad_y, anchor="nw", window=inner)

        # Resize canvas to fit content
        c.configure(width=width, height=height)

        self._bubble_registry[outer] = {
            "canvas": c,
            "inner": inner,
            "text_label": text_lbl,
            "polygon": polygon_id,
            "pad_x": pad_x,
            "pad_y": pad_y
        }
        self._update_bubble_sizes(force=True)

        # autoscroll
        self.root.update_idletasks()
        self.canvas.yview_moveto(1.0)

        return outer

    def add_message_bubble(self, sender, text, role="bot"):
        """
        role in {"user","bot","system"} controls colors and alignment.
        """
        ts = datetime.now().strftime(TIME_FMT)
        self._create_bubble(sender, text, role=role, ts=ts)

    def _build_header(self, header_height):
        self.header_frame = Frame(
            self.root,
            bg=THEME["input_bar_bg"],
            highlightbackground=THEME["input_border"],
            highlightthickness=1,
            height=header_height
        )
        self.header_frame.pack(side=tk.TOP, fill=tk.X)

        self.ai_avatar_photo = self._load_ai_avatar_image()
        avatar_kwargs = {
            "bg": THEME["input_bar_bg"]
        }
        if self.ai_avatar_photo:
            self.avatar_label = Label(self.header_frame, image=self.ai_avatar_photo, **avatar_kwargs)
        else:
            self.avatar_label = Label(
                self.header_frame,
                text=self.llm_obj.ai_username[:1],
                fg=THEME["assistant_name_color"],
                font=FONTS["name"],
                width=2,
                **avatar_kwargs
            )
        self.avatar_label.pack(side=tk.LEFT, padx=(12, 8), pady=8)

        self.header_name_label = Label(
            self.header_frame,
            text=self.llm_obj.ai_username,
            fg=THEME["assistant_name_color"],
            bg=THEME["input_bar_bg"],
            font=FONTS["header_name"]
        )
        self.header_name_label.pack(side=tk.LEFT, pady=8)

    def _load_ai_avatar_image(self, size=52):
        profile_path = getattr(self.llm_obj, "profile_picture_path", None)
        if not profile_path:
            return None
        try:
            image = Image.open(profile_path).convert("RGBA")
        except Exception:
            return None
        image = image.resize((size, size), Image.LANCZOS)
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        image.putalpha(mask)
        return ImageTk.PhotoImage(image)

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    user_name = simpledialog.askstring("User Name", "Please enter your name:")
    if not user_name:
        user_name = "User"
    ai_username = "James"
    root.deiconify()
    llm = LLMModel(ai_username=ai_username, 
                   personality_prompt_file="Personalities/James/james.txt",
                   profile_picture_path="Personalities/James/james.png",
                   user_name=user_name)
    app = BubbleChatApp(root, user_name, ai_username, llm)
    root.mainloop()
    llm.memory_manager.generate_compressed_memory()
