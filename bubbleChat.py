import tkinter as tk
from tkinter import Canvas, Frame, Label, Entry, Button
import threading
from datetime import datetime
from PIL import Image, ImageTk, ImageDraw
from utilities import play_system_sound, SOUNDS
from LLM.llm import LLMModel

# TODO: Rewrite readme with new features and system architecture (memory and emotions structure and hierarchy).
# TODO: Add summarization of important events and informations about the user and the bot profiles: (previous summarization as input, add only informations that are still not memorized, use a text similarity model to discard same memories).
# =========================
# self.theme / BEHAVIOR SETTINGS
# =========================

N = 3  # Number of messages between emotional updates
TIME_FMT = "%H:%M"  # NEW

# ===========================================
#  BUBBLE CHAT UI (TKINTER) + ROUNDED BUBBLES
# ===========================================
class BubbleChatApp:
    def __init__(self, root: tk.Tk, user_name: str, ai_username: str, llm: LLMModel, theme, fonts, max_history_length=8000, on_back=None):
        self.theme = theme
        self.fonts = fonts
        self.llm_obj = llm
        self.root = root
        self.on_back = on_back
        self.root.title("chat")
        self.root.configure(bg=self.theme["bg"])
        self.root.resizable(True, True)

        self.max_history_length = max_history_length

        self._build_header(self.theme["header_height"])

        # Container for canvas + scrollbar
        self.top_container = Frame(self.root, bg=self.theme["bg"])
        self.top_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Canvas + inner frame
        self.canvas = tk.Canvas(self.top_container, bg=self.theme["bg"], highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.top_container, orient="vertical", command=self.canvas.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.chat_frame = Frame(self.canvas, bg=self.theme["bg"])
        self.window_id = self.canvas.create_window((0, 0), window=self.chat_frame, anchor="nw")
        self._bubble_registry = {}
        self._current_wrap = self.theme["wrap"]

        # Bindings to keep width and scroll working correctly
        self.chat_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel(self.canvas)

        # Bottom input area
        self.bottom_frame = Frame(
            self.root,
            bg=self.theme["input_bar_bg"],
            highlightbackground=self.theme["input_border"],
            highlightthickness=1
        )
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.user_input = Entry(
            self.bottom_frame,
            font=self.fonts["input"],
            bg=self.theme["input_bg"],
            fg=self.theme["input_fg"],
            insertbackground=self.theme["input_fg"],
            relief="flat",
            bd=0,
            highlightbackground=self.theme["input_border"],
            highlightcolor=self.theme["input_border"],
            highlightthickness=1
        )
        self.user_input.pack(side=tk.LEFT, padx=(6, 6), pady=6, fill=tk.X, expand=True)
        self.user_input.bind("<Return>", self.on_send)

        self.send_button = Button(
            self.bottom_frame,
            text=f"{self.theme['button_icon']} Send",
            font=self.fonts["button"],
            command=self.on_send,
            bg=self.theme["button_bg"],
            fg=self.theme["button_fg"],
            activebackground=self.theme["button_active_bg"],
            activeforeground=self.theme["button_active_fg"],
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
        self.conversation_history = self.llm_obj.get_system_prompt(verbose=True)
        self.typing_after_id = None
        self.typing_visible = False
        self._typing_frame = None
        self._history_lock = threading.Lock()
        self._generation_id = 0
        self._current_cancel_event = None
        self._generation_thread = None
        self._pending_user_messages = []
        self._pending_history_segments = []
        self.last_interaction_label = None
        self._user_has_sent_first_message = False

        # Initial system greeting
        self.add_message_bubble("Bubble Chat", f"Hello {self.user_name}, welcome to the chat with {self.ai_username}!\nType 'exit' or 'quit' to leave.", role="system")

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
        return self.theme["wrap"]

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
        points = self._rounded_rect_points(width, height, self.theme["radius"])
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

        if self.typing_after_id is not None:
            try:
                self.root.after_cancel(self.typing_after_id)
            except Exception:
                pass
            self.typing_after_id = None
        if self.typing_visible:
            self._remove_typing_bubble()

        self.user_input.delete(0, tk.END)
        self.add_message_bubble(self.user_name, user_text, role="user")
        play_system_sound(SOUNDS["send"])
        if not self._user_has_sent_first_message:
            self._user_has_sent_first_message = True
            if self.last_interaction_label:
                self.last_interaction_label.config(text="online")

        if user_text.lower() in ["exit", "quit"]:
            self.add_message_bubble("System", "Exiting the chat. Goodbye!", role="system")
            self.root.after(800, self.root.destroy)
            return

        # Update conversation and start reply thread
        with self._history_lock:
            addition = f"<|start_header_id|>{self.user_name}<|end_header_id|>\n{user_text} <|eot_id|>\n"
            self.conversation_history += addition
            self.conversation_history = self.conversation_history[-self.max_history_length:]
            self.llm_obj.memory_chunks.append(f"{self.user_name}: {user_text}\n")
            self._pending_user_messages.append(user_text)
            self._pending_history_segments.append(addition)
        self._cancel_ongoing_generation()
        self._start_generation_thread()

    def _update_header_in_conversation_history(self):
        """ Updates the system prompt in the conversation history using init_conversation, but keeping the chat history. """
        with self._history_lock:
            old_history = self.conversation_history
            self.conversation_history = self.llm_obj.get_system_prompt(verbose=False)
            # Append old messages after the new system prompt
            split_index = old_history.find(f"<|start_header_id|>{self.user_name}<|end_header_id|>\n")
            if split_index != -1:
                user_and_bot_messages = old_history[split_index:]
                self.conversation_history += user_and_bot_messages
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def _refresh_conversation_state(self):
        with self._history_lock:
            total_messages = len(self.llm_obj.memory_chunks)
            if total_messages <= N:
                return False
            recent_text = "".join(self.llm_obj.memory_chunks[-N:])
        self.llm_obj.memory_manager.update_current_emotional_state(recent_text)
        self._update_header_in_conversation_history()
        return True
        
    def _cancel_ongoing_generation(self):
        if self._current_cancel_event is not None:
            self._current_cancel_event.set()

    def _start_generation_thread(self):
        self._generation_id += 1
        cancel_event = threading.Event()
        self._current_cancel_event = cancel_event
        thread = threading.Thread(target=self._llm_reply_thread, args=(cancel_event, self._generation_id), daemon=True)
        self._generation_thread = thread
        thread.start()

    def _llm_reply_thread(self, cancel_event: threading.Event, generation_id: int):
        self._refresh_conversation_state()
        # Ensure previous typing bubble is gone before showing new one
        self.root.after(0, self._remove_typing_bubble)
        self.typing_after_id = self.root.after(0, lambda: self._show_typing_bubble(f"{self.llm_obj.ai_username} is writing..."))

        with self._history_lock:
            tail_segments = list(self._pending_history_segments)
            tail_len = sum(len(seg) for seg in tail_segments)
            base_history = self.conversation_history[:-tail_len] if tail_len else self.conversation_history
            combined_user_text = "\n".join(self._pending_user_messages)
            if combined_user_text:
                prompt_history = base_history + f"<|start_header_id|>{self.user_name}<|end_header_id|>\n{combined_user_text} <|eot_id|>\n"
            else:
                prompt_history = self.conversation_history
        if cancel_event.is_set():
            return

        response = self.llm_obj.generate_response(prompt_history)

        if cancel_event.is_set():
            return

        with self._history_lock:
            updated_history = base_history
            if combined_user_text:
                updated_history += f"<|start_header_id|>{self.user_name}<|end_header_id|>\n{combined_user_text} <|eot_id|>\n"
            updated_history += f"<|start_header_id|>{self.llm_obj.ai_username}<|end_header_id|>\n{response}\n<|eot_id|>\n"
            self.conversation_history = updated_history[-self.max_history_length:]
            self.llm_obj.memory_chunks.append(f"{self.llm_obj.ai_username}: {response}\n")
            self._pending_user_messages.clear()
            self._pending_history_segments.clear()

        self._refresh_conversation_state()

        def finalize_ui():
            if cancel_event.is_set() or generation_id != self._generation_id:
                return
            if self.typing_after_id is not None:
                try:
                    self.root.after_cancel(self.typing_after_id)
                except Exception:
                    pass
                self.typing_after_id = None
            if self.typing_visible:
                self._remove_typing_bubble()

            self.add_message_bubble(self.llm_obj.ai_username, response, role="bot")
            play_system_sound(SOUNDS["recv"])

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
        outer = Frame(self.chat_frame, bg=self.theme["bg"])
        outer.pack(fill=tk.X, pady=3, padx=10)

        # alignment
        align = "e" if role == "user" else "w"

        # colors
        if role == "user":
            bb = self.theme["bubble_user_bg"]
            name_color = self.theme["user_name_color"]
            msg_color = self.theme["user_text_color"]
        elif role == "system":
            bb = self.theme["bubble_sys_bg"]
            name_color = self.theme["system_name_color"]
            msg_color = self.theme["system_text_color"]
        elif role == "typing":
            bb = self.theme["bubble_bot_bg"]
            name_color = self.theme["assistant_name_color"]
            msg_color = self.theme["assistant_text_color"]
        else:
            bb = self.theme["bubble_bot_bg"]
            name_color = self.theme["assistant_name_color"]
            msg_color = self.theme["assistant_text_color"]

        # Canvas-based bubble
        c = Canvas(outer, bg=self.theme["bg"], highlightthickness=0)
        c.pack(anchor=align, padx=0)

        # A frame inside canvas to hold labels
        inner = Frame(c, bg=bb)
        # Labels: name on first line (bold), text on second line
        name_lbl = Label(inner, text=name, fg=name_color, bg=bb, font=self.fonts["name"], anchor="w", justify="left")
        name_lbl.pack(anchor="w", padx=(self.theme["pad_in"][0], self.theme["pad_in"][0]), pady=(self.theme["pad_in"][1], 0))

        msg_font = self.fonts["typing"] if role == "typing" else self.fonts["text"]
        if role == "typing":
            msg_color = self.theme["typing_color"]
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
        text_lbl.pack(anchor="w", padx=(self.theme["pad_in"][0], self.theme["pad_in"][0]), pady=(1, self.theme["pad_in"][1]))

        if ts is None:
            ts = datetime.now().strftime(TIME_FMT)
        meta_row = Frame(inner, bg=bb)
        meta_row.pack(anchor="e",
                    fill="x",
                    padx=(self.theme["pad_in"][0], self.theme["pad_in"][0]),
                    pady=(0, self.theme["pad_in"][1]))
        ts_lbl = Label(meta_row, text=ts, fg=self.theme["timestamp_color"],
                    bg=bb, font=self.fonts["timestamp"])
        ts_lbl.pack(side="right")

        # Materialize sizes to draw rounded rect behind
        inner.update_idletasks()
        pad_x = 3
        pad_y = 1
        width = inner.winfo_reqwidth() + pad_x * 2
        height = inner.winfo_reqheight() + pad_y * 2
        points = self._rounded_rect_points(width, height, self.theme["radius"])
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
            bg=self.theme["header_bar_bg"],
            highlightbackground=self.theme["input_border"],
            highlightthickness=1,
            height=header_height
        )
        self.header_frame.pack(side=tk.TOP, fill=tk.X)

        self.ai_avatar_photo = self._load_ai_avatar_image()
        avatar_kwargs = {
            "bg": self.theme["header_avatar_bg"],
        }
        if self.ai_avatar_photo:
            self.avatar_label = Label(self.header_frame, image=self.ai_avatar_photo, bg=self.theme["header_bar_bg"])
        else:
            self.avatar_label = Label(
                self.header_frame,
                text=self.llm_obj.ai_username[:1],
                fg=self.theme["assistant_name_color"],
                font=self.fonts["name"],
                width=2,
                **avatar_kwargs
            )
        self.avatar_label.pack(side=tk.LEFT, padx=(12, 8), pady=8)

        name_container = Frame(self.header_frame, bg=self.theme["header_bar_bg"])
        name_container.pack(side=tk.LEFT, pady=8)

        self.header_name_label = Label(
            name_container,
            text=self.llm_obj.ai_username,
            fg=self.theme["assistant_name_color"],
            bg=self.theme["header_bar_bg"],
            font=self.fonts["header_name"]
        )
        self.header_name_label.pack(anchor="w")

        self.last_interaction_label = Label(
            name_container,
            text=self._format_last_interaction_text(),
            fg=self.theme.get("timestamp_color", self.theme["assistant_text_color"]),
            bg=self.theme["header_bar_bg"],
            font=self.fonts.get("timestamp", self.fonts["header_name"])
        )
        self.last_interaction_label.pack(anchor="w")

        self.back_button = Button(
            self.header_frame,
            text="(Back)",
            command=self._handle_back,
            bg=self.theme["header_bar_bg"],
            fg=self.theme["assistant_name_color"],
            activebackground=self.theme["header_bar_bg"],
            activeforeground=self.theme["assistant_name_color"],
            relief="flat",
            bd=0,
            cursor="hand2",
            font=self.fonts["button"],
            padx=12,
            pady=6
        )
        self.back_button.pack(side=tk.RIGHT, padx=(0, 12), pady=8)

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

    def _format_last_interaction_text(self):
        try:
            timing_info = self.llm_obj.memory_manager.get_timing_information_prompt()
        except Exception:
            return "Last interaction: unknown"
        lines = timing_info.splitlines()
        for line in lines:
            clean = line.replace("[Timing Information]:", "").strip()
            if not clean:
                continue
            if "interaction" in clean.lower():
                return clean
        fallback = timing_info.strip()
        return fallback if fallback else "Last interaction: unknown"

    def _update_last_interaction_label(self):
        if self.last_interaction_label:
            self.last_interaction_label.config(text=self._format_last_interaction_text())

    def _handle_back(self):
        self._cancel_ongoing_generation()
        self._remove_typing_bubble()
        if self.on_back:
            try:
                self.on_back()
            except Exception:
                pass
        else:
            try:
                self.root.quit()
            except Exception:
                pass
