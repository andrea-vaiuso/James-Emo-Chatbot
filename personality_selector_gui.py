import tkinter as tk
from pathlib import Path
from typing import List, Optional
from PIL import Image, ImageTk, ImageDraw
from Personalities.personality import Personality

IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".gif", ".bmp"]


def discover_personalities(base_dir: str = "Personalities") -> List[Personality]:
    personalities = []
    root = Path(base_dir)
    if not root.exists():
        return personalities

    for sub in root.iterdir():
        if not sub.is_dir() or sub.name.startswith("__"):
            continue
        prompt_files = list(sub.glob("*.txt"))
        if not prompt_files:
            continue

        profile_picture = None
        for ext in IMAGE_EXTS:
            for f in sub.glob(f"*{ext}"):
                profile_picture = str(f)
                break
            if profile_picture:
                break

        personalities.append(
            Personality(
                name=sub.name,
                prompt_file=str(prompt_files[0]),
                profile_picture=profile_picture
            )
        )
    return personalities


class PersonalitySelector:
    def __init__(self, parent: tk.Tk, personalities: List[Personality], theme: dict, fonts: dict):
        self.selected: Optional[Personality] = None
        self.reset_requested = False
        self.cancelled = False
        self._theme = theme
        self._fonts = fonts
        self._photos = {}
        self._cards_count = 0
        self._canvas_window = None
        self.window = tk.Toplevel(parent)
        self.window.title("Choose a personality")
        self.window.configure(bg=theme["bg"])
        self.window.geometry("480x820")
        self.window.transient(parent)
        self.window.grab_set()
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self.window.update_idletasks()
        try:
            self.window.attributes("-topmost", True)
        except Exception:
            pass
        try:
            self.window.wait_visibility()
        except Exception:
            pass
        try:
            self.window.eval("tk::PlaceWindow %s center" % self.window.winfo_pathname(self.window.winfo_id()))
        except Exception:
            pass
        try:
            self.window.after(200, lambda: self.window.attributes("-topmost", False))
        except Exception:
            pass
        self.window.lift()
        self.window.focus_force()

        header = tk.Label(
            self.window,
            text="Pick your conversational companion",
            bg=theme["bg"],
            fg=theme["assistant_text_color"],
            font=fonts["header_name"],
            anchor="w"
        )
        header.pack(fill=tk.X, padx=14, pady=(16, 8))

        container = tk.Frame(self.window, bg=theme["bg"])
        container.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, bg=theme["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.list_frame = tk.Frame(canvas, bg=theme["bg"])
        self.list_frame.columnconfigure(0, weight=1)
        self.list_frame.columnconfigure(1, weight=1)
        self.list_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        self._canvas_window = canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        canvas.bind("<Configure>", self._on_canvas_configure)

        for personality in personalities:
            self._add_card(personality)

        btn_row = tk.Frame(self.window, bg=theme["bg"])
        btn_row.pack(fill=tk.X, padx=14, pady=(8, 14))
        tk.Button(
            btn_row,
            text="Start chatting",
            font=fonts["button"],
            command=self._confirm_selection,
            bg=theme["button_bg"],
            fg=theme["button_fg"],
            activebackground=theme["button_active_bg"],
            activeforeground=theme["button_active_fg"],
            relief="flat",
            bd=0,
            padx=14,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.RIGHT)
        tk.Button(
            btn_row,
            text="Reset user",
            font=fonts["button"],
            command=self._reset_user,
            bg=theme["bubble_sys_bg"],
            fg=theme["assistant_text_color"],
            activebackground=theme["bubble_bot_bg"],
            activeforeground=theme["assistant_text_color"],
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.LEFT)
        tk.Button(
            btn_row,
            text="Clear chat memory",
            font=fonts["button"],
            command=self._delete_memory,
            bg=theme["bubble_bot_bg"],
            fg=theme["assistant_text_color"],
            activebackground=theme["bubble_user_bg"],
            activeforeground=theme["assistant_text_color"],
            relief="flat",
            bd=0,
            padx=12,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=(8, 0))

    def _add_card(self, personality: Personality):
        card = tk.Frame(
            self.list_frame,
            bg=self._theme["bubble_bot_bg"],
            highlightbackground=self._theme["input_border"],
            highlightthickness=1,
            bd=0,
            padx=16,
            pady=16,
        )
        row = self._cards_count // 2
        col = self._cards_count % 2
        card.grid(row=row, column=col, padx=12, pady=12, sticky="nsew")
        self.list_frame.grid_rowconfigure(row, weight=1)
        self.list_frame.grid_columnconfigure(col, weight=1)
        self._cards_count += 1
        card.bind("<Button-1>", lambda _e, p=personality, c=card: self._select(p, c))

        avatar = self._load_avatar(personality.profile_picture, size=96)
        avatar_label = tk.Label(
            card,
            image=avatar,
            bg=self._theme["bubble_bot_bg"],
            width=96,
            height=96
        )
        avatar_label.image = avatar
        avatar_label.pack(side=tk.TOP, pady=(4, 10))
        avatar_label.bind("<Button-1>", lambda _e, p=personality, c=card: self._select(p, c))

        text_column = tk.Frame(card, bg=self._theme["bubble_bot_bg"])
        text_column.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        name_label = tk.Label(
            text_column,
            text=personality.name,
            font=(self._fonts["name"][0], self._fonts["name"][1] + 2, "bold"),
            fg=self._theme["assistant_name_color"],
            bg=self._theme["bubble_bot_bg"],
            anchor="center"
        )
        name_label.pack(anchor="center")
        desc = tk.Label(
            text_column,
            text="Click to chat",
            font=self._fonts["text"],
            fg=self._theme["assistant_text_color"],
            bg=self._theme["bubble_bot_bg"],
            anchor="center"
        )
        desc.pack(anchor="center")

        if self.selected is None:
            self._select(personality, card)

    def _load_avatar(self, path: Optional[str], size=54):
        if path is None or not Path(path).exists():
            placeholder = Image.new("RGBA", (size, size), self._hex_to_rgba(self._theme["assistant_name_color"]))
            mask = Image.new("L", (size, size), 0)
            ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
            placeholder.putalpha(mask)
            return ImageTk.PhotoImage(placeholder)

        image = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
        image.putalpha(mask)
        photo = ImageTk.PhotoImage(image)
        return photo

    def _hex_to_rgba(self, hex_color: str):
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 6:
            hex_color += "FF"
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4, 6))

    def _select(self, personality: Personality, card: tk.Frame):
        for child in self.list_frame.winfo_children():
            child.configure(highlightbackground=self._theme["input_border"])
        card.configure(highlightbackground=self._theme["button_bg"])
        self.selected = personality

    def _confirm_selection(self):
        self.window.destroy()

    def _reset_user(self):
        self.reset_requested = True
        self.selected = None
        self.window.destroy()

    def _delete_memory(self):
        if not self.selected:
            return
        mem_path = Path("Memory") / f"{self.selected.name}_memory.json"

        confirm = tk.Toplevel(self.window)
        confirm.title("Confirm delete")
        confirm.configure(bg=self._theme["bg"])
        tk.Label(
            confirm,
            text=f"Delete all memories for {self.selected.name}?",
            bg=self._theme["bg"],
            fg=self._theme["assistant_text_color"],
            font=self._fonts["text"],
            wraplength=280,
            justify="left",
            padx=12,
            pady=12
        ).pack(fill=tk.BOTH, expand=True)

        btns = tk.Frame(confirm, bg=self._theme["bg"])
        btns.pack(fill=tk.X, pady=(0, 10))

        def close_dialog():
            confirm.destroy()

        def do_delete():
            try:
                mem_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
            close_dialog()
            done = tk.Toplevel(self.window)
            done.title("Deleted")
            done.configure(bg=self._theme["bg"])
            tk.Label(
                done,
                text="Memories deleted.",
                bg=self._theme["bg"],
                fg=self._theme["assistant_text_color"],
                font=self._fonts["text"],
                padx=12,
                pady=12
            ).pack(fill=tk.BOTH, expand=True)
            tk.Button(
                done,
                text="OK",
                command=done.destroy,
                font=self._fonts["button"],
                bg=self._theme["button_bg"],
                fg=self._theme["button_fg"],
                activebackground=self._theme["button_active_bg"],
                activeforeground=self._theme["button_active_fg"],
                relief="flat",
                bd=0,
                padx=10,
                pady=6,
                cursor="hand2"
            ).pack(pady=(0, 10))

        tk.Button(
            btns,
            text="Cancel",
            command=close_dialog,
            font=self._fonts["button"],
            bg=self._theme["bubble_sys_bg"],
            fg=self._theme["assistant_text_color"],
            activebackground=self._theme["bubble_bot_bg"],
            activeforeground=self._theme["assistant_text_color"],
            relief="flat",
            bd=0,
            padx=10,
            pady=6,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=8)

        tk.Button(
            btns,
            text="Delete",
            command=do_delete,
            font=self._fonts["button"],
            bg=self._theme["button_bg"],
            fg=self._theme["button_fg"],
            activebackground=self._theme["button_active_bg"],
            activeforeground=self._theme["button_active_fg"],
            relief="flat",
            bd=0,
            padx=10,
            pady=6,
            cursor="hand2"
        ).pack(side=tk.RIGHT, padx=8)

        confirm.transient(self.window)
        confirm.grab_set()
        confirm.lift()
        confirm.update_idletasks()

    def _on_close(self):
        self.cancelled = True
        self.window.destroy()

    def _on_canvas_configure(self, event):
        # Make inner list frame match canvas width with side padding
        padding = 28  # keep same feel as card horizontal padding (14px each side)
        width = max(event.width - padding, 200)
        if self._canvas_window is not None:
            event.widget.itemconfig(self._canvas_window, width=width)


def select_personality(parent: tk.Tk, personalities: List[Personality], theme: dict, fonts: dict):
    if not personalities:
        return None, False, False
    dialog = PersonalitySelector(parent, personalities, theme, fonts)
    parent.wait_window(dialog.window)
    return dialog.selected, dialog.reset_requested, dialog.cancelled
