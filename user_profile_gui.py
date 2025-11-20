import json
from pathlib import Path
import tkinter as tk
from typing import Optional, Dict

PROFILE_FILE = Path(__file__).resolve().parent / "user_profile.json"


def load_saved_profile(path: Path = PROFILE_FILE):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def persist_profile(profile: dict, path: Path = PROFILE_FILE):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


class UserProfileDialog:
    def __init__(self, parent: tk.Tk, theme: dict, fonts: dict, existing: Optional[Dict] = None):
        self.result = None
        self._theme = theme
        self._fonts = fonts
        self._parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("User setup")
        self.window.configure(bg=theme["bg"])
        self.window.transient(parent)
        self.window.grab_set()
        self.window.protocol("WM_DELETE_WINDOW", self._on_cancel)
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

        existing = existing or {}
        self.name_var = tk.StringVar(value=existing.get("name", ""))
        self.age_var = tk.StringVar(value=str(existing.get("age", "")))
        self.gender_var = tk.StringVar(
            value=str(existing.get("gender", "Not specified"))
        )

        title = tk.Label(
            self.window,
            text="Who is chatting?",
            bg=theme["bg"],
            fg=theme["assistant_text_color"],
            font=fonts["header_name"]
        )
        title.pack(pady=(16, 6))

        form = tk.Frame(self.window, bg=theme["bg"])
        form.pack(fill=tk.BOTH, expand=True, padx=16, pady=10)

        self._add_field(form, "Name", self.name_var)
        self._add_field(form, "Age", self.age_var)
        self._add_gender_field(form)

        btn_row = tk.Frame(self.window, bg=theme["bg"])
        btn_row.pack(fill=tk.X, padx=16, pady=(10, 14))

        Button = tk.Button  # keep tkinter reference short
        Button(
            btn_row,
            text="Use profile",
            command=self._on_submit,
            font=fonts["button"],
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

    def _add_field(self, parent, label, var):
        row = tk.Frame(parent, bg=self._theme["bg"])
        row.pack(fill=tk.X, pady=6)
        tk.Label(
            row,
            text=label,
            bg=self._theme["bg"],
            fg=self._theme["user_text_color"],
            font=self._fonts["text"],
            anchor="w"
        ).pack(side=tk.TOP, anchor="w")
        tk.Entry(
            row,
            textvariable=var,
            bg=self._theme["input_bg"],
            fg=self._theme["input_fg"],
            insertbackground=self._theme["input_fg"],
            relief="flat",
            highlightbackground=self._theme["input_border"],
            highlightcolor=self._theme["input_border"],
            highlightthickness=1,
            font=self._fonts["input"]
        ).pack(fill=tk.X, pady=(4, 0))

    def _add_gender_field(self, parent):
        row = tk.Frame(parent, bg=self._theme["bg"])
        row.pack(fill=tk.X, pady=6)
        tk.Label(
            row,
            text="Sex",
            bg=self._theme["bg"],
            fg=self._theme["user_text_color"],
            font=self._fonts["text"],
            anchor="w"
        ).pack(side=tk.TOP, anchor="w")
        options = ["Male", "Female", "Not specified"]
        picker = tk.Frame(row, bg=self._theme["bg"])
        picker.pack(anchor="w", pady=(4, 0))
        for opt in options:
            tk.Radiobutton(
                picker,
                text=opt,
                value=opt,
                variable=self.gender_var,
                bg=self._theme["bg"],
                fg=self._theme["user_text_color"],
                selectcolor=self._theme["bubble_bot_bg"],
                activebackground=self._theme["bg"],
                activeforeground=self._theme["user_text_color"],
                font=self._fonts["input"],
                highlightthickness=0
            ).pack(side=tk.LEFT, padx=(0, 12))

    def _on_submit(self):
        name = self.name_var.get().strip() or "User"
        age_str = self.age_var.get().strip()
        try:
            age_int = int(age_str)
            if age_int < 0:
                raise ValueError
            age = str(age_int)
        except Exception:
            age = age_str if age_str else "Unknown"

        gender = self.gender_var.get().strip() or "Not specified"

        self.result = {"name": name, "age": age, "gender": gender}
        self.window.destroy()

    def _on_cancel(self):
        # Default profile if user closes the dialog
        self.result = {
            "name": self.name_var.get().strip() or "User",
            "age": self.age_var.get().strip() or "Unknown",
            "gender": self.gender_var.get().strip() or "Not specified"
        }
        self.window.destroy()


def load_or_create_user_profile(parent: tk.Tk, theme: dict, fonts: dict) -> dict:
    saved = load_saved_profile()
    if saved and saved.get("name"):
        if "gender" not in saved or not saved.get("gender"):
            saved["gender"] = "Not specified"
        if "age" not in saved or not saved.get("age"):
            saved["age"] = "Unknown"
        return saved

    dialog = UserProfileDialog(parent, theme, fonts, existing=saved)
    parent.wait_window(dialog.window)
    profile = dialog.result or {"name": "User", "age": "Unknown", "gender": "Not specified"}
    persist_profile(profile)
    return profile
