import tkinter as tk
from tkinter import messagebox
import threading
import time
from LLM.llm import LLMModel
from bubbleChat import BubbleChatApp
from personality_selector_gui import select_personality, discover_personalities
from user_profile_gui import load_or_create_user_profile, PROFILE_FILE

def load_theme_and_fonts(theme_path):
    import json
    with open(theme_path, "r", encoding="utf-8") as f:
        theme = json.load(f)
    fonts = theme.pop("fonts", {})
    for key, value in fonts.items():
        fonts[key] = tuple(value)
    return theme, fonts

if __name__ == "__main__":
    THEME, FONTS = load_theme_and_fonts("UI/theme_dark.json")
    root = tk.Tk()
    try:
        root.configure(bg=THEME["bg"])
    except Exception:
        pass
    try:
        root.geometry("440x820")
    except Exception:
        pass
    # Keep root invisible while setup dialogs run, but still present so child windows display
    try:
        root.attributes("-alpha", 0.0)
    except Exception:
        pass
    root.update_idletasks()
    root.deiconify()

    user_profile = load_or_create_user_profile(root, THEME, FONTS)
    if not user_profile:
        user_profile = {
            "name": "Andrea",
            "age": "30",
            "gender": "Not specified"
        }

    available_personalities = discover_personalities()
    save_state = {"thread": None}

    while True:
        try:
            root.deiconify()
            root.lift()
        except Exception:
            pass
        # Personality selection flow
        if save_state["thread"] and save_state["thread"].is_alive():
            save_state["thread"].join()
            save_state["thread"] = None

        while True:
            selected_personality, reset_user, cancelled = select_personality(root, available_personalities, THEME, FONTS)
            if cancelled:
                root.destroy()
                exit()
            if reset_user:
                try:
                    if PROFILE_FILE.exists():
                        PROFILE_FILE.unlink()
                except Exception:
                    pass
                user_profile = load_or_create_user_profile(root, THEME, FONTS)
                continue
            if not selected_personality:
                selected_personality = available_personalities[0]
            break

        user_name = user_profile.get("name", "User")
        ai_username = selected_personality.name

        # Show main window for chat
        try:
            root.attributes("-alpha", 1.0)
        except Exception:
            pass
        root.lift()

        # Splash screen while loading the LLM (prevents white flash)
        splash = tk.Toplevel(root)
        splash.configure(bg=THEME["bg"])
        splash.overrideredirect(True)
        msg = tk.Label(
            splash,
            text="Loading your companion...\nThis may take a moment",
            bg=THEME["bg"],
            fg=THEME["assistant_text_color"],
            font=FONTS["header_name"],
            justify="center",
            padx=24,
            pady=18,
        )
        msg.pack(fill=tk.BOTH, expand=True)
        splash.update_idletasks()
        try:
            w = splash.winfo_reqwidth()
            h = splash.winfo_reqheight()
            x = (splash.winfo_screenwidth() // 2) - (w // 2)
            y = (splash.winfo_screenheight() // 2) - (h // 2)
            splash.geometry(f"+{x}+{y}")
            splash.deiconify()
            splash.lift()
            splash.attributes("-topmost", True)
            splash.wait_visibility()
            splash.update()
            splash.after(400, lambda: splash.attributes("-topmost", False))
        except Exception:
            pass
        root.update_idletasks()
        root.update()

        try:
            llm = LLMModel(ai_username=ai_username, 
                        personality_prompt_file=selected_personality.prompt_file,
                        profile_picture_path=selected_personality.profile_picture,
                        user_name=user_name,
                        user_age=user_profile.get("age"),
                        user_gender=user_profile.get("gender", "Not specified"))
        finally:
            try:
                splash.destroy()
            except Exception:
                pass

        conversation_closed = {"value": False}
        def start_async_save():
            if save_state["thread"] and save_state["thread"].is_alive():
                return save_state["thread"]
            def _save():
                try:
                    llm.end_conversation()
                except Exception:
                    pass
            save_state["thread"] = threading.Thread(target=_save, daemon=True)
            save_state["thread"].start()
            return save_state["thread"]

        def close_conversation(wait: bool = False):
            if conversation_closed["value"]:
                return
            start_async_save()
            conversation_closed["value"] = True
            if wait and save_state["thread"]:
                save_state["thread"].join()

        back_requested = {"value": False}
        def handle_back():
            back_requested["value"] = True
            start_async_save()
            try:
                root.quit()
            except Exception:
                pass

        app = BubbleChatApp(root, user_name, ai_username, llm, theme=THEME, fonts=FONTS, on_back=handle_back)
        root.mainloop()
        close_conversation(wait=True)

        if not root.winfo_exists():
            break
        if back_requested["value"]:
            for child in root.winfo_children():
                try:
                    child.destroy()
                except Exception:
                    pass
            try:
                root.configure(bg=THEME["bg"])
            except Exception:
                pass
            try:
                root.withdraw()
            except Exception:
                pass
            try:
                root.update_idletasks()
            except Exception:
                pass
            continue
        break
