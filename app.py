from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import os

from config_manager import load_settings, save_settings, load_doc_registry, save_doc_registry
from history_manager import (
    ChatMessage,
    derive_title_from_text,
    list_sessions,
    load_session,
    load_session_title,
    new_session_id,
    prune_sessions,
    save_session,
    session_path,
)
from markdown_render import configure_markdown_tags, render_markdown
from spell_checker import SpellGrammarChecker, SpanError
from thread_manager import (
    LlmTask,
    index_in,
    index_out,
    llm_in,
    llm_out,
    next_task_id,
    spell_in,
    spell_out,
    start_workers,
)

from guardrails import filter_rag_contexts

ctk.set_default_color_theme("green")


SIDEBAR_W = 280


LIGHT = {
    # monochrome surfaces
    "bg": "#FFFFFF",
    "sidebar": "#FFFFFF",
    "header": "#1A1A1A",
    "header_text": "#FFFFFF",
    "header_muted": "#AAAAAA",
    "input": "#FFFFFF",
    "chat_panel": "#FFFFFF",
    "chat_inset": "#FFFFFF",
    # bubbles
    "user_bg": "#000000",
    "assistant_bg": "#FFFFFF",
    # text
    "text": "#000000",
    "muted": "#3A3A3A",
    "border": "#CFCFCF",
    # interactive (still monochrome)
    "accent": "#000000",
    "send": "#000000",
    "send_hover": "#1A1A1A",
    "send_disabled": "#BDBDBD",
    # scrollbar
    "scroll_thumb": "#BDBDBD",
    "scroll_hover": "#8F8F8F",
    # code/highlights
    "code_bg": "#F4F4F4",
    "hl_spell_bg": "#F2F2F2",
    "hl_grammar_bg": "#ECECEC",
    "error_spell": "#000000",
    "error_grammar": "#000000",
    # active states
    "session_active": "#F2F2F2",
    "session_border": "#000000",
    "pill_active": "#333333",
    "pill_border": "#555555",
}

# Dark mode intentionally removed: always use the LIGHT palette.
DARK = LIGHT


@dataclass
class UiMessage:
    role: str
    content: str
    mode: str
    ts: float


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        if self.settings.font_size < 12:
            self.settings.font_size = 15
            save_settings(self.settings)

        self.colors = DARK if self.settings.theme == "dark" else LIGHT

        self.title("Spell & Grammar Check")
        self.minsize(1100, 780)

        # Open maximized covering the entire screen
        self.after(10, lambda: self._post_init_window())

        self.session_title = "New chat"
        self.session_id = new_session_id(self.session_title)
        self.messages: List[UiMessage] = []
        self._active_session_id: Optional[str] = None
        self._session_buttons: dict[str, ctk.CTkButton] = {}

        self._pending_task_id: Optional[float] = None
        self._pending_mode: Optional[str] = None
        self._pending_widget: Optional[tk.Text] = None
        self._placeholder_active = False
        self._debounce_after_id: Optional[str] = None
        self._last_spell_req_id: float = 0.0
        self._mode = "Spell & Grammar"
        self._indexing_inflight = 0
        self._upload_anim_phase = 0
        self._doc_items: dict[str, dict] = {}
        self._active_doc_id: Optional[str] = None
        self._doc_ids: List[str] = []

        saved_registry = load_doc_registry()
        for doc_id, name in saved_registry.items():
            self._doc_ids.append(doc_id)
            self._doc_items[doc_id] = {"name": name, "button": None}
        if self._doc_ids:
            self._active_doc_id = self._doc_ids[0]
        self._sidebar_collapsed = False
        self._index_toast: Optional[ctk.CTkFrame] = None
        self._send_spin_phase = 0
        self._send_spinning = False
        self._pending_rag_sources: Optional[str] = None

        self._checker = SpellGrammarChecker()
        start_workers(spell_checker=self._checker)

        self._show_splash()
        self._build_layout()
        self._apply_theme()
        if self._doc_items:
            self._rebuild_docs_list()
        self._refresh_docs_sidebar()
        self._load_sidebar_sessions()
        self._active_session_id = self.session_id
        self._refresh_active_chat_style()

        self._poll_llm_results()
        self._poll_spell_results()
        self._poll_index_results()

        self.after(150, lambda: self._text.focus_set())

    # ---------- layout ----------
    def _build_layout(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.root = ctk.CTkFrame(self, corner_radius=0)
        self.root.grid(row=0, column=0, sticky="nsew")
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Sidebar (ChatGPT-like, collapsible) with right border
        self.sidebar = ctk.CTkFrame(self.root, corner_radius=0, width=SIDEBAR_W,
                                    border_width=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.sidebar.grid_columnconfigure(0, weight=1)
        self.sidebar.grid_rowconfigure(2, weight=1)

        self.sidebar_right_border = ctk.CTkFrame(self.root, corner_radius=0, width=1,
                                                  fg_color=self.colors["border"])
        self.sidebar_right_border.grid(row=0, column=0, sticky="nse")

        # Black header bar in sidebar (matches main header)
        self.sidebar_top = ctk.CTkFrame(self.sidebar, corner_radius=0,
                                        fg_color=self.colors["header"], height=56)
        self.sidebar_top.grid(row=0, column=0, sticky="ew")
        self.sidebar_top.grid_propagate(False)
        self.sidebar_top.grid_columnconfigure(1, weight=1)

        self.sidebar_toggle = ctk.CTkButton(
            self.sidebar_top,
            text="❯",
            width=34,
            height=34,
            corner_radius=10,
            fg_color="transparent",
            hover_color="#333333",
            text_color=self.colors["header_text"],
            command=self._toggle_sidebar,
        )
        self.sidebar_toggle.grid(row=0, column=0, padx=(10, 0), pady=11, sticky="w")

        self.brand = ctk.CTkLabel(
            self.sidebar_top, text="Spell & Grammar",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=self.colors["header_text"])
        self.brand.grid(row=0, column=1, padx=(8, 0), pady=11, sticky="w")

        # Sidebar bottom border to match header border
        self.sidebar_header_border = ctk.CTkFrame(self.sidebar, corner_radius=0,
                                                   height=1, fg_color=self.colors["border"])
        self.sidebar_header_border.grid(row=1, column=0, sticky="ew")

        # Sidebar content (hidden when collapsed)
        self.sidebar_content = ctk.CTkFrame(self.sidebar, corner_radius=0, fg_color="transparent")
        self.sidebar_content.grid(row=2, column=0, sticky="nsew", padx=8, pady=(10, 0))
        self.sidebar_content.grid_columnconfigure(0, weight=1)
        self.sidebar_content.grid_rowconfigure(5, weight=1)

        self.new_chat_btn = ctk.CTkButton(
            self.sidebar_content,
            text="New chat",
            height=38,
            corner_radius=10,
            fg_color="transparent",
            border_width=1,
            command=self._new_chat,
        )
        self.new_chat_btn.grid(row=0, column=0, padx=8, pady=(2, 8), sticky="ew")

        self.history_limit_btn = ctk.CTkButton(
            self.sidebar_content,
            text=f"Prompt limit: {self.settings.history_keep_messages}",
            height=34, corner_radius=10,
            fg_color="transparent", border_width=1, border_color=self.colors["border"],
            text_color=self.colors["text"],
            hover_color=self._shade(self.colors["bg"], 0.06),
            command=self._set_history_limit,
        )
        self.history_limit_btn.grid(row=1, column=0, padx=8, pady=(0, 10), sticky="ew")

        # Doc Q&A uploaded documents — hidden until files are actually uploaded
        self.docs_label = ctk.CTkLabel(self.sidebar_content, text="Uploaded Documents",
                                       text_color=self.colors["muted"],
                                       font=ctk.CTkFont(size=12))
        self.docs_label.grid(row=2, column=0, padx=10, pady=(0, 0), sticky="w")
        self.docs_label.grid_remove()

        self.docs_frame = ctk.CTkFrame(self.sidebar_content, corner_radius=0, fg_color="transparent")
        self.docs_frame.grid(row=3, column=0, padx=8, pady=(0, 0), sticky="ew")
        self.docs_frame.grid_columnconfigure(0, weight=1)
        self.docs_frame.grid_remove()

        self.chats_heading = ctk.CTkLabel(
            self.sidebar_content, text="Chats",
            text_color=self.colors["muted"],
            font=ctk.CTkFont(size=12))
        self.chats_heading.grid(row=4, column=0, padx=10, pady=(4, 2), sticky="w")

        self.sessions_frame = ctk.CTkScrollableFrame(
            self.sidebar_content, corner_radius=0, border_width=0)
        self.sessions_frame.grid(row=5, column=0, padx=0, pady=(0, 0), sticky="nsew")
        self.sessions_frame.grid_columnconfigure(0, weight=1)

        # Main panel
        self.main = ctk.CTkFrame(self.root, corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew")
        # main window: header + thread + composer
        self.main.grid_rowconfigure(1, weight=1)
        self.main.grid_columnconfigure(0, weight=1)

        # Header (main top bar)
        self.header = ctk.CTkFrame(self.main, corner_radius=0, height=56)
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.grid_propagate(False)
        self.header.grid_columnconfigure(1, weight=1)
        self.header.grid_columnconfigure(2, weight=1)

        # Header title removed (mode buttons start immediately)
        self.header_spacer = ctk.CTkFrame(self.header, width=12, height=1, corner_radius=0, fg_color="transparent")
        self.header_spacer.grid(row=0, column=0, padx=12, pady=16, sticky="w")

        # Mode buttons
        self.mode_bar = ctk.CTkFrame(self.header, corner_radius=0, fg_color="transparent")
        self.mode_bar.grid(row=0, column=1, pady=12, padx=(0, 8), sticky="w")
        self.mode_buttons: dict[str, ctk.CTkButton] = {}
        mode_widths = {
            "Spell & Grammar": 120,
            "Email": 62,
            "Academic": 90,
            "LaTeX": 64,
            "Doc Q&A": 78,
        }
        for i, m in enumerate(["Spell & Grammar", "Email", "Academic", "LaTeX", "Doc Q&A"]):
            b = ctk.CTkButton(
                self.mode_bar,
                text=m,
                width=mode_widths.get(m, 78),
                height=30,
                corner_radius=8,
                fg_color="transparent",
                hover_color="#333333",
                text_color=self.colors["header_text"],
                command=lambda mm=m: self._set_mode(mm),
            )
            b.grid(row=0, column=i, padx=1)
            self.mode_buttons[m] = b

        # Top-right controls
        self.font_menu = ctk.CTkOptionMenu(
            self.header,
            values=["Georgia", "Times New Roman", "Garamond", "Arial",
                    "Segoe UI", "Calibri", "Cambria", "Courier New"],
            command=self._set_font_family,
            width=155,
        )
        self.font_menu.set(self.settings.font_family)
        self.font_menu.grid(row=0, column=3, padx=(0, 8), pady=12, sticky="e")

        self.font_size_menu = ctk.CTkOptionMenu(
            self.header,
            values=[str(x) for x in [12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 28, 32]],
            command=self._set_font_size,
            width=72,
        )
        self.font_size_menu.set(str(self.settings.font_size))
        self.font_size_menu.grid(row=0, column=4, padx=(0, 8), pady=12, sticky="e")

        self.theme_toggle = ctk.CTkSwitch(self.header, text="Theme",
                                         text_color=self.colors["header_text"],
                                         command=self._toggle_theme)
        self.theme_toggle.grid(row=0, column=5, padx=(0, 10), pady=12, sticky="e")

        self.clear_btn = ctk.CTkButton(self.header, text="🗑", width=36, height=32, corner_radius=8,
                                       fg_color="transparent",
                                       text_color=self.colors["header_text"],
                                       hover_color="#333333",
                                       command=self._clear_chat)
        self.clear_btn.grid(row=0, column=6, padx=(0, 18), pady=12, sticky="e")

        # Header bottom border (1px line)
        self.header_border = ctk.CTkFrame(self.main, corner_radius=0, height=1)
        self.header_border.grid(row=0, column=0, sticky="ews")

        # Thread (scrollable, fills remaining space)
        self.thread = ctk.CTkFrame(self.main, corner_radius=0)
        self.thread.grid(row=1, column=0, sticky="nsew")
        self.thread.grid_rowconfigure(0, weight=1)
        self.thread.grid_columnconfigure(0, weight=1)

        # Centered conversation column like ChatGPT
        self.chat = ctk.CTkScrollableFrame(self.thread, corner_radius=0)
        self.chat.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        self.chat.grid_columnconfigure(0, weight=1)

        self.chat_inner = ctk.CTkFrame(self.chat, corner_radius=0, fg_color="transparent")
        self.chat_inner.grid(row=0, column=0, padx=60, pady=(10, 4), sticky="ew")
        self.chat_inner.grid_columnconfigure(0, weight=1)

        # Session title inside messages pane (above messages)
        self.chat_session_title = ctk.CTkLabel(
            self.chat_inner, text="New chat",
            text_color=self.colors["text"],
            font=ctk.CTkFont(size=20, weight="bold"))
        self.chat_session_title.grid(row=0, column=0, sticky="w", pady=(4, 10))

        self.empty_state = ctk.CTkFrame(self.chat_inner, corner_radius=0, fg_color="transparent")
        self.empty_state.grid(row=1, column=0, pady=60)
        ctk.CTkLabel(self.empty_state, text="Start a conversation", font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(0, 6))
        ctk.CTkLabel(self.empty_state, text="Ask questions about your documents or research papers", text_color=self.colors["muted"]).pack()

        # Composer (sticky bottom)
        self.composer = ctk.CTkFrame(self.main, corner_radius=0)
        self.composer.grid(row=2, column=0, sticky="ew")
        self.composer.grid_columnconfigure(0, weight=1)

        self.composer_border = ctk.CTkFrame(self.composer, corner_radius=0, height=1)
        self.composer_border.grid(row=0, column=0, sticky="ew")

        self.input_row = ctk.CTkFrame(self.composer, corner_radius=0, fg_color="transparent")
        self.input_row.grid(row=1, column=0, sticky="ew", padx=60, pady=(6, 8))
        self.input_row.grid_columnconfigure(0, weight=1)

        self.input_wrap = ctk.CTkFrame(self.input_row, corner_radius=12,
                                       border_width=1, border_color=self.colors["border"])
        self.input_wrap.grid(row=0, column=0, sticky="ew")
        self.input_wrap.grid_propagate(True)
        self.input_wrap.grid_columnconfigure(1, weight=1)

        self.upload_btn = ctk.CTkButton(self.input_wrap, text="📎", width=42, height=42,
                                       corner_radius=10, font=ctk.CTkFont(size=20),
                                       command=self._upload_files)
        self.upload_btn.grid(row=0, column=0, padx=(10, 8), pady=8, sticky="w")

        self.input = ctk.CTkTextbox(
            self.input_wrap,
            height=56,
            corner_radius=10
        )

        self.input._textbox.configure(
            padx=16,
            pady=10
        )
        self.input.grid(row=0, column=1, padx=(0, 8), pady=8, sticky="ew")

        self.send_btn = ctk.CTkButton(self.input_wrap, text="➤", width=40, height=40, corner_radius=10,
                                      font=ctk.CTkFont(size=18), command=self._on_send)
        self.send_btn.grid(row=0, column=2, padx=(0, 10), pady=8, sticky="e")

        self._text = self.input._textbox

        self.placeholder_label = tk.Label(
            self._text,
            text="Type a message…",
            fg=self.colors["muted"],
            bg=self.colors["input"],
            font=(self.settings.font_family, self.settings.font_size),
            anchor="w",
            cursor="xterm",
        )
        self.placeholder_label.place(in_=self._text, x=0, y=0)
        self.placeholder_label.bind("<Button-1>", lambda e: self._focus_input())

        self._text.bind("<FocusIn>", self._on_focus_in)
        self._text.bind("<FocusOut>", self._on_focus_out)
        self._text.bind("<KeyRelease>", self._on_input_key)
        self._text.bind("<Return>", self._on_enter_send)
        self._text.bind("<Shift-Return>", self._on_shift_enter)
        self._text.bind("<<Modified>>", self._on_text_modified)

        self._text.tag_config("spell_error", underline=True, underlinefg=self.colors["error_spell"], background=self.colors["hl_spell_bg"])
        self._text.tag_config("grammar_error", underline=True, underlinefg=self.colors["error_grammar"], background=self.colors["hl_grammar_bg"])

        self._update_placeholder_visibility()
        self._set_mode("Spell & Grammar")

        self._attach_tooltips()

    # ---------- theme / style ----------
    def _apply_theme(self):
        # Light-only UI
        ctk.set_appearance_mode("Light")
        self.settings.theme = "light"
        self.colors = LIGHT

        self.configure(fg_color=self.colors["bg"])
        self.root.configure(fg_color=self.colors["bg"])
        self.sidebar.configure(fg_color=self.colors["sidebar"])
        try:
            self.sidebar_right_border.configure(fg_color=self.colors["border"])
        except Exception:
            pass
        self.main.configure(fg_color=self.colors["chat_panel"])
        self.header.configure(fg_color=self.colors["header"])
        self.header_border.configure(fg_color=self.colors["border"])
        try:
            self.sidebar_top.configure(fg_color=self.colors["header"])
        except Exception:
            pass
        try:
            self.sidebar_header_border.configure(fg_color=self.colors["border"])
        except Exception:
            pass
        try:
            self.sidebar_content.configure(fg_color=self.colors["sidebar"])
        except Exception:
            pass
        
        try:
            self.thread.configure(fg_color=self.colors["chat_panel"])
        except Exception:
            pass
        self.chat.configure(fg_color=self.colors["chat_panel"])
        self.composer.configure(fg_color=self.colors["chat_panel"])
        try:
            self.composer_border.configure(fg_color=self.colors["border"])
        except Exception:
            pass
        self.input_wrap.configure(
            fg_color=self.colors["input"],
            border_width=0,
            border_color=self.colors["border"],
        )

        self.input.configure(
            fg_color="transparent",
            border_width=0
        )       
        self.send_btn.configure(
            fg_color=self.colors["send"],
            hover_color=self.colors["send_hover"],
            text_color="#FFFFFF",
        )
        try:
            self.upload_btn.configure(
                fg_color="transparent",
                text_color=self.colors["text"],
                hover_color=self._shade(self.colors["bg"], 0.08),
                border_width=1,
                border_color=self.colors["border"],
            )
        except Exception:
            pass
        # New chat dashed-ish feel
        self.new_chat_btn.configure(
            border_color=self.colors["border"],
            fg_color="transparent",
            text_color=self.colors["text"],
            hover_color=self._shade(self.colors["bg"], 0.06),
        )
        self.history_limit_btn.configure(
            text_color=self.colors["text"], fg_color="transparent",
            border_color=self.colors["border"],
            hover_color=self._shade(self.colors["bg"], 0.06))
        self.clear_btn.configure(
            text_color=self.colors["header_text"], fg_color="transparent",
            hover_color="#333333")
        self.theme_toggle.configure(text_color=self.colors["header_text"])
        self.brand.configure(text_color=self.colors["header_text"])
        try:
            self.sidebar_toggle.configure(
                text_color=self.colors["header_text"],
                hover_color="#333333",
            )
        except Exception:
            pass
        try:
            self.chat_session_title.configure(text_color=self.colors["text"])
        except Exception:
            pass

        # Style option menus on dark header
        try:
            for om in [self.font_menu, self.font_size_menu]:
                om.configure(
                    fg_color="#2A2A2A",
                    text_color=self.colors["header_text"],
                    button_color="#444444",
                    button_hover_color="#555555",
                    dropdown_fg_color="#2A2A2A",
                    dropdown_text_color=self.colors["header_text"],
                    dropdown_hover_color="#444444",
                )
        except Exception:
            pass

        self._style_scrollbars()
        self._refresh_active_chat_style()

        self._text.tag_config("spell_error", underline=True, underlinefg=self.colors["error_spell"], background=self.colors["hl_spell_bg"])
        self._text.tag_config("grammar_error", underline=True, underlinefg=self.colors["error_grammar"], background=self.colors["hl_grammar_bg"])
        try:
            self.placeholder_label.configure(fg=self.colors["muted"], bg=self.colors["input"], font=(self.settings.font_family, self.settings.font_size))
        except Exception:
            pass
        self._update_placeholder_visibility()

        self.theme_toggle.deselect()

        # Re-render chat so tk.Text backgrounds match theme
        self._rerender_messages()

        # Force mode button text colors (white on dark header)
        try:
            for b in self.mode_buttons.values():
                b.configure(text_color=self.colors["header_text"])
        except Exception:
            pass
        try:
            for b in self._session_buttons.values():
                b.configure(text_color=self.colors["text"])
        except Exception:
            pass
        try:
            self.docs_label.configure(text_color=self.colors["muted"])
            for info in self._doc_items.values():
                info["button"].configure(text_color=self.colors["text"])
        except Exception:
            pass
        try:
            self.chats_heading.configure(text_color=self.colors["muted"])
        except Exception:
            pass

        self._update_send_enabled()

        try:
            self.sessions_frame.configure(
                fg_color=self.colors["sidebar"], border_width=0)
        except Exception:
            pass

        # Top-right controls on dark header
        try:
            self.font_menu.configure(button_color="#444444", fg_color="#2A2A2A")
            self.font_size_menu.configure(button_color="#444444", fg_color="#2A2A2A")
        except Exception:
            pass

        # input focus glow
        self._text.configure(
            insertbackground=self.colors["text"],
            highlightthickness=0,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["accent"],
        )

    def _style_scrollbars(self):
        for sf in [self.chat]:
            try:
                sb = sf._scrollbar
                sb.configure(width=4, corner_radius=999)
                sb.configure(
                    button_color="transparent",
                    button_hover_color="transparent",
                    fg_color="transparent",
                    scrollbar_color=self.colors["scroll_thumb"],
                    scrollbar_hover_color=self.colors["scroll_hover"],
                )
            except Exception:
                pass
        # Completely hide the sessions scrollbar so no vertical line appears
        try:
            sb = self.sessions_frame._scrollbar
            sb.configure(
                width=0,
                button_color="transparent",
                button_hover_color="transparent",
                fg_color="transparent",
                scrollbar_color="transparent",
                scrollbar_hover_color="transparent",
            )
            sb.grid_remove()
        except Exception:
            pass
        # Also hide any internal border on the sessions scrollable frame
        try:
            self.sessions_frame._parent_frame.configure(border_width=0)
        except Exception:
            pass

    # ---------- sidebar collapse ----------
    def _toggle_sidebar(self):
        self._sidebar_collapsed = not self._sidebar_collapsed
        if self._sidebar_collapsed:
            self.sidebar.configure(width=56)
            self.sidebar_content.grid_remove()
            self.sidebar_toggle.configure(text="❮")
        else:
            self.sidebar.configure(width=SIDEBAR_W)
            self.sidebar_content.grid()
            self.sidebar_toggle.configure(text="❯")

    @staticmethod
    def _shade(hex_color: str, amount: float) -> str:
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        r = int(min(255, r + (255 - r) * amount))
        g = int(min(255, g + (255 - g) * amount))
        b = int(min(255, b + (255 - b) * amount))
        return f"#{r:02x}{g:02x}{b:02x}"

    # ---------- sidebar / sessions ----------
    def _load_sidebar_sessions(self):
        for w in self.sessions_frame.winfo_children():
            w.destroy()
        prune_sessions(self.settings.history_keep_sessions)

        sessions = list_sessions()[: self.settings.history_keep_sessions]
        self._session_buttons = {}

        if not sessions:
            ctk.CTkLabel(self.sessions_frame, text="No conversations yet", text_color=self.colors["muted"]).grid(
                row=0, column=0, padx=12, pady=12, sticky="w"
            )
            return

        for i, info in enumerate(sessions):
            rowf = ctk.CTkFrame(self.sessions_frame, corner_radius=6, fg_color="transparent")
            rowf.grid(row=i, column=0, sticky="ew", padx=0, pady=1)
            rowf.grid_columnconfigure(0, weight=1)

            b = ctk.CTkButton(
                rowf,
                text=info.title,
                anchor="w",
                height=30,
                corner_radius=6,
                fg_color="transparent",
                hover_color=self._shade(self.colors["bg"], 0.08),
                font=ctk.CTkFont(size=13),
                command=lambda s=info.session_id: self._open_session(s),
            )
            b.grid(row=0, column=0, sticky="ew")

            del_btn = ctk.CTkButton(
                rowf,
                text="✕",
                width=24,
                height=24,
                corner_radius=6,
                fg_color="transparent",
                hover_color=self._shade(self.colors["bg"], 0.10),
                command=lambda s=info.session_id: self._delete_session(s),
            )
            del_btn.grid(row=0, column=1, padx=(0, 0), sticky="e")
            try:
                del_btn.configure(text_color=self.colors["muted"])
            except Exception:
                pass

            self._session_buttons[info.session_id] = b

        self._refresh_active_chat_style()

    def _refresh_active_chat_style(self):
        for sid, btn in self._session_buttons.items():
            try:
                if sid == self._active_session_id:
                    btn.configure(
                        border_width=0,
                        fg_color=self.colors["session_active"],
                        text_color=self.colors["text"],
                    )
                else:
                    btn.configure(border_width=0, fg_color="transparent", text_color=self.colors["text"])
            except Exception:
                pass

    def _delete_session(self, session_id: str):
        try:
            p = session_path(session_id)
            if p.exists():
                p.unlink()
        except Exception:
            pass

        if session_id == self._active_session_id:
            self.session_title = "New chat"
            self.session_id = new_session_id(self.session_title)
            self._active_session_id = self.session_id
            self.messages = []
            self._clear_chat_ui()
            self._refresh_header_title()

        self._load_sidebar_sessions()

    def _open_session(self, session_id: str):
        self._persist_session()
        self.session_id = session_id
        self.session_title = load_session_title(session_id)
        self._active_session_id = session_id

        self.messages = []
        self._clear_chat_ui()
        loaded = load_session(session_id)
        kept = loaded[-self.settings.history_keep_messages :]
        for m in kept:
            self._add_message(role=m.role, content=m.content, mode=m.mode, ts=m.ts, persist=False)
            self.messages.append(UiMessage(role=m.role, content=m.content, mode=m.mode, ts=m.ts))
        self._refresh_header_title()
        self._refresh_active_chat_style()

    def _new_chat(self):
        self._persist_session()
        self.session_title = "New chat"
        self.session_id = new_session_id(self.session_title)
        self._active_session_id = self.session_id
        self.messages = []
        self._clear_chat_ui()
        self._load_sidebar_sessions()
        self._refresh_header_title()

    def _persist_session(self):
        if not self.messages:
            return
        msgs = [ChatMessage(role=m.role, content=m.content, mode=m.mode, ts=m.ts) for m in self.messages]
        save_session(self.session_id, self.session_title, msgs)
        prune_sessions(self.settings.history_keep_sessions)

    # ---------- header ----------
    def _refresh_header_title(self):
        # Session title is shown inside messages pane
        try:
            self.chat_session_title.configure(text=self.session_title if self.session_title else "New chat")
        except Exception:
            pass

    def _clear_chat(self):
        self.messages = []
        self._clear_chat_ui()
        self._persist_session()
        self._refresh_header_title()

    # ---------- input behavior ----------
    def _set_placeholder(self, refresh_only: bool = False):
        # Deprecated: placeholder is an overlay label now.
        self._update_placeholder_visibility()

    def _focus_input(self):
        self._text.focus_set()

    def _on_focus_in(self, _e=None):
        self._hide_placeholder()
        return

    def _on_focus_out(self, _e=None):
        txt = self.input.get("1.0", "end-1c")
        if not txt.strip():
            self._show_placeholder()
        return

    def _on_input_key(self, _e=None):
        self._hide_placeholder()
        self._update_send_enabled()

    def _show_placeholder(self):
        placeholder = "Type a message…"
        if self._mode == "Doc Q&A":
            placeholder = "Ask about your document…"
        try:
            self.placeholder_label.configure(text=placeholder)
            bbox = self._text.bbox("1.0")
            if bbox:
                x, y = bbox[0], bbox[1]
            else:
                x, y = 2, 2
            self.placeholder_label.place(in_=self._text, x=x, y=y)
        except Exception:
            pass

    def _hide_placeholder(self):
        try:
            self.placeholder_label.place_forget()
        except Exception:
            pass

    def _update_placeholder_visibility(self):
        txt = self.input.get("1.0", "end-1c")
        focused = (self._text.focus_get() == self._text)
        if focused or txt.strip():
            self._hide_placeholder()
        else:
            self._show_placeholder()

    def _on_enter_send(self, _e=None):
        if self._placeholder_active:
            return "break"
        self._on_send()
        return "break"

    def _on_shift_enter(self, _e=None):
        return None

    def _set_history_limit(self):
        dialog = ctk.CTkInputDialog(
            text=f"Current limit: {self.settings.history_keep_messages}\n"
                 "Keep how many recent prompts visible in each chat?",
            title="Prompt limit")
        val = dialog.get_input()
        try:
            n = int(val)
            n = max(1, min(50, n))
            self.settings.history_keep_messages = n
            save_settings(self.settings)
            self.history_limit_btn.configure(text=f"Prompt limit: {n}")
        except Exception:
            return

    def _toggle_theme(self):
        # Theme toggle kept in UI, but app is intentionally light-only.
        self.settings.theme = "light"
        save_settings(self.settings)
        self._apply_theme()

    def _set_font_family(self, value: str):
        self.settings.font_family = value
        save_settings(self.settings)
        self._apply_font_to_ui()

    def _set_font_size(self, value: str):
        try:
            self.settings.font_size = int(value)
            save_settings(self.settings)
        except Exception:
            return
        self._apply_font_to_ui()

    def _apply_font_to_ui(self):
        _f = (self.settings.font_family, self.settings.font_size)
        self._text.configure(font=_f)
        try:
            self.placeholder_label.configure(font=_f)
        except Exception:
            pass
        try:
            self.font_menu.set(self.settings.font_family)
            self.font_size_menu.set(str(self.settings.font_size))
        except Exception:
            pass
        self._rerender_messages()
        self._update_placeholder_visibility()

    def _set_mode(self, mode: str):
        self._mode = mode
        if mode == "Doc Q&A":
            self.upload_btn.grid()
        else:
            self.upload_btn.grid_remove()
        self._refresh_docs_sidebar()
        self._set_placeholder(refresh_only=False)
        for m, b in self.mode_buttons.items():
            if m == mode:
                b.configure(fg_color=self.colors["pill_active"],
                            border_width=1,
                            border_color=self.colors["pill_border"])
            else:
                b.configure(fg_color="transparent", border_width=0)
            b.configure(text_color=self.colors["header_text"])
        self._update_send_enabled()

    def _on_mode_changed(self):
        # legacy (not used)
        pass
        self._auto_grow_input()

    # ---------- chat rendering ----------
    def _max_bubble_px(self) -> int:
        """Max user bubble wraplength — based on chat_inner width minus padding."""
        try:
            w = int(self.chat_inner.winfo_width())
            if w > 200:
                return int(w * 0.70)
        except Exception:
            pass
        return 550

    def _fit_text_height(self, t: tk.Text) -> None:
        """Shrink a full-width tk.Text to exactly fit its content."""
        t.update_idletasks()
        try:
            info = t.count("1.0", "end-1c", "displaylines")
            if info is not None:
                n = info[0] if isinstance(info, (tuple, list)) else info
                if isinstance(n, int) and n > 0:
                    t.configure(height=n + 1)
                    return
        except Exception:
            pass
        content = t.get("1.0", "end-1c")
        t.configure(height=max(1, content.count("\n") + 1))

    def _clear_chat_ui(self):
        for w in self.chat_inner.winfo_children():
            w.destroy()
        self.chat_session_title = ctk.CTkLabel(
            self.chat_inner, text=self.session_title or "New chat",
            text_color=self.colors["text"],
            font=ctk.CTkFont(size=20, weight="bold"))
        self.chat_session_title.grid(row=0, column=0, sticky="w", pady=(4, 10))
        self.empty_state = ctk.CTkFrame(self.chat_inner, corner_radius=0, fg_color="transparent")
        self.empty_state.grid(row=1, column=0, pady=60)
        ctk.CTkLabel(self.empty_state, text="Start a conversation",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(0, 6))
        ctk.CTkLabel(self.empty_state, text="Ask questions about your documents or research papers",
                     text_color=self.colors["muted"]).pack()

    def _add_message(self, *, role: str, content: str, mode: str, ts: float, persist: bool):
        if hasattr(self, "empty_state") and self.empty_state.winfo_exists():
            self.empty_state.destroy()

        row = len(self.chat_inner.winfo_children())
        _tkfont = (self.settings.font_family, self.settings.font_size)

        if role == "user":
            wrapper = ctk.CTkFrame(self.chat_inner, corner_radius=0, fg_color="transparent")
            wrapper.grid(row=row, column=0, sticky="e", pady=(2, 10))
            wrapper.grid_columnconfigure(0, weight=1)

            bubble = ctk.CTkFrame(wrapper, corner_radius=18,
                                  fg_color=self.colors["user_bg"])
            bubble.grid(row=0, column=0, sticky="e", padx=(60, 0))

            lbl = ctk.CTkLabel(
                bubble, text=content, text_color="#FFFFFF",
                fg_color="transparent",
                font=ctk.CTkFont(family=self.settings.font_family,
                                 size=self.settings.font_size),
                wraplength=self._max_bubble_px() - 32,
                justify="left", anchor="w")
            lbl.grid(row=0, column=0, padx=18, pady=10)

            actions = ctk.CTkFrame(wrapper, corner_radius=0, fg_color="transparent")
            actions.grid(row=1, column=0, padx=0, pady=(3, 0), sticky="e")
            _ucopy = ctk.CTkButton(
                actions, text="📋", width=32, height=28, corner_radius=6,
                fg_color="transparent", text_color=self.colors["muted"],
                hover_color=self.colors["border"],
                font=ctk.CTkFont(size=14))
            _ucopy.configure(command=lambda c=content, b=_ucopy: self._copy(c, b))
            _ucopy.grid(row=0, column=0)

        else:
            wrapper = ctk.CTkFrame(self.chat_inner, corner_radius=0,
                                   fg_color="transparent")
            wrapper.grid(row=row, column=0, sticky="ew", padx=(0, 60), pady=(2, 10))
            wrapper.grid_columnconfigure(0, weight=1)

            t = tk.Text(wrapper, wrap="word", bd=0, highlightthickness=0,
                        bg=self.colors["chat_panel"], fg=self.colors["text"],
                        font=_tkfont, padx=4, pady=0, height=1, cursor="arrow",
                        spacing1=2, spacing3=2)
            t.grid(row=0, column=0, sticky="ew")
            configure_markdown_tags(
                t, base_fg=self.colors["text"],
                muted_fg=self.colors["muted"], accent=self.colors["accent"],
                code_bg=self.colors["code_bg"],
                font_family=self.settings.font_family,
                font_size=self.settings.font_size)
            render_markdown(t, content)
            self._fit_text_height(t)
            t.configure(state="disabled")

            actions = ctk.CTkFrame(wrapper, corner_radius=0, fg_color="transparent")
            actions.grid(row=1, column=0, padx=0, pady=(3, 0), sticky="w")
            _acopy = ctk.CTkButton(
                actions, text="📋", width=32, height=28, corner_radius=6,
                fg_color="transparent", text_color=self.colors["muted"],
                hover_color=self.colors["border"],
                font=ctk.CTkFont(size=14))
            _acopy.configure(command=lambda c=content, b=_acopy: self._copy(c, b))
            _acopy.grid(row=0, column=0)

        if persist:
            self.messages.append(UiMessage(role=role, content=content, mode=mode, ts=ts))
            if len(self.messages) > self.settings.history_keep_messages:
                self.messages = self.messages[-self.settings.history_keep_messages:]
                current = list(self.messages)
                self._clear_chat_ui()
                for m in current:
                    self._add_message(role=m.role, content=m.content, mode=m.mode,
                                      ts=m.ts, persist=False)
                self.messages = current

        self._scroll_to_bottom()

    def _copy(self, text: str, btn: ctk.CTkButton = None):
        self.clipboard_clear()
        self.clipboard_append(text)
        if btn is not None:
            try:
                btn.configure(text="Copied", font=ctk.CTkFont(size=11))
                self.after(1200, lambda: btn.configure(
                    text="📋", font=ctk.CTkFont(size=14)))
            except Exception:
                pass

    def _scroll_to_bottom(self):
        try:
            self.chat._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    # ---------- send / tasks ----------
    def _on_send(self):
        if self.send_btn.cget("state") == "disabled":
            return
        text = self.input.get("1.0", "end-1c").strip()
        if not text:
            return

        mode = self._mode

        if mode == "Email":
            to_d = ctk.CTkInputDialog(text="Recipient (To):", title="Email details")
            to_v = (to_d.get_input() or "").strip()
            if not to_v:
                return
            fr_d = ctk.CTkInputDialog(text="Sender (From):", title="Email details")
            fr_v = (fr_d.get_input() or "").strip()
            if not fr_v:
                return
            sb_d = ctk.CTkInputDialog(text="Subject (optional):", title="Email details")
            sb_v = (sb_d.get_input() or "").strip() or "<subject>"
            text = f"To: {to_v}\nFrom: {fr_v}\nSubject: {sb_v}\n\n{text}"

        if self.session_title == "New chat":
            self.session_title = derive_title_from_text(text)
            self.session_id = new_session_id(self.session_title)
            self._active_session_id = self.session_id
            self._load_sidebar_sessions()
            self._refresh_header_title()

        self._add_message(role="user", content=text, mode=mode, ts=time.time(), persist=True)
        self.input.delete("1.0", "end")
        self._update_placeholder_visibility()

        # Pending assistant message — lock mode at send time
        tid = next_task_id()
        self._pending_task_id = tid
        self._pending_mode = mode
        self._pending_widget = self._add_pending_message()
        self._update_send_enabled()

        if mode == "Doc Q&A":
            if not self._active_doc_id:
                self._pending_task_id = None
                render_markdown(self._pending_widget, "Upload a document first.")
                self._pending_widget = None
                return

            ranked = []
            filtered = []
            try:
                from RAG.embedder import embed_texts
                from RAG.vector_store import ChromaPerDocStore

                q_emb = embed_texts([text])[0]
                store = ChromaPerDocStore()
                all_hits = []
                search_errors = []
                for did in self._doc_ids:
                    try:
                        hits = store.search(doc_id=did, query_embedding=q_emb, k=10)
                        all_hits.extend(hits)

                    except Exception as se:
                        search_errors.append(f"{did}: {se}")
                if search_errors:
                    self._show_toast(
                        f"Search issues: {'; '.join(search_errors[:3])}",
                        duration_ms=6000)
                
                filtered_hits, has_enough = filter_rag_contexts(all_hits)
                if not has_enough:
                    self._pending_task_id = None
                    render_markdown(
                        self._pending_widget,
                        "I couldn't find enough relevant content in the uploaded "
                        "documents to answer your question. Please try rephrasing "
                        "or ensure the relevant document is uploaded.")
                    self._pending_widget = None
                    self._update_send_enabled()
                    return
                chunks = []
                seen_sources = {}
                
                for idx, h in enumerate(filtered_hits):
                    txt = h.get("text", "").strip()
                    if not txt:
                        continue
                    
                    
                    src = h.get("source", "document")
                    pg = h.get("page", "?")
                    chunks.append(
                        f"--- Chunk {idx+1} ---\n"
                        f"Document: {src}\n"
                        f"Page: {pg}\n"
                        f"Content: {txt}")
                    
                    if src not in seen_sources:
                        seen_sources[src] = set()
                    seen_sources[src].add(str(pg))
                context = "\n\n".join(chunks)
                
                parts = []
                for s, pages in seen_sources.items():
                    sorted_pg = sorted(pages, key=lambda x: (int(x) if x.isdigit() else 999))
                    parts.append(f"{s} (p. {', '.join(sorted_pg)})")
                self._pending_rag_sources = " | ".join(parts) if parts else None
            except Exception as exc:
                context = ""
                self._show_toast(f"RAG search error: {exc}", duration_ms=6000)

            if not context.strip():
                self._pending_task_id = None
                render_markdown(
                    self._pending_widget,
                    "I couldn't find enough relevant content in the uploaded "
                    "documents to answer your question. Please try rephrasing "
                    "or ensure the relevant document is uploaded.")
                self._pending_widget = None
                self._update_send_enabled()
                return

            llm_in.put(LlmTask(kind="rag", text=text, mode=mode, context=context, task_id=tid))
        else:
            self._pending_rag_sources = None
            llm_in.put(LlmTask(kind="writing", text=text, mode=mode, task_id=tid))

    def _add_pending_message(self) -> tk.Text:
        row = len(self.chat_inner.winfo_children())
        _tkfont = (self.settings.font_family, self.settings.font_size)

        wrapper = ctk.CTkFrame(self.chat_inner, corner_radius=0, fg_color="transparent")
        wrapper.grid(row=row, column=0, sticky="ew", padx=(0, 60), pady=(2, 10))
        wrapper.grid_columnconfigure(0, weight=1)
        self._pending_wrapper = wrapper

        t = tk.Text(wrapper, wrap="word", bd=0, highlightthickness=0,
                    bg=self.colors["chat_panel"], fg=self.colors["text"],
                    font=_tkfont, padx=4, pady=0, height=1, cursor="arrow")
        t.grid(row=0, column=0, sticky="ew")
        configure_markdown_tags(
            t, base_fg=self.colors["text"],
            muted_fg=self.colors["muted"], accent=self.colors["accent"],
            code_bg=self.colors["code_bg"],
            font_family=self.settings.font_family,
            font_size=self.settings.font_size)
        render_markdown(t, "Thinking...")
        self._fit_text_height(t)
        self._animate_pending(t, phase=0)
        self._scroll_to_bottom()
        return t

    def _animate_pending(self, widget: tk.Text, *, phase: int):
        if self._pending_task_id is None or widget is None or not widget.winfo_exists():
            return
        dots = ["", ".", "..", "..."][phase % 4]
        render_markdown(widget, f"*Thinking{dots}*")
        self._fit_text_height(widget)
        self.after(300, lambda: self._animate_pending(widget, phase=phase + 1))

    # ---------- background workers polling ----------
    def _poll_llm_results(self):
        try:
            while True:
                r = llm_out.get_nowait()
                if self._pending_task_id is None or r.task_id != self._pending_task_id:
                    continue
                self._pending_task_id = None
                locked_mode = self._pending_mode or self._mode
                self._pending_mode = None
                self._refresh_header_title()

                text = r.text if r.ok else f"Error: {r.error}"
                if self._pending_rag_sources:
                    text = text.rstrip() + f"\n\n*Sources: {self._pending_rag_sources}*"
                    self._pending_rag_sources = None
                if self._pending_widget is not None and self._pending_widget.winfo_exists():
                    render_markdown(self._pending_widget, text)
                    self._fit_text_height(self._pending_widget)
                    self._pending_widget.configure(state="disabled")
                    if hasattr(self, "_pending_wrapper") and self._pending_wrapper and self._pending_wrapper.winfo_exists():
                        actions = ctk.CTkFrame(self._pending_wrapper, corner_radius=0,
                                               fg_color="transparent")
                        actions.grid(row=1, column=0, padx=0, pady=(3, 0), sticky="w")
                        _pcopy = ctk.CTkButton(
                            actions, text="📋", width=32, height=28, corner_radius=6,
                            fg_color="transparent", text_color=self.colors["muted"],
                            hover_color=self.colors["border"],
                            font=ctk.CTkFont(size=14))
                        _pcopy.configure(command=lambda c=text, b=_pcopy: self._copy(c, b))
                        _pcopy.grid(row=0, column=0)
                    self.messages.append(UiMessage(role="assistant", content=text,
                                                   mode=locked_mode, ts=time.time()))
                    self._pending_widget = None
                    self._pending_wrapper = None
                else:
                    self._add_message(role="assistant", content=text, mode=locked_mode,
                                      ts=time.time(), persist=True)

                self._persist_session()
                self._update_send_enabled()
        except Exception:
            pass
        self.after(120, self._poll_llm_results)

    def _on_text_modified(self, _evt=None):
        try:
            self._text.edit_modified(False)
        except Exception:
            pass
        if self._placeholder_active:
            return
        if self._debounce_after_id is not None:
            try:
                self.after_cancel(self._debounce_after_id)
            except Exception:
                pass
        self._debounce_after_id = self.after(450, self._request_spell_check)
        self._auto_grow_input()
        self._update_send_enabled()

    def _auto_grow_input(self):
        try:
            line_count = int(self._text.index("end-1c").split(".")[0])
        except Exception:
            line_count = 1
        target = 56 + max(0, line_count - 1) * 22
        target = max(56, min(200, target))
        self.input.configure(height=target)

    def _request_spell_check(self):
        txt = self.input.get("1.0", "end-1c")
        req_id = time.time()
        self._last_spell_req_id = req_id
        spell_in.put({"id": req_id, "text": txt})

    def _poll_spell_results(self):
        try:
            while True:
                r = spell_out.get_nowait()
                if r.get("id", 0.0) != self._last_spell_req_id:
                    continue
                self._apply_highlights(r.get("errors", []))
        except Exception:
            pass
        self.after(160, self._poll_spell_results)

    def _apply_highlights(self, errors: List[SpanError]):
        self._text.tag_remove("spell_error", "1.0", "end")
        self._text.tag_remove("grammar_error", "1.0", "end")
        for e in errors:
            start = f"1.0+{e.start}c"
            end = f"1.0+{e.end}c"
            tag = "spell_error" if e.type == "spell" else "grammar_error"
            self._text.tag_add(tag, start, end)

    def _upload_files(self):
        if self._mode != "Doc Q&A":
            return
        paths = filedialog.askopenfilenames(
            title="Select documents",
            filetypes=[("Documents", "*.pdf;*.docx;*.doc"), ("PDF files", "*.pdf"),
                       ("Word", "*.docx;*.doc")],
        )
        if not paths:
            return
        self._indexing_inflight += len(paths)
        names = ", ".join(os.path.basename(p) for p in paths)
        self._index_toast = self._show_toast(f"Indexing {len(paths)} file(s): {names}")
        self._start_upload_animation()
        self._update_send_enabled()
        for p in paths:
            index_in.put({"path": p})

    def _poll_index_results(self):
        try:
            while True:
                r = index_out.get_nowait()
                if not r.get("ok"):
                    err = r.get("error", "Unknown error")
                    self._show_toast(f"Indexing failed: {err}", duration_ms=8000)
                    self._indexing_inflight = max(0, self._indexing_inflight - 1)
                    if self._indexing_inflight == 0:
                        self._stop_upload_animation()
                        self._dismiss_index_toast()
                    self._update_send_enabled()
                    continue
                doc_id = r.get("doc_id")
                if doc_id and doc_id not in self._doc_ids:
                    self._doc_ids.append(doc_id)
                self._active_doc_id = doc_id
                saved_path = r.get("saved_path") or ""
                display_name = os.path.basename(saved_path) if saved_path else doc_id
                self._upsert_doc_item(doc_id, display_name)
                self._indexing_inflight = max(0, self._indexing_inflight - 1)
                if self._indexing_inflight == 0:
                    self._stop_upload_animation()
                    self._dismiss_index_toast()
                    self._show_toast(f"Ready: {display_name}", duration_ms=4000)
                else:
                    if hasattr(self, "_index_toast") and self._index_toast:
                        self._update_toast(self._index_toast,
                                           f"Indexed {display_name}. {self._indexing_inflight} remaining…")
                self._update_send_enabled()
        except Exception:
            pass
        self.after(250, self._poll_index_results)

    def _dismiss_index_toast(self):
        if hasattr(self, "_index_toast") and self._index_toast:
            self._dismiss_toast(self._index_toast)
            self._index_toast = None

    def _refresh_docs_sidebar(self):
        """Show/hide the docs heading + list based on whether any docs exist.
        Always visible when docs exist, regardless of active mode."""
        if self._doc_items:
            self.docs_label.grid()
            self.docs_frame.grid()
        else:
            self.docs_label.grid_remove()
            self.docs_frame.grid_remove()

    def _rebuild_docs_list(self):
        """Re-render all doc items in the sidebar frame."""
        for w in self.docs_frame.winfo_children():
            w.destroy()
        for i, (doc_id, info) in enumerate(self._doc_items.items()):
            rowf = ctk.CTkFrame(self.docs_frame, corner_radius=6, fg_color="transparent")
            rowf.grid(row=i, column=0, sticky="ew", padx=0, pady=1)
            rowf.grid_columnconfigure(0, weight=1)

            btn = ctk.CTkButton(
                rowf, text=info["name"], anchor="w", height=28, corner_radius=6,
                fg_color="transparent",
                hover_color=self._shade(self.colors["bg"], 0.08),
                text_color=self.colors["text"],
                font=ctk.CTkFont(size=13),
                command=lambda d=doc_id: self._select_doc(d))
            btn.grid(row=0, column=0, sticky="ew")
            info["button"] = btn

            rm_btn = ctk.CTkButton(
                rowf, text="✕", width=24, height=24, corner_radius=6,
                fg_color="transparent", text_color=self.colors["muted"],
                hover_color=self._shade(self.colors["bg"], 0.10),
                command=lambda d=doc_id: self._remove_doc(d))
            rm_btn.grid(row=0, column=1, padx=(0, 0))

        self._highlight_active_doc()
        self._refresh_docs_sidebar()

    def _upsert_doc_item(self, doc_id: str, name: str):
        if doc_id in self._doc_items:
            self._doc_items[doc_id]["name"] = name
        else:
            self._doc_items[doc_id] = {"name": name, "button": None}
        self._persist_doc_registry()
        self._rebuild_docs_list()

    def _persist_doc_registry(self):
        registry = {did: info["name"] for did, info in self._doc_items.items()}
        save_doc_registry(registry)

    def _remove_doc(self, doc_id: str):
        try:
            from RAG.vector_store import ChromaPerDocStore
            ChromaPerDocStore().delete_collection(doc_id)
        except Exception:
            pass
        if doc_id in self._doc_items:
            del self._doc_items[doc_id]
        if doc_id in self._doc_ids:
            self._doc_ids.remove(doc_id)
        if self._active_doc_id == doc_id:
            self._active_doc_id = self._doc_ids[0] if self._doc_ids else None
        self._persist_doc_registry()
        self._rebuild_docs_list()

    def _select_doc(self, doc_id: str):
        self._active_doc_id = doc_id
        self._highlight_active_doc()

    def _highlight_active_doc(self):
        for did, info in self._doc_items.items():
            btn = info.get("button")
            if btn is None:
                continue
            try:
                if did == self._active_doc_id:
                    btn.configure(border_width=2, border_color=self.colors["accent"],
                                  fg_color=self._shade(self.colors["bg"], 0.06))
                else:
                    btn.configure(border_width=0, fg_color="transparent")
            except Exception:
                pass

    def _start_upload_animation(self):
        if self._indexing_inflight <= 0:
            return
        self._upload_anim_phase = 0
        self._tick_upload_animation()

    def _tick_upload_animation(self):
        if self._indexing_inflight <= 0:
            return
        frames = ["⏳", "⌛"]
        self.upload_btn.configure(text=frames[self._upload_anim_phase % len(frames)])
        self._upload_anim_phase += 1
        self.after(350, self._tick_upload_animation)

    def _stop_upload_animation(self):
        self.upload_btn.configure(text="📎", font=ctk.CTkFont(size=20))

    def _update_send_enabled(self):
        busy = self._pending_task_id is not None
        indexing = self._mode in {"Doc Q&A",} and self._indexing_inflight > 0
        empty = not self.input.get("1.0", "end-1c").strip()

        if busy:
            self.send_btn.configure(state="disabled")
            if not self._send_spinning:
                self._send_spinning = True
                self._send_spin_phase = 0
                self._tick_send_spinner()
        elif indexing or empty:
            self._stop_send_spinner()
            self.send_btn.configure(
                state="disabled",
                fg_color=self.colors["send_disabled"],
                hover_color=self.colors["send_disabled"],
                text_color="#888888")
        else:
            self._stop_send_spinner()
            self.send_btn.configure(
                state="normal", fg_color=self.colors["send"],
                hover_color=self.colors["send_hover"],
                text_color="#FFFFFF")

    def _tick_send_spinner(self):
        if not self._send_spinning:
            return
        if not hasattr(self, "_spin_canvas") or self._spin_canvas is None:
            self.send_btn.configure(text="")
            c = tk.Canvas(self.send_btn, width=22, height=22,
                          bg=self.colors["send"], highlightthickness=0, bd=0)
            c.place(relx=0.5, rely=0.5, anchor="center")
            self._spin_canvas = c

        canvas = self._spin_canvas
        canvas.delete("all")
        start_angle = (self._send_spin_phase * 30) % 360
        canvas.create_arc(3, 3, 19, 19, start=start_angle, extent=270,
                          outline="#FFFFFF", width=2, style="arc")
        self._send_spin_phase += 1
        self.after(50, self._tick_send_spinner)

    def _stop_send_spinner(self):
        if self._send_spinning:
            self._send_spinning = False
            if hasattr(self, "_spin_canvas") and self._spin_canvas is not None:
                self._spin_canvas.destroy()
                self._spin_canvas = None
            self.send_btn.configure(
                text="➤",
                font=ctk.CTkFont(size=18),
                text_color="#FFFFFF")

    def _rerender_messages(self):
        current = list(self.messages)
        self._clear_chat_ui()
        for m in current:
            self._add_message(role=m.role, content=m.content, mode=m.mode, ts=m.ts, persist=False)
        self.messages = current

    # ---------- toast notifications (bottom-right) ----------
    def _show_toast(self, message: str, duration_ms: int = 0) -> ctk.CTkFrame:
        """Show a toast notification anchored to bottom-right of main panel.
        If duration_ms > 0, auto-dismiss after that time.
        Returns the toast frame (caller can keep a reference)."""
        toast = ctk.CTkFrame(self.main, corner_radius=10,
                             fg_color="#1a1a1a", border_width=1,
                             border_color="#333333")
        toast.place(relx=1.0, rely=1.0, anchor="se", x=-16, y=-16)

        inner = ctk.CTkFrame(toast, corner_radius=0, fg_color="transparent")
        inner.pack(padx=12, pady=10, fill="x")
        inner.grid_columnconfigure(0, weight=1)

        lbl = ctk.CTkLabel(inner, text=message, text_color="#FFFFFF",
                           font=ctk.CTkFont(size=13), anchor="w",
                           wraplength=300, justify="left")
        lbl.grid(row=0, column=0, sticky="w", padx=(0, 8))

        close_btn = ctk.CTkButton(
            inner, text="✕", width=24, height=24, corner_radius=6,
            fg_color="transparent", text_color="#999999",
            hover_color="#333333",
            command=lambda: self._dismiss_toast(toast))
        close_btn.grid(row=0, column=1, sticky="ne")

        toast._label = lbl

        if duration_ms > 0:
            toast._auto_id = self.after(duration_ms, lambda: self._dismiss_toast(toast))

        return toast

    def _update_toast(self, toast: ctk.CTkFrame, message: str):
        try:
            if toast.winfo_exists():
                toast._label.configure(text=message)
        except Exception:
            pass

    def _dismiss_toast(self, toast: ctk.CTkFrame):
        try:
            if hasattr(toast, "_auto_id"):
                self.after_cancel(toast._auto_id)
        except Exception:
            pass
        try:
            toast.destroy()
        except Exception:
            pass

    # ---------- splash ----------
    def _show_splash(self):
        splash = ctk.CTkToplevel(self)
        splash.overrideredirect(True)
        splash.attributes("-topmost", True)
        w, h = 420, 220
        x = max(0, int(self.winfo_screenwidth() / 2 - w / 2))
        y = max(0, int(self.winfo_screenheight() / 2 - h / 2))
        splash.geometry(f"{w}x{h}+{x}+{y}")
        splash.configure(fg_color="#202123")

        logo = ctk.CTkLabel(splash, text="grammar-ish", font=ctk.CTkFont(size=26, weight="bold"), text_color="#ECECF1")
        logo.pack(pady=(42, 10))
        tag = ctk.CTkLabel(splash, text="Don’t worry, I got it.", font=ctk.CTkFont(size=14), text_color="#A9A9B3")
        tag.pack(pady=(0, 18))
        mini = ctk.CTkLabel(splash, text="✦", font=ctk.CTkFont(size=18), text_color="#A9A9B3")
        mini.pack()
        self.after(900, splash.destroy)

    # ---------- tooltips ----------
    def _attach_tooltips(self):
        Tooltip.attach(self.new_chat_btn, "New chat")
        Tooltip.attach(self.history_limit_btn, "History limit")
        Tooltip.attach(self.clear_btn, "Clear chat")
        Tooltip.attach(self.theme_toggle, "Toggle theme")
        Tooltip.attach(self.font_menu, "Font family")
        Tooltip.attach(self.font_size_menu, "Font size")
        Tooltip.attach(self.send_btn, "Send")
        Tooltip.attach(self.upload_btn, "Upload document")

    # ---------- window init ----------
    def _post_init_window(self):
        """Maximize and apply dark title bar after window is fully created."""
        try:
            self.state("zoomed")
        except Exception:
            pass
        self._set_dark_titlebar()

    def _set_dark_titlebar(self):
        """Force the Windows title bar to dark/black."""
        try:
            import ctypes
            import ctypes.wintypes
            self.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            if not hwnd:
                hwnd = self.winfo_id()
            value = ctypes.c_int(1)
            # DWMWA_USE_IMMERSIVE_DARK_MODE (works on Windows 10 build 18985+ and Windows 11)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
            # Also set caption color to black (Windows 11)
            DWMWA_CAPTION_COLOR = 35
            black = ctypes.c_int(0x001A1A1A)  # COLORREF BGR
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_CAPTION_COLOR, ctypes.byref(black), ctypes.sizeof(black))
            # Also set text color to white (Windows 11)
            DWMWA_TEXT_COLOR = 36
            white = ctypes.c_int(0x00FFFFFF)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, DWMWA_TEXT_COLOR, ctypes.byref(white), ctypes.sizeof(white))
        except Exception:
            pass

    # ---------- misc ----------


class Tooltip:
    _current = None

    @staticmethod
    def attach(widget, text: str):
        def on_enter(_e=None):
            Tooltip.show(widget, text)

        def on_leave(_e=None):
            Tooltip.hide()

        try:
            widget.bind("<Enter>", on_enter)
            widget.bind("<Leave>", on_leave)
        except Exception:
            pass

    @staticmethod
    def show(widget, text: str):
        Tooltip.hide()
        try:
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() + widget.winfo_height() + 8
        except Exception:
            return
        # Attach to root to reduce random window manager issues
        try:
            root = widget.winfo_toplevel()
        except Exception:
            root = None
        tip = tk.Toplevel(root) if root is not None else tk.Toplevel()
        tip.wm_overrideredirect(True)
        tip.wm_attributes("-topmost", True)
        tip.configure(bg="#111827")
        lbl = tk.Label(tip, text=text, bg="#111827", fg="#F9FAFB", font=("Segoe UI", 10), padx=8, pady=4)
        lbl.pack()
        tip.geometry(f"+{x}+{y}")
        Tooltip._current = tip

    @staticmethod
    def hide():
        if Tooltip._current is not None:
            try:
                Tooltip._current.destroy()
            except Exception:
                pass
        Tooltip._current = None


if __name__ == "__main__":
    App().mainloop()
