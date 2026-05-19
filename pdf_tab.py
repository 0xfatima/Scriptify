from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import customtkinter as ctk
import tkinter as tk


@dataclass
class PdfOpenResult:
    ok: bool
    title: str = ""
    error: str = ""


class PdfTab(ctk.CTkFrame):
    """
    Simple in-app PDF viewer:
    - Renders the current page as an image (PyMuPDF)
    - Supports click-drag rectangle selection of text (mapped via word boxes)
    - Keeps the selection stored even while the user types a question
    """

    def __init__(self, master, *, colors: dict, font_family: str, font_size: int):
        super().__init__(master, corner_radius=0, fg_color="transparent")
        self._colors = colors
        self._font_family = font_family
        self._font_size = font_size

        self._pdf_path: Optional[Path] = None
        self._doc = None
        self._page_index = 0
        self._page_count = 0
        self._zoom = 1.6

        self._page_img = None
        self._page_words = []

        self._sel_rect_id = None
        self._sel_start = None
        self._selected_text = ""

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        top = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=40, pady=(10, 6))
        top.grid_columnconfigure(1, weight=1)

        self._title = ctk.CTkLabel(
            top,
            text="PDF",
            text_color=self._colors["text"],
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self._title.grid(row=0, column=0, sticky="w")

        nav = ctk.CTkFrame(top, corner_radius=0, fg_color="transparent")
        nav.grid(row=0, column=2, sticky="e")

        self._prev_btn = ctk.CTkButton(nav, text="◀", width=34, height=30, corner_radius=8, command=self.prev_page)
        self._prev_btn.grid(row=0, column=0, padx=(0, 6))
        self._page_lbl = ctk.CTkLabel(nav, text="0/0", text_color=self._colors["muted"])
        self._page_lbl.grid(row=0, column=1, padx=(0, 6))
        self._next_btn = ctk.CTkButton(nav, text="▶", width=34, height=30, corner_radius=8, command=self.next_page)
        self._next_btn.grid(row=0, column=2, padx=(0, 10))
        self._clear_sel_btn = ctk.CTkButton(nav, text="Clear selection", height=30, corner_radius=8, command=self.clear_selection)
        self._clear_sel_btn.grid(row=0, column=3)

        self._hint = ctk.CTkLabel(
            self,
            text="Drag to select text on the page, then ask a question below.",
            text_color=self._colors["muted"],
            font=ctk.CTkFont(size=12),
        )
        self._hint.grid(row=1, column=0, sticky="w", padx=40, pady=(0, 6))

        self._wrap = ctk.CTkFrame(
            self,
            corner_radius=12,
            fg_color=self._colors["chat_panel"],
            border_width=1,
            border_color=self._colors["border"],
        )
        self._wrap.grid(row=2, column=0, sticky="nsew", padx=40, pady=(0, 12))
        self._wrap.grid_rowconfigure(0, weight=1)
        self._wrap.grid_columnconfigure(0, weight=1)

        self._canvas = tk.Canvas(
            self._wrap,
            bg=self._colors["chat_panel"],
            highlightthickness=0,
            bd=0,
        )
        self._canvas.grid(row=0, column=0, sticky="nsew")

        self._vscroll = tk.Scrollbar(self._wrap, command=self._canvas.yview)
        self._vscroll.grid(row=0, column=1, sticky="ns")
        self._canvas.configure(yscrollcommand=self._vscroll.set)

        self._canvas.bind("<Button-1>", self._on_mouse_down)
        self._canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        self._sel_info = ctk.CTkLabel(self, text="", text_color=self._colors["muted"], font=ctk.CTkFont(size=12))
        self._sel_info.grid(row=3, column=0, sticky="w", padx=40, pady=(0, 10))

        self._set_placeholder()

    def _set_placeholder(self):
        self._title.configure(text="PDF")
        self._page_lbl.configure(text="0/0")
        self._canvas.delete("all")
        self._canvas.create_text(
            20,
            20,
            anchor="nw",
            fill=self._colors["muted"],
            text="Open a PDF from the Uploaded Documents list to view it here.",
            font=(self._font_family, self._font_size),
        )
        self._canvas.configure(scrollregion=self._canvas.bbox("all") or (0, 0, 1, 1))
        self.clear_selection()

    def set_font(self, *, font_family: str, font_size: int):
        self._font_family = font_family
        self._font_size = font_size

    def show_error(self, *, title: str, error: str):
        self._title.configure(text=title or "PDF")
        self._canvas.delete("all")
        self._canvas.create_text(
            20,
            20,
            anchor="nw",
            fill=self._colors["muted"],
            text=f"Could not open PDF.\n\n{error}",
            font=(self._font_family, self._font_size),
        )
        self._canvas.configure(scrollregion=self._canvas.bbox("all") or (0, 0, 1, 1))
        self.clear_selection()

    def open_pdf(self, *, path: str, title: str = "") -> PdfOpenResult:
        try:
            import fitz  # PyMuPDF

            self._pdf_path = Path(path)
            if not self._pdf_path.exists():
                raise RuntimeError("File not found.")
            self._doc = fitz.open(str(self._pdf_path))
            self._page_count = len(self._doc)
            self._page_index = 0
            self._title.configure(text=title or self._pdf_path.name)
            self._render_page()
            return PdfOpenResult(ok=True, title=title or self._pdf_path.name)
        except Exception as e:
            self.show_error(title=title or "PDF", error=str(e))
            return PdfOpenResult(ok=False, title=title or "PDF", error=str(e))

    def prev_page(self):
        if not self._doc or self._page_count <= 0:
            return
        if self._page_index <= 0:
            return
        self._page_index -= 1
        self._render_page(clear_selection=True)

    def next_page(self):
        if not self._doc or self._page_count <= 0:
            return
        if self._page_index >= self._page_count - 1:
            return
        self._page_index += 1
        self._render_page(clear_selection=True)

    def clear_selection(self):
        self._selected_text = ""
        self._sel_info.configure(text="")
        self._sel_start = None
        if self._sel_rect_id is not None:
            try:
                self._canvas.delete(self._sel_rect_id)
            except Exception:
                pass
        self._sel_rect_id = None

    def get_selected_text(self) -> str:
        return (self._selected_text or "").strip()

    def _render_page(self, *, clear_selection: bool = False):
        if clear_selection:
            self.clear_selection()
        if not self._doc or self._page_count <= 0:
            self._set_placeholder()
            return

        try:
            import fitz  # PyMuPDF

            page = self._doc.load_page(self._page_index)
            mat = fitz.Matrix(self._zoom, self._zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            ppm = pix.tobytes("ppm")

            self._page_img = tk.PhotoImage(data=ppm)
            self._canvas.delete("all")
            self._canvas.create_image(0, 0, anchor="nw", image=self._page_img)
            self._canvas.configure(scrollregion=(0, 0, self._page_img.width(), self._page_img.height()))

            self._page_words = page.get_text("words") or []
            self._page_lbl.configure(text=f"{self._page_index + 1}/{self._page_count}")
        except Exception as e:
            self.show_error(title=self._title.cget("text"), error=str(e))

    def _on_mouse_down(self, e):
        if not self._doc:
            return
        self._sel_start = (self._canvas.canvasx(e.x), self._canvas.canvasy(e.y))
        if self._sel_rect_id is not None:
            try:
                self._canvas.delete(self._sel_rect_id)
            except Exception:
                pass
        x, y = self._sel_start
        self._sel_rect_id = self._canvas.create_rectangle(x, y, x + 1, y + 1, outline="#333333", width=2)

    def _on_mouse_drag(self, e):
        if not self._doc or not self._sel_start or self._sel_rect_id is None:
            return
        x0, y0 = self._sel_start
        x1, y1 = (self._canvas.canvasx(e.x), self._canvas.canvasy(e.y))
        self._canvas.coords(self._sel_rect_id, x0, y0, x1, y1)

    def _on_mouse_up(self, e):
        if not self._doc or not self._sel_start or self._sel_rect_id is None:
            return
        x0, y0 = self._sel_start
        x1, y1 = (self._canvas.canvasx(e.x), self._canvas.canvasy(e.y))
        self._sel_start = None

        left, right = (x0, x1) if x0 <= x1 else (x1, x0)
        top, bottom = (y0, y1) if y0 <= y1 else (y1, y0)
        if abs(right - left) < 4 or abs(bottom - top) < 4:
            return

        # Map rectangle (image coords) to PDF coords by dividing by zoom.
        zl = left / self._zoom
        ზt = top / self._zoom
        zr = right / self._zoom
        zb = bottom / self._zoom

        chosen = []
        for w in self._page_words:
            # w = (x0, y0, x1, y1, "word", block_no, line_no, word_no)
            wx0, wy0, wx1, wy1 = w[0], w[1], w[2], w[3]
            if wx1 < zl or wx0 > zr or wy1 < ზt or wy0 > zb:
                continue
            chosen.append(w)

        if not chosen:
            self._selected_text = ""
            self._sel_info.configure(text="No text selected.")
            return

        # Sort by block/line/word for natural reading order
        chosen.sort(key=lambda w: (w[5], w[6], w[7]))
        out_parts = []
        last_key = None
        for w in chosen:
            key = (w[5], w[6])
            if last_key is not None and key != last_key:
                out_parts.append("\n")
            out_parts.append(w[4])
            out_parts.append(" ")
            last_key = key

        text = "".join(out_parts).strip()
        self._selected_text = text
        self._sel_info.configure(text=f"Selected {len(text)} characters (selection stays until cleared).")

