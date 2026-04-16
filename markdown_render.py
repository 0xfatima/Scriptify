from __future__ import annotations

import re
from typing import Optional


def configure_markdown_tags(
    text_widget,
    *,
    base_fg: str,
    muted_fg: str,
    accent: str,
    code_bg: str,
    font_family: str,
    font_size: int,
):
    # Basic tags
    base = max(10, int(font_size))
    text_widget.configure(font=(font_family, base))
    text_widget.tag_configure("md_bold", font=(font_family, base, "bold"), foreground=base_fg)
    text_widget.tag_configure("md_italic", font=(font_family, base, "italic"), foreground=base_fg)
    text_widget.tag_configure("md_code", font=("Consolas", max(9, base - 1)), background=code_bg, foreground=base_fg)
    text_widget.tag_configure("md_h", font=(font_family, base + 1, "bold"), foreground=base_fg)
    text_widget.tag_configure("md_link", foreground=accent, underline=True)
    text_widget.tag_configure("md_muted", foreground=muted_fg)


def render_markdown(text_widget, md: str):
    """
    Lightweight Markdown renderer into a tkinter.Text widget.
    Supports: headings (#), bullets (-), inline **bold**, *italic*, `code`, fenced ``` blocks, and links [t](url).
    """
    text_widget.configure(state="normal")
    text_widget.delete("1.0", "end")

    in_code_block = False

    lines = (md or "").splitlines()
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            _insert_with_tag(text_widget, line + "\n", "md_code")
            continue

        if line.startswith("#"):
            title = line.lstrip("#").strip()
            _insert_with_tag(text_widget, title + "\n", "md_h")
            continue

        if line.strip().startswith("- "):
            _insert_inline(text_widget, "• " + line.strip()[2:] + "\n")
            continue

        _insert_inline(text_widget, line + "\n")

    text_widget.configure(state="disabled")


def _insert_with_tag(tw, text: str, tag: str):
    start = tw.index("end-1c")
    tw.insert("end", text)
    end = tw.index("end-1c")
    tw.tag_add(tag, start, end)


def _insert_inline(tw, text: str):
    # Parse links first: [text](url)
    pos = 0
    for m in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)", text):
        _insert_emphasis_and_code(tw, text[pos : m.start()])
        label = m.group(1)
        url = m.group(2)
        start = tw.index("end-1c")
        tw.insert("end", label)
        end = tw.index("end-1c")
        tw.tag_add("md_link", start, end)
        tw.tag_bind("md_link", "<Button-1>", lambda _e, u=url: _open_url(u))
        pos = m.end()
    _insert_emphasis_and_code(tw, text[pos:])


def _insert_emphasis_and_code(tw, text: str):
    # Inline code `...`
    parts = re.split(r"(`[^`]+`)", text)
    for part in parts:
        if part.startswith("`") and part.endswith("`") and len(part) >= 2:
            _insert_with_tag(tw, part[1:-1], "md_code")
        else:
            _insert_bold_italic(tw, part)


def _insert_bold_italic(tw, text: str):
    # Bold **...** and italic *...* (simple, non-nested)
    i = 0
    while i < len(text):
        if text.startswith("**", i):
            j = text.find("**", i + 2)
            if j != -1:
                _insert_with_tag(tw, text[i + 2 : j], "md_bold")
                i = j + 2
                continue
        if text.startswith("*", i):
            j = text.find("*", i + 1)
            if j != -1:
                _insert_with_tag(tw, text[i + 1 : j], "md_italic")
                i = j + 1
                continue
        tw.insert("end", text[i])
        i += 1


def _open_url(url: str):
    try:
        import webbrowser

        webbrowser.open(url)
    except Exception:
        pass

