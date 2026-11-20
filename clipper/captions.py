"""Word-by-word ASS captions: short lines, the spoken word highlighted, rotating styles."""
from __future__ import annotations
import re

DEFAULT_STYLES = [
    {"name": "bold_white", "font": "Arial Black", "size": 88, "primary": "#FFFFFF", "highlight": "#FFE600",
     "outline": "#000000", "outline_w": 6, "y_frac": 0.62, "upper": True},
    {"name": "yellow_pop", "font": "Arial Black", "size": 96, "primary": "#FFE600", "highlight": "#FFFFFF",
     "outline": "#000000", "outline_w": 7, "y_frac": 0.58, "upper": True},
    {"name": "clean_lower", "font": "Arial", "size": 76, "primary": "#FFFFFF", "highlight": "#3DDC84",
     "outline": "#000000", "outline_w": 5, "y_frac": 0.70, "upper": False},
]
MAX_Y_FRAC = 0.75   # keep captions out of the bottom HUD / platform-UI zone


def ass_time(t: float) -> str:
    cs = max(0, int(round(t * 100)))
    return f"{cs // 360000}:{cs % 360000 // 6000:02d}:{cs % 6000 // 100:02d}.{cs % 100:02d}"


def ass_color(hexrgb: str) -> str:
    h = hexrgb.lstrip("#")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H00{b}{g}{r}&".upper()


def esc(text: str) -> str:
    return re.sub(r"[{}\\]", "", text).replace("\n", " ").strip()


def pick_style(styles: list[dict], cand_id: str) -> dict:
    styles = styles or DEFAULT_STYLES
    n = int(re.sub(r"\D", "", cand_id) or 0)
    return styles[n % len(styles)]


def group_words(words: list[dict], max_words: int = 3, max_gap: float = 0.8) -> list[list[dict]]:
    groups, cur = [], []
    for w in words:
        if cur and (len(cur) >= max_words or w["start"] - cur[-1]["end"] > max_gap
                    or cur[-1]["w"].rstrip().endswith((".", "?", "!"))):
            groups.append(cur)
            cur = []
        cur.append(w)
    if cur:
        groups.append(cur)
    return groups


def build_ass(words: list[dict], clip_start: float, clip_end: float, style: dict,
              W: int, H: int, y_frac: float | None = None, words_per_line: int = 3) -> str:
    dur = clip_end - clip_start
    y = int(H * min(MAX_Y_FRAC, y_frac if y_frac is not None else style["y_frac"]))
    hi, base = ass_color(style["highlight"]), ass_color(style["primary"])
    head = (
        "[Script Info]\nScriptType: v4.00+\nWrapStyle: 2\nScaledBorderAndShadow: yes\n"
        f"PlayResX: {W}\nPlayResY: {H}\n\n[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,"
        "Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,"
        "MarginL,MarginR,MarginV,Encoding\n"
        f"Style: Cap,{style['font']},{style['size']},{base},{base},{ass_color(style['outline'])},&H64000000&,"
        f"-1,0,0,0,100,100,0,0,1,{style['outline_w']},2,5,60,60,60,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    inside = [w for w in words if clip_start <= (w["start"] + w["end"]) / 2 < clip_end]
    groups = group_words(inside, words_per_line)
    lines = []
    for gi, g in enumerate(groups):
        nxt = groups[gi + 1][0]["start"] if gi + 1 < len(groups) else clip_end
        for i, w in enumerate(g):
            t0 = max(0.0, w["start"] - clip_start)
            t1 = (g[i + 1]["start"] if i + 1 < len(g) else min(w["end"] + 0.25, nxt)) - clip_start
            t1 = min(t1, dur)
            if t1 - t0 < 0.05:
                continue
            parts = []
            for j, x in enumerate(g):
                txt = esc(x["w"]).upper() if style.get("upper") else esc(x["w"])
                parts.append(f"{{\\c{hi}}}{txt}{{\\c{base}}}" if j == i else txt)
            lines.append(f"Dialogue: 0,{ass_time(t0)},{ass_time(t1)},Cap,,0,0,0,,"
                         f"{{\\pos({W // 2},{y})}}{' '.join(parts)}")
    return head + "\n".join(lines) + ("\n" if lines else "")
