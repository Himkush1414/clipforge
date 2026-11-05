"""Caption preset format, built-in presets and the bundled font registry."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

FONTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"

# Family name as libass sees it -> file. All are SIL OFL, see docs/CREDITS.md.
FONT_FILES: dict[str, str] = {
    "Anton": "Anton-Regular.ttf",
    "Archivo Black": "ArchivoBlack-Regular.ttf",
    "Bangers": "Bangers-Regular.ttf",
    "Bebas Neue": "BebasNeue-Regular.ttf",
    "Hind": "Hind-Bold.ttf",
    "Lilita One": "LilitaOne-Regular.ttf",
    "Poppins Black": "Poppins-Black.ttf",
    "Poppins ExtraBold": "Poppins-ExtraBold.ttf",
}

HEX_COLOR = re.compile(r"^#[0-9a-fA-F]{6}$")


class CaptionPreset(BaseModel):
    """All sizes are relative, so a preset looks the same at 1080p and 8K."""

    name: str = "Custom"
    font: str = "Anton"
    emphasis_font: str | None = None
    uppercase: bool = True
    max_words: int = Field(default=3, ge=1, le=3)
    size: float = Field(default=0.1, ge=0.03, le=0.25, description="Text size as a share of the short side")
    emphasis_scale: float = Field(default=1.3, ge=1.0, le=2.0)
    filler_scale: float = Field(default=0.8, ge=0.5, le=1.0)
    text_color: str = "#FFFFFF"
    active_color: str = "#FFE600"
    emphasis_color: str | None = None
    outline_color: str = "#000000"
    outline: float = Field(default=0.08, ge=0.0, le=0.3, description="Outline as a share of the text size")
    shadow_color: str = "#000000"
    shadow: float = Field(default=0.04, ge=0.0, le=0.3)
    glow_color: str | None = None
    glow: float = Field(default=0.0, ge=0.0, le=0.5)
    active_style: Literal["color", "box"] = "color"
    box_color: str = "#7C3AED"
    animation: Literal["pop", "bounce", "fade", "slide", "none"] = "pop"
    active_pop: float = Field(default=1.12, ge=1.0, le=1.5)
    tilt: float = Field(default=0.0, ge=0.0, le=12.0, description="Alternating word tilt in degrees")
    position: Literal["lower", "center", "upper"] = "lower"
    y_offset: float = Field(default=0.0, ge=-0.2, le=0.2)
    word_gap: float = Field(default=0.28, ge=0.1, le=0.8, description="Gap between words, share of text size")

    @field_validator("font", "emphasis_font")
    @classmethod
    def _known_font(cls, value: str | None) -> str | None:
        if value is not None and value not in FONT_FILES:
            raise ValueError(f"Unknown font: {value}")
        return value

    @field_validator("text_color", "active_color", "outline_color", "shadow_color", "box_color")
    @classmethod
    def _color(cls, value: str) -> str:
        if not HEX_COLOR.match(value):
            raise ValueError(f"Colors must look like #RRGGBB, got {value}")
        return value.upper()

    @field_validator("emphasis_color", "glow_color")
    @classmethod
    def _optional_color(cls, value: str | None) -> str | None:
        return cls._color(value) if value is not None else None


BUILTIN_PRESETS: dict[str, CaptionPreset] = {
    "anime-impact": CaptionPreset(
        name="Anime Impact",
        font="Anton",
        size=0.14,
        emphasis_scale=1.4,
        filler_scale=0.72,
        active_color="#FFE600",
        emphasis_color="#FF3B30",
        outline=0.1,
        shadow=0.05,
        animation="pop",
        active_pop=1.16,
    ),
    "manga-punch": CaptionPreset(
        name="Manga Punch",
        font="Bangers",
        size=0.15,
        emphasis_scale=1.45,
        filler_scale=0.75,
        active_color="#FF2D55",
        outline=0.12,
        shadow=0.07,
        animation="bounce",
        active_pop=1.2,
        tilt=4.0,
        word_gap=0.22,
    ),
    "neon-glow": CaptionPreset(
        name="Neon Glow",
        font="Lilita One",
        size=0.125,
        emphasis_scale=1.3,
        active_color="#22F7FF",
        outline_color="#12002B",
        outline=0.06,
        shadow=0.0,
        glow_color="#FF2BD6",
        glow=0.22,
        animation="fade",
        active_pop=1.1,
    ),
    "karaoke-box": CaptionPreset(
        name="Karaoke Box",
        font="Poppins Black",
        size=0.095,
        emphasis_scale=1.15,
        filler_scale=0.9,
        active_color="#FFFFFF",
        active_style="box",
        box_color="#7C3AED",
        outline=0.07,
        shadow=0.0,
        animation="none",
        active_pop=1.06,
    ),
    "one-word-punch": CaptionPreset(
        name="One Word Punch",
        font="Archivo Black",
        max_words=1,
        size=0.17,
        emphasis_scale=1.3,
        filler_scale=0.85,
        active_color="#FFE600",
        outline=0.09,
        animation="pop",
        active_pop=1.14,
        position="center",
    ),
    "podcast-clean": CaptionPreset(
        name="Podcast Clean",
        font="Poppins ExtraBold",
        uppercase=False,
        size=0.085,
        emphasis_scale=1.15,
        filler_scale=0.92,
        active_color="#34D399",
        outline=0.05,
        shadow=0.06,
        animation="fade",
        active_pop=1.04,
    ),
    "cinema-subtle": CaptionPreset(
        name="Cinema Subtle",
        font="Bebas Neue",
        size=0.1,
        emphasis_scale=1.1,
        filler_scale=0.95,
        active_color="#FBBF24",
        outline=0.05,
        shadow=0.07,
        animation="slide",
        active_pop=1.0,
    ),
}

DEFAULT_PRESET = "anime-impact"
