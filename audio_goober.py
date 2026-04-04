import asyncio
from dataclasses import dataclass
from enum import Enum
import os
from typing import (
    Any, Dict, Optional, Tuple
)
import wave

from utils import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH

import flet as ft
import numpy as np
import pyaudio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed_audio")

APP_NAME = "Audio Goober"
NO_AUDIO = ">>> N/A"
INVISIBLE = ""

PROGRESS_BAR_WIDTH = 300
DEFAULT_VOLUME = 1.0
MAX_VOLUME = 3.0
VOLUME_NUDGE = 0.25
NUM_DIGITS = len(
    str(VOLUME_NUDGE).split(".")[-1]
)

CHUNK = 1024

SCALE_DEFAULT = 1.0

SCALE_HOVER = 1.02
SCALE_HOVER_DURATION = 50

SCALE_PRESS = 0.98
SCALE_PRESS_DURATION = 100


class Timing(Enum):
    START = "start"
    END = "end"


# is_playing = False -> nothing loaded, player is idle
# is_playing = True && is_paused = False -> actively streaming audio
# is_playing = True && is_paused = True  -> audio loaded & stream open, but outputting silence
@dataclass
class PlayerState:
    is_playing: bool = False
    is_paused: bool = False
    pa: Optional[Any] = None  # PyAudio...
    stream: Optional[Any] = None

    frames: bytes = b""
    frame_pos: int = 0
    total_frames: int = 0

    volume: float = DEFAULT_VOLUME
    audio_path: Optional[str] = None


@dataclass
class MiscState:
    curr_dir: str = PROCESSED_DIR
    indicators: Optional[Dict] = None


def _text(text: Optional[str] = None, **kwargs) -> ft.Text:
    kwargs.setdefault("color", ft.Colors.BLACK)
    if text:
        kwargs["value"] = text
    return ft.Text(**kwargs)


def _icon(icon: str, **kwargs) -> ft.Icon:
    return ft.Icon(icon, color=ft.Colors.BLACK, **kwargs)


def _icon_button(icon: str, **kwargs) -> ft.IconButton:
    return ft.IconButton(icon=icon, icon_color=ft.Colors.BLACK, **kwargs)


def _toggle_clickability(control: ft.Control, enabled: bool) -> None:
    control.disabled = not enabled
    control.mouse_cursor = ft.MouseCursor.CLICK if enabled else ft.MouseCursor.FORBIDDEN
    control.icon_color = ft.Colors.BLACK if enabled else ft.Colors.GREY_500


def _border(**kwargs) -> ft.Border:
    return ft.Border.all(2, color=ft.Colors.BLACK, **kwargs)


def _shadow(**kwargs) -> ft.BoxShadow:
    return ft.BoxShadow(
        blur_radius=0,
        spread_radius=0,
        color=ft.Colors.BLUE_GREY_300,
        offset=ft.Offset(5, 5),
        **kwargs
    )


def _press_animation() -> ft.Animation:
    return ft.Animation(SCALE_PRESS_DURATION, ft.AnimationCurve.EASE_OUT)


def _hover(
        container: ft.Container,
        on_click: Optional[Any] = None,
        timing: Optional[Timing] = None
) -> None:
    container.scale = SCALE_DEFAULT

    def _on_hover(event: ft.ControlEvent) -> None:
        container.animate_scale = ft.Animation(SCALE_HOVER_DURATION, ft.AnimationCurve.EASE_IN_OUT)
        container.scale = SCALE_HOVER if event.data else SCALE_DEFAULT
        container.update()

    # Mouse button down...
    def _on_tap_down(event: ft.ControlEvent) -> None:
        container.animate_scale = _press_animation()
        container.scale = SCALE_PRESS
        container.update()

        async def _press() -> None:
            await asyncio.sleep(SCALE_PRESS_DURATION / 1000)
            if timing is Timing.END:
                on_click(event)
            try:
                container.scale = SCALE_HOVER
                container.update()
            except RuntimeError:
                pass

        container.page.run_task(_press)

        if on_click and timing is Timing.START:
            on_click(event)

    container.on_hover = _on_hover
    container.on_tap_down = _on_tap_down


def _card(
        tile: ft.ListTile,
        add_padding: bool = False,
        timing: Timing = Timing.START,
        **kwargs
) -> ft.Column:
    tile.mouse_cursor = ft.MouseCursor.CLICK

    container = ft.Container(
        content=tile,
        border=_border(),
        border_radius=6,
        bgcolor=ft.Colors.WHITE,
        shadow=_shadow(),
        animate_scale=_press_animation()
    )

    # Bubble-up tile's `on_click` event to container...
    # (Otherwise the tile consumes the event and the container's `on_click` never fires...)
    _on_click = tile.on_click
    _hover(container, on_click=_on_click, timing=timing)
    tile.on_click = None

    items = [container]
    if add_padding:
        # `GestureDetector(...)` used as padding b.c. can
        # disable all ink & set mouse cursor...
        items.append(
            ft.GestureDetector(
                content=ft.Container(height=16),
                mouse_cursor=ft.MouseCursor.BASIC,
                on_tap=_on_click
            )
        )

    return ft.Column(
        items,
        spacing=0,
        **kwargs
    )


def app(page: ft.Page) -> None:
    page.title = APP_NAME
    page.fonts = {"Meslo": "fonts/Meslo.ttf"}
    page.theme = ft.Theme(font_family="Meslo")

    page.window.resizable = False

    page.padding = 20
    page.window.width = 500
    page.window.height = 600

    player_state = PlayerState()
    misc_state = MiscState()

    def _toggle_playback() -> None:
        not_playing = not player_state.is_playing

        if not_playing and player_state.audio_path:
            play(player_state.audio_path, reset_volume=False, bounce=False)

        if not_playing:
            return

        player_state.is_paused = not player_state.is_paused
        _draw()

    def _adjust_volume(delta: float) -> None:
        # Allow changes to volume for finished audio...
        if not player_state.audio_path:
            return

        player_state.volume = max(
            0.0, min(MAX_VOLUME, round(player_state.volume + delta, NUM_DIGITS))
        )

        _toggle_clickability(volume_up_button, enabled=player_state.volume < MAX_VOLUME)
        _toggle_clickability(volume_down_button, enabled=player_state.volume > 0.0)
        _draw()

    pause_button = _icon_button(
        ft.Icons.PLAY_ARROW, on_click=lambda _: _toggle_playback()
    )

    volume_up_button = _icon_button(
        ft.Icons.ADD, on_click=lambda _: _adjust_volume(VOLUME_NUDGE)
    )

    volume_down_button = _icon_button(
        ft.Icons.REMOVE, on_click=lambda _: _adjust_volume(-VOLUME_NUDGE)
    )

    for button in [pause_button, volume_up_button, volume_down_button]:
        _toggle_clickability(button, enabled=False)

    progress_bar = ft.ProgressBar(
        value=0,
        width=PROGRESS_BAR_WIDTH,
        bar_height=10,
        border_radius=4,
        color=ft.Colors.RED,
        bgcolor=ft.Colors.BLACK_12
    )

    progress_container = ft.Container(
        content=progress_bar,
        border=_border(),
        border_radius=6,
        padding=2,
        margin=ft.Margin.only(top=6),
        bgcolor=ft.Colors.WHITE,
        shadow=_shadow()
    )

    player_text = _text(
        NO_AUDIO, size=14
    )

    volume_text = _text(
        INVISIBLE, size=14
    )

    bottom_div = ft.Container(
        content=ft.Row(
            [
                ft.Column(
                    [
                        progress_container,
                        ft.Column(
                            [player_text, volume_text],
                            spacing=2,
                            horizontal_alignment=ft.CrossAxisAlignment.END,
                            width=PROGRESS_BAR_WIDTH + 10
                        )
                    ],
                    spacing=8
                ),
                ft.Container(
                    ft.VerticalDivider(width=2, thickness=2, color=ft.Colors.BLACK),
                    margin=ft.Margin.only(left=10)
                ),
                ft.Column(
                    [
                        ft.Container(
                            content=pause_button,
                            margin=ft.Margin.only(top=-4.05)  # Weird magic number...
                        ),
                        ft.Container(
                            content=ft.Row(
                                [
                                    volume_up_button,
                                    _text("/", size=16, color=ft.Colors.GREY_500),
                                    volume_down_button
                                ],
                                spacing=0
                            ),
                            margin=ft.Margin.only(top=8.15)
                        )
                    ],
                    spacing=0
                )
                # Add another row later...
            ],
            spacing=4,
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
            height=76
        ),
        padding=20,
        scale=SCALE_DEFAULT,
        animate_scale=ft.Animation(SCALE_PRESS_DURATION, ft.AnimationCurve.EASE_IN_OUT)
    )

    # Navigate directories...
    def navigate(path: str) -> None:
        misc_state.curr_dir = path
        render_elements()

    def _reset_player(
            hard_reset: bool = False,
            reset_volume: bool = True,
            reset_bar: bool = True
    ) -> None:
        if player_state.stream:
            player_state.stream.stop_stream()
            player_state.stream.close()
            player_state.stream = None

        if player_state.pa:
            player_state.pa.terminate()
            player_state.pa = None

        player_state.is_playing = False
        player_state.is_paused = False

        # Hard reset...
        if hard_reset:
            player_state.audio_path = None
            player_text.value = NO_AUDIO

        if hard_reset or reset_volume:
            player_state.volume = DEFAULT_VOLUME

        if hard_reset or reset_bar:
            player_state.frames = b""
            player_state.frame_pos = 0
            player_state.total_frames = 0
            progress_bar.value = 0

    async def _bounce_player() -> None:
        bottom_div.animate_scale = ft.Animation(SCALE_PRESS_DURATION, ft.AnimationCurve.EASE_IN_OUT)
        bottom_div.scale = SCALE_PRESS - 0.02
        bottom_div.update()
        await asyncio.sleep(SCALE_PRESS_DURATION / 1000)
        bottom_div.scale = SCALE_DEFAULT
        bottom_div.update()

    # Play audio file...
    def play(path: str, reset_volume: bool = True, bounce: bool = True) -> None:
        if path == player_state.audio_path and player_state.is_playing:
            _toggle_playback()
            return

        _reset_player(reset_volume=reset_volume)

        if bounce:
            page.run_task(_bounce_player)

        # Load WAV frames...
        try:
            with wave.open(path) as wf:
                player_state.frames = wf.readframes(wf.getnframes())
                player_state.total_frames = wf.getnframes()
        except (EOFError, wave.Error):
            _reset_player(hard_reset=True)
            return

        player_state.is_playing = True
        player_state.audio_path = path

        def _apply_volume(data: bytes) -> bytes:
            if player_state.volume == DEFAULT_VOLUME:
                return data
            samples = np.frombuffer(data, dtype=np.int16)
            samples = np.clip(samples * player_state.volume, -32768, 32767).astype(np.int16)
            return samples.tobytes()

        def _callback(_: Any, frame_count: int, __: Any, ___: Any) -> Tuple[bytes, Any]:
            num_bytes = frame_count * SAMPLE_WIDTH * CHANNELS

            if player_state.is_paused:
                return b"\x00" * num_bytes, pyaudio.paContinue

            start = player_state.frame_pos * SAMPLE_WIDTH * CHANNELS
            end = start + num_bytes
            data = player_state.frames[start:end]

            player_state.frame_pos += frame_count

            if len(data) < num_bytes:
                data += b"\x00" * (num_bytes - len(data))
                return _apply_volume(data), pyaudio.paComplete

            return _apply_volume(data), pyaudio.paContinue

        player_state.pa = pyaudio.PyAudio()
        player_state.stream = player_state.pa.open(
            format=player_state.pa.get_format_from_width(SAMPLE_WIDTH),
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK,
            stream_callback=_callback
        )

        _draw()

    def render_elements() -> None:
        items = []
        misc_state.indicators = {}
        curr_dir = misc_state.curr_dir
        entries = sorted(os.listdir(curr_dir))

        # Render back button...
        if curr_dir != PROCESSED_DIR:
            items.append(
                _card(
                    ft.ListTile(
                        leading=_icon(ft.Icons.ARROW_BACK),
                        title=_text(".."),
                        on_click=lambda _: navigate(os.path.dirname(curr_dir))
                    ),
                    add_padding=len(entries) > 0,
                    timing=Timing.END
                )
            )

        for i, entry in enumerate(entries):
            add_padding = i != len(entries) - 1
            full_path = os.path.join(curr_dir, entry)

            # Render dirs...
            if os.path.isdir(full_path):
                items.append(
                    _card(
                        ft.ListTile(
                            leading=_icon(ft.Icons.FOLDER),
                            title=_text(entry),
                            on_click=lambda _, _path=full_path: navigate(_path)
                        ),
                        add_padding=add_padding,
                        timing=Timing.END
                    )
                )

            # Render audio files...
            else:
                indicator = ft.Container(
                    width=18,
                    height=18,
                    border_radius=2,
                    bgcolor=ft.Colors.RED,
                    visible=False
                )
                misc_state.indicators[full_path] = indicator
                items.append(
                    _card(
                        ft.ListTile(
                            leading=_icon(ft.Icons.AUDIO_FILE),
                            title=_text(entry),
                            trailing=indicator,
                            on_click=lambda _, path=full_path: play(path)
                        ),
                        add_padding=add_padding
                    )
                )

        page.controls.clear()
        page.controls.append(
            ft.Column(
                [
                    _text(
                        spans=[
                            ft.TextSpan("> "),
                            ft.TextSpan(
                                os.path.relpath(curr_dir, PROCESSED_DIR)
                                if curr_dir != PROCESSED_DIR else APP_NAME,
                                ft.TextStyle(italic=True)
                            )
                        ],
                        size=24,
                        weight=ft.FontWeight.BOLD
                    ),
                    ft.Container(
                        content=ft.Divider(thickness=2, color=ft.Colors.BLACK),
                        margin=ft.Margin.only(top=0, bottom=12)
                    ),
                    ft.ListView(
                        items,
                        expand=True,
                        spacing=0
                    ),
                    bottom_div
                ],
                expand=True,
                spacing=4
            )
        )
        page.update()

    def _draw(manual_update: bool = True) -> None:
        is_playing = (
            player_state.is_playing and
            player_state.stream and player_state.stream.is_active()
        )

        just_finished = player_state.is_playing and not is_playing

        if is_playing or just_finished or manual_update:
            # Enable buttons...
            _toggle_clickability(pause_button, enabled=True)
            _toggle_clickability(volume_up_button, enabled=player_state.volume < MAX_VOLUME)
            _toggle_clickability(volume_down_button, enabled=player_state.volume > 0.0)

            # Progress bar...
            elapsed = player_state.frame_pos / SAMPLE_RATE
            duration = player_state.total_frames / SAMPLE_RATE
            progress_bar.value = min(elapsed / duration, 1.0) if duration else 0.0

            # Current audio file...
            audio_name = os.path.basename(player_state.audio_path)
            player_text.value = (
                f"> {audio_name} [{int(elapsed)}s / {int(duration)}s]"
            )

            # Volume...
            _volume = format(player_state.volume, f".{NUM_DIGITS}f")
            volume_text.value = f"> volume: {_volume}"

            # Restart...
            if is_playing:
                pause_button.icon = ft.Icons.PLAY_ARROW if player_state.is_paused else ft.Icons.PAUSE

            # Active indicator...
            if misc_state.indicators:
                for path, indicator in misc_state.indicators.items():
                    if path == player_state.audio_path:
                        indicator.visible = True
                        indicator.opacity = 0.3 if player_state.is_paused else 1.0
                    else:
                        indicator.visible = False

        if just_finished:
            _reset_player(reset_volume=False, reset_bar=False)
            pause_button.icon = ft.Icons.RESTART_ALT

        page.update()

    async def draw() -> None:
        while True:
            _draw(manual_update=False)
            await asyncio.sleep(0.01)  # Automatically update draw...

    render_elements()
    page.run_task(draw)
    page.update()

    async def _show_window() -> None:
        await asyncio.sleep(1)
        page.window.visible = True
        page.update()

    page.run_task(_show_window)


def main() -> None:
    ft.run(
        app,
        view=ft.AppView.FLET_APP_HIDDEN,
        assets_dir=SCRIPT_DIR
    )


if __name__ == "__main__":
    main()
