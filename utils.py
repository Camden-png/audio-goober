import copy
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import shutil
import shlex
import subprocess
from typing import (
    Any, Dict, Optional
)

from pydub import AudioSegment

TRIM_START = "trim_start"
TRIM_END = "trim_end"
DURATION = "duration"
VOLUME = "volume"
FADE_IN = "fade_in"
FADE_OUT = "fade_out"
PADDING_START = "padding_start"
PADDING_END = "padding_end"

# Define order of operations...
DEFAULT_METADATA_DICT = {
    TRIM_START: 0.0,
    TRIM_END: 0.0,
    DURATION: -1.0,
    VOLUME: 1.0,
    FADE_IN: 0.0,
    FADE_OUT: 0.0,
    PADDING_START: 0.0,
    PADDING_END: 0.0
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "audio")
PROCESSED_DIR = os.path.join(SCRIPT_DIR, "processed_audio")

METADATA_JSON = "metadata.json"

AUDIO_EXTENSIONS = [".mp3", ".wav"]
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2
CHANNELS = 1


def _run_command(command: str) -> str:
    process = subprocess.run(
        shlex.split(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    stdout = process.stdout.decode().strip()
    stderr = process.stderr.decode().strip()
    return stdout or stderr


def _validate_metadata(metadata: Dict[str, Any]) -> None:
    for key in [TRIM_START, TRIM_END, FADE_IN, FADE_OUT, PADDING_START, PADDING_END]:
        if metadata[key] < 0:
            raise ValueError(f"`{key}` must be >= 0, got `{metadata[key]}`!")
    if metadata[VOLUME] <= 0:
        raise ValueError(f"`{VOLUME}` must be > 0, got `{metadata[VOLUME]}`!")
    if metadata[DURATION] != -1.0 and metadata[DURATION] <= 0:
        raise ValueError(f"`{DURATION}` must be > 0 or -1.0 (disabled), got `{metadata[DURATION]}`!")


def _is_bundle(_dict: Dict[str, Any]) -> bool:
    return isinstance(list(_dict.values())[0], dict)


def _process_entry(key: str, _dict: Dict[str, Any]) -> None:
    # Standalone file...
    if not _is_bundle(_dict):
        processed_file_name = os.path.splitext(key)[0] + ".wav"
        processed_file_path = os.path.join(PROCESSED_DIR, processed_file_name)

        raw_file_path = os.path.join(RAW_DIR, key)
        segment = _apply_effects(raw_file_path, _dict)
        segment.export(processed_file_path, format="wav")
        return

    # Bundle - combine files into one...
    processed_file_name = key + ".wav"
    processed_file_path = os.path.join(PROCESSED_DIR, processed_file_name)
    os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)

    combined_audio: Optional[AudioSegment] = None
    for _file, metadata_dict in _dict.items():
        raw_file_path = os.path.join(RAW_DIR, key, _file)
        segment = _apply_effects(raw_file_path, metadata_dict)
        combined_audio = segment if not combined_audio else combined_audio.overlay(segment)

    combined_audio.export(processed_file_path, format="wav")


def _apply_effects(file_path: str, metadata: Dict[str, Any]) -> AudioSegment:
    segment = AudioSegment.from_file(file_path).set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(SAMPLE_WIDTH)

    if metadata[TRIM_START] != 0.0:
        segment = segment[metadata[TRIM_START] * 1000:]
    if metadata[TRIM_END] != 0.0:
        segment = segment[:-metadata[TRIM_END] * 1000]
    if metadata[DURATION] > 0.0 and metadata[DURATION] * 1000 < len(segment):
        segment = segment[:metadata[DURATION] * 1000]
    if metadata[VOLUME] != 1.0:
        segment = segment.apply_gain(10 * math.log10(metadata[VOLUME]))
    if metadata[FADE_IN] != 0.0:
        segment = segment.fade_in(metadata[FADE_IN] * 1000)
    if metadata[FADE_OUT] != 0.0:
        segment = segment.fade_out(metadata[FADE_OUT] * 1000)
    if metadata[PADDING_START] != 0.0:
        segment = AudioSegment.silent(duration=metadata[PADDING_START] * 1000, frame_rate=SAMPLE_RATE) + segment
    if metadata[PADDING_END] != 0.0:
        segment = segment + AudioSegment.silent(duration=metadata[PADDING_END] * 1000, frame_rate=SAMPLE_RATE)

    return segment


def dumps(_input: Any, **kwargs) -> str:
    json_str = json.dumps(
        _input, **{"indent": 2, "ensure_ascii": False, **kwargs}
    )
    return json_str.replace(r"\\", "/")


def get_bundles_dict(_dir: str = RAW_DIR) -> Dict[str, Any]:
    bundles_dict, metadata_dict = {}, {}

    # Recurse through files...
    for root, _, files in os.walk(_dir):

        for _file in files:
            key = os.path.relpath(root, RAW_DIR)
            is_bundle = key != "."
            key = key if is_bundle else _file

            if _file == METADATA_JSON:
                metadata_path = os.path.join(root, _file)
                with open(metadata_path) as metadata_file:
                    metadata_dict[key] = json.load(metadata_file)

            extension = os.path.splitext(_file)[1]
            if extension not in AUDIO_EXTENSIONS:
                continue

            if key not in bundles_dict:
                bundles_dict[key] = {}

            _copy = copy.deepcopy(DEFAULT_METADATA_DICT)
            if is_bundle:
                bundles_dict[key][_file] = _copy
            else:
                bundles_dict[key] = _copy

    # Process metadata files...
    for key, files_dict in metadata_dict.items():
        for _file, override_dict in files_dict.items():
            if _file not in bundles_dict[key]:
                continue
            bundles_dict[key][_file].update(override_dict)

    return bundles_dict


def validate_bundles_dict(bundles_dict: Dict[str, Any]) -> None:
    for _dict in list(bundles_dict.values()):
        if _is_bundle(_dict):
            for _file, metadata in _dict.items():
                _validate_metadata(metadata)
        else:
            _validate_metadata(_dict)


def process_bundles_dict(bundles_dict: Dict[str, Any]) -> None:
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_process_entry, key, _dict)
            for key, _dict in bundles_dict.items()
        ]
        for _future in futures:
            _future.result()  # Exceptions can be raised here!


def main() -> None:
    for binary in ["ffmpeg", "ffprobe"]:
        output = _run_command(f"{binary} -version")
        if not output.startswith(f"{binary} version"):
            raise RuntimeError(f"`{binary}` is not installed!")

    if os.path.exists(PROCESSED_DIR):
        shutil.rmtree(PROCESSED_DIR)
    os.makedirs(PROCESSED_DIR)

    bundles_dict = get_bundles_dict()
    validate_bundles_dict(bundles_dict)
    print(dumps(bundles_dict))

    process_bundles_dict(bundles_dict)


if __name__ == "__main__":
    main()
