"""Master three complete synthetic performances without splicing any words.
Usage: python mix_voices.py assets/performances assets (requires ffmpeg).
"""
import argparse
import subprocess
from pathlib import Path

TAKES = (
    ('welcome', 1.0, 'atempo=0.82'),
    ('welcome-low', 1.02, 'bass=g=1:f=155,equalizer=f=1650:t=q:w=1:g=1'),
    ('welcome-hotel', .94, 'bass=g=2:f=150,treble=g=-1.5:f=3900,'
     'aecho=0.85:0.85:53|113|217|389:0.18|0.13|0.08|0.04'),
)


def render(source, output):
    output.mkdir(parents=True, exist_ok=True)
    for name, rate, tone in TAKES:
        filters = (f'aresample=48000,highpass=f=50,'
                   f'silenceremove=start_periods=1:start_duration=0.01:start_silence=0.04:start_threshold=-52dB,'
                   f'asetrate={round(48000 * rate)},aresample=48000,{tone},loudnorm=I=-14:TP=-1:LRA=7')
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source / f'{name}-out.wav'),
                        '-af', filters, '-ar', '48000', '-ac', '1', '-c:a', 'pcm_s16le',
                        str(output / f'{name}.wav')], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    render(args.source,args.output)
