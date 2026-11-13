#!/usr/bin/env python3
"""Synthesize the voice-eval mic WAVs, their manifests, and mock reply WAVs.

  python3 make_clips.py [--out DIR] [--voice alloy] [--espeak] [--force]

Outputs (DIR defaults to $VOICE_EVAL_DIR, else <repo>/runs/voice_eval):
  clips/<id>.wav          one trimmed, peak-normalized clip per utterance (48 kHz mono s16)
  clips/clips.json        text / kind / source (openai|espeak) / duration per clip
  mic_mixed.wav  + .json  4.0 s silence, then 8 questions + 4 commands interleaved, each + 10 s silence
  mic_smoke.wav  + .json  2 questions
  mic_commands.wav + .json the 4 commands only
  mock/*.wav + index.json mock-mode reply WAVs (24 kHz mono s16, espeak-ng)

Manifest per mic WAV: {wav, sr, duration_s, lead_s, gap_s, threshold_dbfs, clips:[{id, kind, text, start_s,
end_s}]}; start_s/end_s = first/last sample whose |x| is above -45 dBFS inside that clip's region of the WAV.

TTS: OpenAI gpt-4o-mini-tts (pcm 24 kHz), OPENAI_API_KEY used only in the Authorization header. Clips already
on disk are reused (no second charge) unless --force; the run prints an estimated TTS cost.
Falls back to espeak-ng per clip if the TTS call fails.
"""
import argparse, json, os, subprocess, sys, tempfile, time, urllib.request, urllib.error, wave

import numpy as np
from scipy.signal import resample_poly

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.environ.get('VOICE_EVAL_DIR') or os.path.join(REPO, 'runs', 'voice_eval')
SR = 48000
THRESH_DBFS = -45.0
THRESH = 10 ** (THRESH_DBFS / 20)          # 0.00562 of full scale
LEAD_S, GAP_S = 4.0, 10.0
USD_PER_MIN = 0.015                         # list-price estimate for gpt-4o-mini-tts

QUESTIONS = [
    ('q01_hi', "Hi Felix, how are you today?"),
    ('q02_food', "What's your favorite food?"),
    ('q03_fact', "Can you tell me a fun fact about foxes?"),
    ('q04_morning', "What did you do this morning?"),
    ('q05_snow', "Do you like playing in the snow?"),
    ('q06_friend', "What's your best friend's name?"),
    ('q07_tail', "What color is your tail?"),
    ('q08_sleep', "Where do foxes sleep at night?"),
]
COMMANDS = [
    ('c01_paw', "Raise your right paw."),
    ('c02_wave', "Wave at me!"),
    ('c03_wink', "Can you wink your left eye?"),
    ('c04_look', "Look up at the ceiling."),
]
MIXES = {
    'mic_mixed': ['q01_hi', 'q02_food', 'c01_paw', 'q03_fact', 'q04_morning', 'c02_wave',
                  'q05_snow', 'c03_wink', 'q06_friend', 'q07_tail', 'c04_look', 'q08_sleep'],
    'mic_smoke': ['q01_hi', 'q02_food'],
    'mic_commands': ['c01_paw', 'c02_wave', 'c03_wink', 'c04_look'],
}
MOCK = [
    ('mock_01_hi', "Hi! I'm Felix the fox."),
    ('mock_02_food', "I love blueberries and crunchy apples."),
    ('mock_03_paw', "Okay, raising my right paw!"),
    ('mock_04_fact', "Foxes can hear a mouse squeak far away."),
    ('mock_05_wave', "Here is a big wave for you!"),
    ('mock_06_wink', "Wink! Did you see that?"),
]
TTS_INSTRUCTIONS = ("Speak naturally and clearly at a normal conversational pace, like a friendly adult "
                    "talking to a cartoon fox. No long pauses.")


def write_wav(path, x, sr):
    x = np.clip(np.asarray(x, dtype=np.float64), -1, 1)
    pcm = np.round(x * 32767).astype('<i2')
    with wave.open(path, 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(path, 'rb') as w:
        assert w.getsampwidth() == 2, path
        sr, ch, n = w.getframerate(), w.getnchannels(), w.getnframes()
        x = np.frombuffer(w.readframes(n), dtype='<i2').astype(np.float64) / 32768
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    return x, sr


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    g = np.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g)


def tts_openai(text, voice):
    key = os.environ.get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY not set')
    body = json.dumps({'model': 'gpt-4o-mini-tts', 'voice': voice, 'input': text,
                       'instructions': TTS_INSTRUCTIONS, 'response_format': 'pcm'}).encode()
    req = urllib.request.Request('https://api.openai.com/v1/audio/speech', data=body, method='POST',
                                 headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'TTS HTTP {e.code}: {e.read()[:200]!r}') from None
    if len(raw) < 4800:
        raise RuntimeError(f'TTS returned only {len(raw)} bytes')
    return np.frombuffer(raw[:len(raw) // 2 * 2], dtype='<i2').astype(np.float64) / 32768, 24000


def tts_espeak(text, speed=165, voice='en-us'):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'e.wav')
        subprocess.run(['espeak-ng', '-v', voice, '-s', str(speed), '-w', p, text], check=True)
        return read_wav(p)


def trim_norm(x, sr, peak=0.708, pre_s=0.03, post_s=0.03):
    """Trim to the region above -45 dBFS (after peak-normalizing), keep short faded margins."""
    m = np.max(np.abs(x)) or 1.0
    x = x * (peak / m)
    idx = np.flatnonzero(np.abs(x) > THRESH)
    a = max(0, idx[0] - int(pre_s * sr)); b = min(len(x), idx[-1] + 1 + int(post_s * sr))
    y = x[a:b].copy()
    nf = int(0.01 * sr)
    y[:nf] *= np.linspace(0, 1, nf); y[-nf:] *= np.linspace(1, 0, nf)
    return y


def cap_pauses(y, sr, max_s=0.25, quiet_dbfs=-40.0):
    """Shorten internal quiet runs (10 ms RMS < quiet_dbfs) longer than max_s to max_s, so a mic VAD with a
    short hangover does not split one utterance into two turns. Returns (y, n_cut)."""
    hop = int(0.01 * sr); n = len(y) // hop
    db = 20 * np.log10(np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-12)
    loud = np.flatnonzero(db >= quiet_dbfs)
    if len(loud) < 2:
        return y, 0
    keep, cuts, prev = [], 0, 0
    for i0, i1 in zip(loud[:-1], loud[1:]):
        gap = i1 - i0 - 1
        if gap * hop > max_s * sr:
            a = (i0 + 1) * hop + int(max_s * sr / 2); b = i1 * hop - int(max_s * sr / 2)
            keep.append(y[prev:a]); prev = b; cuts += 1
    keep.append(y[prev:])
    return np.concatenate(keep), cuts


def above_bounds(seg, sr, offset):
    idx = np.flatnonzero(np.abs(seg) > THRESH)
    if not len(idx):
        return None, None
    return round((offset + idx[0]) / sr, 4), round((offset + idx[-1] + 1) / sr, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=DEFAULT_OUT, help='output dir (default $VOICE_EVAL_DIR or <repo>/runs/voice_eval)')
    ap.add_argument('--voice', default='alloy')
    ap.add_argument('--espeak', action='store_true', help='skip OpenAI TTS, use espeak-ng for all clips')
    ap.add_argument('--force', action='store_true', help='re-synthesize clips already on disk')
    a = ap.parse_args()
    cdir, mdir = os.path.join(a.out, 'clips'), os.path.join(a.out, 'mock')
    os.makedirs(cdir, exist_ok=True); os.makedirs(mdir, exist_ok=True)

    # ---- clips ----
    meta_path = os.path.join(cdir, 'clips.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) and not a.force else {}
    tts_seconds, n_tts, fails = 0.0, 0, []
    for kind, items in (('question', QUESTIONS), ('command', COMMANDS)):
        for cid, text in items:
            p = os.path.join(cdir, cid + '.wav')
            if os.path.exists(p) and cid in meta and meta[cid]['text'] == text and not a.force:
                continue
            src = 'openai'
            try:
                if a.espeak:
                    raise RuntimeError('--espeak')
                x, sr = tts_openai(text, a.voice)
                tts_seconds += len(x) / sr; n_tts += 1
            except Exception as e:  # fall back per clip
                if not a.espeak:
                    fails.append(f'{cid}: {e}')
                x, sr = tts_espeak(text); src = 'espeak-ng'
            y = trim_norm(resample(x, sr, SR), SR)
            write_wav(p, y, SR)
            meta[cid] = {'id': cid, 'kind': kind, 'text': text, 'source': src,
                         'voice': a.voice if src == 'openai' else 'en-us', 'dur_s': round(len(y) / SR, 3)}
            print(f'clip {cid:12s} {src:9s} {len(y)/SR:5.2f}s  {text}')
    json.dump(meta, open(meta_path, 'w'), indent=1)
    if n_tts:
        usd = round(tts_seconds / 60 * USD_PER_MIN, 5)
        print(f'TTS: {n_tts} calls, {tts_seconds:.1f}s audio, est ${usd}')
    for f_ in fails:
        print('TTS fallback ->', f_, file=sys.stderr)

    # ---- mic WAVs + manifests ----
    for name, order in MIXES.items():
        parts, spans, cur = [np.zeros(int(LEAD_S * SR))], [], int(LEAD_S * SR)
        for cid in order:
            y, sr = read_wav(os.path.join(cdir, cid + '.wav'))
            assert sr == SR
            y, ncut = cap_pauses(y, SR)
            if ncut:
                print(f'  {name}: {cid} {ncut} internal pause(s) capped at 0.25 s')
            spans.append((cid, cur, len(y))); parts.append(y); cur += len(y)
            parts.append(np.zeros(int(GAP_S * SR))); cur += int(GAP_S * SR)
        mix = np.concatenate(parts)
        wpath = os.path.join(a.out, name + '.wav')
        write_wav(wpath, mix, SR)
        q = np.round(np.clip(mix, -1, 1) * 32767) / 32768  # measure bounds on what was written
        clips = []
        for cid, off, n in spans:
            s, e = above_bounds(q[off:off + n], SR, off)
            m = meta[cid]
            clips.append({'id': cid, 'kind': m['kind'], 'text': m['text'], 'source': m['source'],
                          'start_s': s, 'end_s': e})
        man = {'wav': wpath, 'sr': SR, 'channels': 1, 'sample_format': 's16', 'duration_s': round(len(mix) / SR, 4),
               'lead_s': LEAD_S, 'gap_s': GAP_S, 'threshold_dbfs': THRESH_DBFS, 'max_internal_pause_s': 0.25,
               'n_questions': sum(c['kind'] == 'question' for c in clips),
               'n_commands': sum(c['kind'] == 'command' for c in clips), 'clips': clips}
        json.dump(man, open(os.path.join(a.out, name + '.json'), 'w'), indent=1)
        print(f'{name}.wav  {man["duration_s"]:.1f}s  {man["n_questions"]}q + {man["n_commands"]}c')

    # ---- mock reply WAVs (24 kHz) ----
    idx = []
    for mid, text in MOCK:
        speed = 165
        for _ in range(6):
            x, sr = tts_espeak(text, speed=speed)
            y = trim_norm(resample(x, sr, 24000), 24000, peak=0.5)
            d = len(y) / 24000
            if d > 3.0: speed += 20
            elif d < 1.5: speed -= 20
            else: break
        if len(y) < int(1.5 * 24000):
            y = np.concatenate([y, np.zeros(int(1.5 * 24000) - len(y))])
        write_wav(os.path.join(mdir, mid + '.wav'), y, 24000)
        idx.append({'file': mid + '.wav', 'text': text, 'dur_s': round(len(y) / 24000, 3)})
        print(f'mock {mid}.wav {len(y)/24000:4.2f}s  {text}')
    json.dump(idx, open(os.path.join(mdir, 'index.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
