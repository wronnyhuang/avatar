"""Record screen+sound of the page in flash mode; mode A vs mode B. python run_sync.py <mode> <outdir>

Needs a display + PulseAudio sink to record: either set HEADENV_DIR to a folder holding a `headenv.py` with a
`HeadEnv(w, h)` context manager (it starts Xvfb + a private PulseAudio with an 'evsink' sink and returns the env via
.env()), or run inside an existing session where DISPLAY and PULSE_SERVER are already set (sink 'evsink' must exist).

Env: PORT / INST (the tools/devinst.sh instance; key = dev-<INST>-0123456789abcdef unless AVATAR_KEY is set),
     AVATAR_URL (overrides PORT; base URL of a running server), MIC (mic WAV; default
     $VOICE_EVAL_DIR/mic_commands.wav, VOICE_EVAL_DIR default <repo>/runs/voice_eval), FLASH, MOCKWAV, DQ, XQ, SECS.
"""
import subprocess, sys, time, os, json, contextlib
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if os.environ.get('HEADENV_DIR'):
    sys.path.insert(0, os.environ['HEADENV_DIR'])
try:
    from headenv import HeadEnv
except ImportError:
    class HeadEnv(contextlib.AbstractContextManager):
        """fallback: use the current DISPLAY / PULSE_SERVER"""
        def __init__(self, w, h):
            if not os.environ.get('DISPLAY') or not os.environ.get('PULSE_SERVER'):
                sys.exit('run_sync.py: set HEADENV_DIR, or DISPLAY and PULSE_SERVER')
        def env(self):
            return {'DISPLAY': os.environ['DISPLAY'], 'PULSE_SERVER': os.environ['PULSE_SERVER']}
        def __exit__(self, *a):
            return False
if len(sys.argv) < 3 or sys.argv[1] in ('-h', '--help'):
    sys.exit(__doc__)
mode, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
PORT = os.environ.get('PORT', '18910'); INST = os.environ.get('INST', 'voice')
BASE = os.environ.get('AVATAR_URL') or ('http://127.0.0.1:' + PORT + '/')
KEY = os.environ.get('AVATAR_KEY') or ('dev-' + INST + '-0123456789abcdef')
URL = (BASE + '?k=' + KEY + '&voice=' + mode + '&flash=' + os.environ.get('FLASH', '1')
       + '&camera=0&auto=1&hud=0&mock=' + os.environ.get('MOCKWAV', 'beeps.wav') + '&mock_lat_ms=200&intent=0'
       + os.environ.get('DQ', '&delay=262&dmin=262&dmax=262') + os.environ.get('XQ', ''))
EVAL_DIR = os.environ.get('VOICE_EVAL_DIR') or os.path.join(REPO, 'runs', 'voice_eval')
MIC = os.environ.get('MIC', os.path.join(EVAL_DIR, 'mic_commands.wav'))
with HeadEnv(800, 1000) as h:
    env = dict(os.environ, **h.env())
    rec = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-thread_queue_size', '512', '-f', 'x11grab', '-draw_mouse', '0',
                            '-framerate', '60', '-video_size', '800x1000', '-i', h.env()['DISPLAY'],
                            '-thread_queue_size', '1024', '-f', 'pulse', '-server', h.env()['PULSE_SERVER'].replace('unix:', ''),
                            '-i', 'evsink.monitor', '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18', '-c:a', 'pcm_s16le',
                            out + '/rec.mkv'], stdin=subprocess.PIPE, env=env)
    time.sleep(1.0)
    subprocess.run(['node', os.path.join(HERE, 'drive.cjs'), URL, MIC, os.environ.get('SECS', '58'), out + '/page.json'],
                   env=env, timeout=200)
    rec.stdin.write(b'q'); rec.stdin.flush()
    try: rec.wait(10)
    except subprocess.TimeoutExpired: rec.kill()
print('recorded', out)
