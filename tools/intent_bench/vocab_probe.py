"""Which words does the vosk model's vocabulary know? (grammar recognizer warns on out-of-vocabulary words)
usage: <venv python> -I vocab_probe.py <model dir> word1,word2,...   -> prints OOV words (one JSON line)"""
import json, os, sys, tempfile
import vosk
m = vosk.Model(sys.argv[1])
words = [w for w in sys.argv[2].split(",") if w]
# vosk prints "Ignoring word missing in vocabulary: 'x'" on stderr (fd 2) at log level 0
vosk.SetLogLevel(0)
fd, path = tempfile.mkstemp()
saved = os.dup(2); os.dup2(fd, 2)
try:
    vosk.KaldiRecognizer(m, 16000, json.dumps(words + ["[unk]"]))
finally:
    os.dup2(saved, 2); os.close(fd)
txt = open(path).read(); os.unlink(path)
oov = sorted({ln.split("'")[1] for ln in txt.splitlines() if "missing in vocabulary" in ln and "'" in ln})
print(json.dumps({"n": len(words), "oov": oov}))
