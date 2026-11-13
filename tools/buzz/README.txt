# buzz: audio-output capture + analysis
Null sink: pactl load-module module-null-sink sink_name=avatarbuzz (unload after; name via BUZZ_SINK). Chrome streams are tagged media.role=test so no other Chrome's mute state changes.
run.cjs = voice.js playout graph in isolation; devcheck.cjs = the real page on a running instance (AVATAR_URL / AVATAR_KEY); sinkrun.sh/sinkdev.sh add the speaker capture.
cmp.py ref.wav cap.wav: aligned, per-200 ms 257-tap LS fit removes linear filtering -> residual SNR per band, clicks, time-jumps, gaps, clipping. Run with OMP_NUM_THREADS=2.
