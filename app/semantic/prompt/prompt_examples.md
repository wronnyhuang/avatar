## Examples

Command: "Tap your temple like you just had a great idea"
```python
with parallel():
    touch('right', 'right_temple', dur=0.6)
    look_at('upper_left', dur=0.6)
    expression('thinking', 0.7)
for i in range(2):
    move_hand('right', left=-3, dur=0.12)
    move_hand('right', left=3, dur=0.12)
with parallel():
    point_at('up', hand='right', dur=0.4)
    look_at('camera', dur=0.4)
    expression('big_smile')
    raise_brow(amount=0.8)
hold(0.7)
```

Command: "Blow me a kiss"
```python
with parallel():
    reach('right', 'mouth', part='fingertips', shape='pinch', dur=0.5)
    expression('kiss', dur=0.4)
    close_eyes(amount=0.6, dur=0.3)
hold(0.3)
with parallel():
    arm('right', 'forward_up', extend=0.85, palm='up', shape='open', dur=0.5)
    look_at('camera', dur=0.3)
    expression('smile', dur=0.4)
    open_eyes()
hold(0.6)
```

Command: "Do a little victory dance"
```python
with parallel():
    arm('both', 'up', extend=0.9, shape='fist', dur=0.5)
    expression('big_smile')
    jump(height_cm=15)
with parallel():
    sway(cycles=2, deg=10)
    oscillate('left_hand', 'x', 8, cycles=4, dur=2.4)
    oscillate('right_hand', 'x', 8, cycles=4, dur=2.4, phase_deg=180)
    look_at('camera')
```
