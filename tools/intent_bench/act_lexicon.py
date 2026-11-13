"""intent: the source of the "act" section of app/intent_rules.json (whole-body moves + props).

    python3 tools/intent_bench/act_lexicon.py        # rewrites the "act" key of app/intent_rules.json in place

The act grammar runs on the RAW clause tokens (after aliases, before the legacy phrase table) and is tried before the
legacy rules (app/intent.py parse_act / app/intent.js parseAct — identical engines). Pattern syntax (verbs, moves, mods):
space-separated elements; 'a/b' = one of; leading '?' = optional; '<O>' = an object noun phrase (np), '<N>' = a number
or quantity ("three", "3", "a couple of"), '<S>' = a destination paw ("to your other paw"), '<P>' = a place ("on the
table"), '@cls' = one word of lexicon class cls (value captured as cls).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RULES = os.path.join(HERE, "..", "..", "app", "intent_rules.json")

NUMS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "to": 2, "too": 2, "for": 4, "won": 1, "tree": 3}
# quantity phrases usable wherever <N> is (multi-word); "to"/"too"/"for"/"won"/"tree" = ASR homophones, ONLY inside <N> times
QTY = [["a couple of", 2], ["a couple", 2], ["couple of", 2], ["a few", 3], ["few", 3], ["several", 4],
       ["a bunch of", 5], ["bunch of", 5], ["lots of", 5], ["a lot of", 5], ["many", 5], ["a hundred", 10],
       ["a million", 10], ["one hundred", 10],
       ["one or two", 2], ["two or three", 3], ["three or four", 4], ["a couple or three", 3]]  # intent c2

COLORS = {c: c for c in ["red", "orange", "yellow", "green", "blue", "purple", "violet", "pink", "brown", "black", "white",
                         "gray", "gold", "silver", "teal", "turquoise", "cyan", "aqua", "magenta", "maroon", "navy",
                         "beige", "lavender", "lilac", "crimson", "scarlet", "indigo", "rainbow", "peach", "mint",
                         "lime", "cream", "tan", "copper", "bronze"]}
COLORS.update({"grey": "gray", "golden": "gold", "read": "red", "blew": "blue", "yellowish": "yellow", "reddish": "red",
               "bluish": "blue", "greenish": "green", "pinkish": "pink", "purply": "purple"})
SIZES = {"big": "big", "bigger": "big", "biggest": "big", "large": "big", "larger": "big", "largest": "big",
         "huge": "big", "giant": "big", "tall": "big", "taller": "big", "tallest": "big", "fat": "big",
         "little": "small", "small": "small", "smaller": "small", "smallest": "small", "tiny": "small", "mini": "small",
         "short": "small", "shorter": "small", "teeny": "small", "itty": "small", "baby": "small"}
ORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "1st": 1, "2nd": 2, "3rd": 3, "4th": 4,
        "last": -1}
# adjectives accepted (and ignored) before a known noun
SHADES = ["light", "dark", "bright", "pale", "shiny", "sparkly", "glittery", "round", "fluffy", "soft", "squishy",
          "old", "new", "pretty", "nice", "cute", "favorite", "favourite", "special", "whole", "real", "fake", "toy",
          "striped", "spotted", "spotty", "empty", "full", "heavy", "wooden", "plastic", "metal", "shiny", "juicy",
          "yummy", "tasty", "delicious", "ripe", "fresh", "stuffed", "furry", "smelly", "stinky", "funny", "cool",
          "lovely", "beautiful", "magic", "magical", "glowing", "square", "long", "cuddly", "squeaky", "bouncy",
          "same", "very", "super", "really", "lil", "wee", "colorful", "colourful", "fancy", "sweet", "precious",
          "silly", "happy", "sad", "mysterious", "ancient", "tasty", "crunchy", "golden"]
# specific nouns (canonical name). Plurals are generated. Multi-word keys use spaces.
NOUN_LIST = """ball balloon block book bell box present gift package parcel cup mug teacup bowl plate spoon fork
apple banana carrot pear lemon cherry strawberry grape watermelon melon coconut cookie cupcake cake candy lollipop
donut doughnut honey jar acorn pinecone leaf flower rose daisy tulip sunflower lily mushroom shell seashell starfish
rock stone pebble stick feather crystal gem jewel coin star moon sun wand crown hat umbrella lantern lamp candle bucket
pail shovel drum teddy doll duck duckie ducky fish bone egg nut acorn map scroll key telescope globe kite yoyo car truck
train boat rocket dinosaur robot puzzle crayon pencil paintbrush brush marble dice cube lettuce bean frisbee trophy medal
flag sock shoe pillow blanket scarf ribbon trumpet guitar xylophone whistle basket backpack bottle glass vase pot plant
cactus sandwich pizza cheese corn peanut berry orange peach lime jewel orb
pumpkin snowball snowflake ring necklace bracelet compass hammer chair desk table stool laptop computer keyboard
monitor tablet notebook headset cap helmet mask scarf glove mitten""".split()
# pair nouns: one thing, never "all of them"
PAIR = {"headphones": "headphones", "headphone": "headphones", "earphones": "earphones", "earbuds": "earbuds",
        "earmuffs": "earmuffs", "glasses": "glasses", "sunglasses": "sunglasses", "goggles": "goggles",
        "spectacles": "glasses", "specs": "glasses", "headphone set": "headphones"}
NOUN_MULTI = {"teddy bear": "teddy_bear", "honey pot": "honey_pot", "honey jar": "honey_jar", "magic wand": "magic_wand",
              "rubber duck": "rubber_duck", "rubber ducky": "rubber_duck", "jelly bean": "jelly_bean",
              "ice cream": "ice_cream", "toy car": "car", "picture book": "book", "spinning top": "spinning_top",
              "water bottle": "water_bottle", "watering can": "watering_can", "beach ball": "beach_ball",
              "soccer ball": "soccer_ball", "tennis ball": "tennis_ball", "basket ball": "basketball",
              "basketball": "basketball", "football": "football", "pine cone": "pinecone", "sea shell": "seashell",
              "star fish": "starfish", "tea cup": "teacup", "cup cake": "cupcake", "water melon": "watermelon",
              "yo yo": "yoyo", "treasure chest": "treasure_chest", "music box": "music_box", "toy box": "toy_box",
              "lunch box": "lunch_box", "flower pot": "flower_pot", "magnifying glass": "magnifying_glass",
              "paint brush": "paintbrush", "jack in the box": "jack_in_the_box",
              "stuffed animal": "stuffed_animal", "cart wheel": "cartwheel_toy", "head phones": "headphones", "ear phones": "earphones",
              "ear muffs": "earmuffs", "ear buds": "earbuds", "sun glasses": "sunglasses", "lap top": "laptop",
              "desk chair": "desk_chair", "office chair": "office_chair", "bow tie": "bow_tie", "teddy bare": "teddy_bear",
              "jelly been": "jelly_bean", "jelly beans": "jelly_bean", "office chair": "chair", "side table": "side_table", "coffee mug": "mug", "jelly beans": "jelly_bean", "teddy bears": "teddy_bear"}
PLURAL_SPECIAL = {"leaves": "leaf", "berries": "berry", "cherries": "cherry", "strawberries": "strawberry",
                  "boxes": "box", "glasses": "glass", "brushes": "brush", "dice": "dice", "fish": "fish",
                  "fishes": "fish", "candies": "candy", "puppies": "puppy", "peaches": "peach", "dominoes": "domino",
                  "bunnies": "bunny", "cookies": "cookie", "lillies": "lily", "lilies": "lily", "daisies": "daisy",
                  "potatoes": "potato", "tomatoes": "tomato", "scarves": "scarf"}
GENERIC = {"toy": "toy", "toys": "toy", "food": "food", "snack": "food", "snacks": "food", "treat": "food",
           "treats": "food", "fruit": "fruit", "fruits": "fruit", "drink": "drink", "drinks": "drink",
           "veggie": "food", "vegetable": "food", "veggies": "food", "vegetables": "food"}
GENERIC_PLURAL = ["toys", "snacks", "treats", "fruits", "drinks", "veggies", "vegetables"]
ANYN = {"won": 0, "thing": 0, "object": 0, "item": 0, "one": 0, "stuff": 1, "things": 1, "objects": 1, "items": 1, "ones": 1}
PLACES = {"table": "table", "desk": "desk", "counter": "counter", "floor": "floor", "ground": "floor", "shelf": "shelf",
          "box": "box", "chair": "chair", "basket": "basket", "bowl": "bowl", "bucket": "bucket", "lap": "lap",
          "head": "head", "bed": "bed", "rug": "rug", "blanket": "blanket", "plate": "plate", "bag": "bag",
          "stool": "stool", "bench": "bench", "seat": "seat", "pillow": "pillow", "mat": "mat", "tray": "tray",
          "cup": "cup", "nose": "nose", "pot": "pot", "jar": "jar", "chest": "chest", "toybox": "box"}
PAWS = {"paw": 1, "hand": 1, "arm": 1, "paws": 2, "hands": 2, "arms": 2, "pod": 1, "pause": 2, "pa": 1, "palm": 1,
        "side": 1, "one": 1, "hoof": 1, "hooves": 2, "flipper": 1, "fin": 1, "pore": 1, "pour": 1, "paul": 1}

# ---------------------------------------------------------------------------------------------- moves
# [pattern, name, set{}]   ('$lr' etc. come from captures; mods add the rest). Order = priority.
DET = "?a/an/the/another/some/one"
MOVES = [
    # turning
    ["turn/spin around and face me/forward/front", "turn_back"],
    ["turn and face me/forward/front", "turn_back"],
    ["turn/spin around and face away", "turn_away"],
    ["turn back ?around/round ?to/toward/towards ?me/front/forward", "turn_back"],
    ["turn back to the front", "turn_back"],
    ["turn ?around/round to/and face me/forward/front/us", "turn_back"],
    ["turn toward/towards/to me/us", "turn_back"],
    ["turn toward/towards the camera/front", "turn_back"],
    ["turn to the front", "turn_back"],
    ["turn this way", "turn_back"],
    ["face me/forward/front/us/frontward", "turn_back"],
    ["face this way", "turn_back"],
    ["face the camera/front/screen", "turn_back"],
    ["come back ?here [to me] ?now", "turn_back"],
    ["talk/speak to/with me/us ?again", "turn_back"],
    ["turn ?around/round/back and talk/speak to me/us", "turn_back"],
    ["pay attention to me/us", "turn_back"],
    ["look forward/front/ahead again", "turn_back"],
    ["turn/spin ?around/round in/like a circle", "spin"],
    ["turn/go around/round and around/round", "spin", {"count": 3}],
    ["spin/twirl around/round and around/round", "spin", {"count": 3}],
    ["turn ?yourself [all the way] around/round <N> times", "spin", {"count": "$N"}],
    ["turn all the way around/round", "spin"],
    ["turn in a circle", "spin"],
    ["turn/spin yourself around/round", "spin"],
    ["spin ?around/round/about", "spin"],
    ["twirl ?around/round/about", "twirl"],
    # intent c2 (fresh-set generalization): more verbs / frames for the same moves
    ["whirl ?yourself ?around/round/about", "spin"],
    ["go round/around in a circle", "spin"],
    ["spin/turn/whirl ?yourself all the way around/round", "spin"],
    ["turn back ?around/round to/and face me/us/forward/front", "turn_back"],
    ["turn back ?around/round toward/towards/to the user/camera/person/front", "turn_back"],
    ["turn back ?around/round to say hi/hello", "turn_back"],
    ["turn back ?around/round and say hi/hello", "turn_back"],
    ["come ?and talk/speak to/with me/us", "turn_back"],
    ["come back and talk/speak to/with me/us", "turn_back"],
    ["turn around to/and look at me/us", "turn_back"],
    ["jump/leap/launch/go/spring into ?a/an @mn", "$mn"],
    ["move/go/head/walk ?straight forward/forwards/ahead", "walk", {"dir": "forward"}],
    ["move/go/head/walk ?straight back/backward/backwards", "walk", {"dir": "back"}],
    ["move/head/scoot ?yourself", "walk"],
    ["get on your feet", "stand"],
    ["get ?back ?up on your feet", "stand"],
    ["stand ?up straight/tall", "stand"],
    ["plop/plonk/plunk/settle/flop/sink/park ?yourself down", "sit"],
    ["sit yourself ?down", "sit"],
    ["grab/find a seat", "sit"],
    ["pirouette", "twirl"],
    ["do/make ?a/an ?full 360", "spin"],
    ["turn away [from me|from us|from the camera]", "turn_away"],
    ["turn ?around/round away [from me|from us]", "turn_away"],
    ["keep facing/looking away", "turn_away", {"hold": True}],
    ["keep your back to me/us", "turn_away", {"hold": True}],
    ["stay turned away/around", "turn_away", {"hold": True}],
    ["turn/face the other way", "turn_away"],
    ["face away [from me|from us|from the camera]", "turn_away"],
    ["turn your back [to me|on me|to us|on us]", "turn_away"],
    ["show me/us your back ?side", "turn_away"],
    ["let me/us see your back", "turn_away"],
    ["turn ?yourself around/round", "turn_away"],
    ["turn/turns/rotate/face/swivel ?your ?body/self/whole ?body", "turn"],
    ["do/make a ?little/quick/full turn", "spin"],
    ["look/turn sideways", "turn", {"dir": "left"}],
    # flips
    ["do/make ?a/an back flip/flips", "backflip"],
    ["back flip/flips", "backflip"],
    ["flip/flips back/backward/backwards", "backflip"],
    ["flip over backwards/backward", "backflip"],
    ["front flip/flips", "frontflip"],
    ["flip/flips forward/forwards/frontwards", "frontflip"],
    ["do/make ?a/an front flip", "frontflip"],
    ["forward roll", "frontflip"],
    ["flip ?over ?around", "flip"],
    ["cart wheel/wheels", "cartwheel"],
    # jumps
    ["jump up and down", "jump"],
    ["jump for joy", "jump"],
    ["jump ?up/around [in the air]", "jump"],
    ["jump/jumps/leap/leaps ?up/around/about", "jump"],
    ["hop up and down", "hop"],
    ["hop ?up/around/about", "hop"],
    ["bounce up and down", "bounce"],
    ["bounce ?up/around/about", "bounce"],
    # walking
    ["take/make <N> steps", "walk", {"distance": "$N"}],
    ["walk/go <N> steps", "walk", {"distance": "$N"}],
    ["take/make ?a/one ?little/small/tiny/big ?baby step", "step"],
    ["step", "step"],
    ["go for/on a ?little/short/quick walk ?around", "walk"],
    ["go for a little/short walk", "walk"],
    ["take a ?little/short/quick walk", "walk"],
    ["walk/stroll ?around/about/over ?here", "walk"],
    ["come to me/us", "walk", {"dir": "forward"}],
    ["march ?around/about", "walk", {"style": "march"}],
    ["tiptoe ?around/about", "walk", {"style": "tiptoe"}],
    ["run/jog ?around/about", "walk", {"speed": "fast"}],
    ["go for a run", "walk", {"speed": "fast"}],
    # dancing
    ["do the @dstyle", "dance", {"style": "$dstyle"}],
    ["do the @dstyle dance", "dance", {"style": "$dstyle"}],
    ["dance the @dstyle", "dance", {"style": "$dstyle"}],
    ["dance ?around/about", "dance"],
    ["moonwalk", "dance", {"style": "moonwalk"}],
    ["moon walk", "dance", {"style": "moonwalk"}],
    ["floss", "dance", {"style": "floss"}],
    ["do the moon walk", "dance", {"style": "moonwalk"}],
    ["boogie ?down", "dance"],
    ["groove", "dance"],
    ["bust a move", "dance"],
    ["get your groove on", "dance"],
    ["show me/us your ?best ?dance moves", "dance"],
    ["show me/us some ?dance moves", "dance"],
    ["show me/us what you got", "dance"],
    # postures
    ["sit ?down/up", "sit"],
    ["sit back down", "sit"],
    ["take/have a seat", "sit"],
    ["stand ?up [on your feet|on your legs|on your hind legs|on your back legs|on your two feet|on two feet]", "stand"],
    ["stand up tall", "stand"],
    ["get up [on your feet]", "get_up"],
    ["get back up", "get_up"],
    ["rise ?up", "get_up"],
    ["lie/lay/lye down on your back", "lie_down", {"style": "back"}],
    ["lie/lay/lye on your back", "lie_down", {"style": "back"}],
    ["lie/lay/lye ?down on your tummy/belly/stomach/front", "lie_down", {"style": "front"}],
    ["lie/lay/lye ?down ?flat [on the floor|on the ground]", "lie_down"],
    ["lie/lay/lye flat", "lie_down"],
    ["play dead", "lie_down", {"style": "play_dead"}],
    ["roll over/around/about", "roll_over"],
    ["do/make ?a/an barrel roll", "roll_over"],
    ["roll", "roll_over"],
    ["bow ?down", "bow"],
    ["take/do/make/give ?us/me a ?deep/little/big/quick/nice bow", "bow"],
    ["curtsy/curtsey", "bow", {"style": "curtsy"}],
    ["take/do/make ?us/me a curtsy/curtsey", "bow", {"style": "curtsy"}],
    # wiggles
    ["shake ?it/yourself off", "shake_off"],
    ["shake like a wet dog", "shake_off"],
    ["shake ?your ?whole booty/butt/bottom/bum/hips/tush/body ?around", "wiggle"],
    ["wiggle/wobble/jiggle/waggle ?your ?whole booty/butt/bottom/bum/hips/tush/body ?around", "wiggle"],
    ["wiggle/wobble/jiggle/waggle ?around/about", "wiggle"],
    ["wiggle wiggle", "wiggle"],
    # a move as a noun: "do a backflip", "two backflips", "how about a spin", "another cartwheel"
    ["do/make/perform/try/attempt/doing ?<N>/a/an/another/some/the ?@madj ?@madj ?@madj @mn", "$mn", {"count": "$N"}],
    ["take/have/throw/bust/pull/nail/land/rock ?out/off ?<N>/a/an/another/some/the ?@madj ?@madj ?@madj @mn", "$mn", {"count": "$N"}],
    ["<N>/a/an/another/some/the ?@madj ?@madj ?@madj @mn", "$mn", {"count": "$N"}],
    ["?@madjb ?@madjb @mn", "$mn"],
    ["do/make/perform/try/attempt/doing ?<N>/a/an/another/some/the ?@madj ?@madj ?@madj @mnstyle dance", "dance", {"style": "$mnstyle", "count": "$N"}],
    ["<N>/a/an/another/some/the ?@madj ?@madj ?@madj @mnstyle dance", "dance", {"style": "$mnstyle", "count": "$N"}],
    ["@mnstyle dance", "dance", {"style": "$mnstyle"}],
    ["?do/make/perform/try/attempt/doing one more @mn", "$mn"],
]
# move nouns ("do a <mn>") -> name
MN = {"spin": "spin", "spins": "spin", "twirl": "twirl", "twirls": "twirl", "pirouette": "twirl", "pirouettes": "twirl",
      "360": "spin", "backflip": "backflip", "backflips": "backflip",
      "frontflip": "frontflip", "frontflips": "frontflip", "flip": "flip", "flips": "flip", "somersault": "frontflip",
      "somersaults": "frontflip", "summersault": "frontflip", "cartwheel": "cartwheel", "cartwheels": "cartwheel",
      "jump": "jump", "jumps": "jump", "leap": "jump", "leaps": "jump", "hop": "hop", "hops": "hop", "bounce": "bounce",
      "bounces": "bounce", "dance": "dance", "dances": "dance", "jig": "dance", "boogie": "dance", "bow": "bow",
      "bows": "bow", "curtsy": "bow", "wiggle": "wiggle", "wiggles": "wiggle", "wobble": "wiggle", "roll": "roll_over",
      "rolls": "roll_over", "walk": "walk", "step": "step", "steps": "step", "backflop": "backflip",
      "handspring": "frontflip", "back flip": "backflip", "front flip": "frontflip", "cart wheel": "cartwheel",
      "summer salt": "frontflip", "circle": "spin", "circles": "spin", "rotation": "spin", "rotations": "spin", "whirl": "spin", "whirls": "spin", "three sixty": "spin", "some salt": "frontflip", "car wheel": "cartwheel", "moonwalk": "dance", "twirly": "twirl", "flippy": "flip", "spinny": "spin", "dancey": "dance"}
# adjectives inside a move noun phrase: -> assignments (filtered per move later)
MADJ = {"double": {"count": 2}, "triple": {"count": 3}, "quadruple": {"count": 4},
        "big": {"height": 0.8}, "huge": {"height": 0.8}, "giant": {"height": 0.8}, "high": {"height": 0.8},
        "tall": {"height": 0.8}, "massive": {"height": 0.8}, "enormous": {"height": 0.8}, "great": {},
        "little": {"height": 0.3}, "small": {"height": 0.3}, "tiny": {"height": 0.3}, "mini": {"height": 0.3},
        "teeny": {"height": 0.3}, "baby": {"height": 0.3}, "low": {"height": 0.3},
        "quick": {"speed": "fast"}, "fast": {"speed": "fast"}, "speedy": {"speed": "fast"}, "slow": {"speed": "slow"},
        "slowmotion": {"speed": "slow"},
        "backward": {"_flip": "backflip"}, "backwards": {"_flip": "backflip"}, "back": {"_flip": "backflip"},
        "forward": {"_flip": "frontflip"}, "front": {"_flip": "frontflip"},
        "full": {}, "good": {}, "nice": {}, "real": {}, "proper": {}, "perfect": {}, "cool": {}, "awesome": {},
        "super": {}, "fun": {}, "little bit": {}, "more": {}, "single": {"count": 1}, "one": {},
        "silly": {"style": "silly"}, "happy": {"style": "happy"}, "funny": {"style": "funny"},
        "crazy": {"style": "crazy"}, "wild": {"style": "wild"}, "goofy": {"style": "goofy"}, "fancy": {"style": "fancy"},
        "funky": {"style": "funky"}, "wiggly": {"style": "wiggly"}, "robot": {"style": "robot"},
        "disco": {"style": "disco"}, "victory": {"style": "victory"}, "excited": {"style": "excited"},
        "sad": {"style": "sad"}, "spooky": {"style": "spooky"}, "graceful": {"style": "graceful"},
        "ballet": {"style": "ballet"}, "chicken": {"style": "chicken"}, "dizzy": {"style": "dizzy"},
        "flippy": {"_k": 1}, "twirly": {"_k": 1}, "spinny": {"_k": 1}, "jumpy": {"_k": 1}, "dancey": {"_k": 1},
        "deep": {"_k": 1}, "graceful": {"style": "graceful"},
        "really slow": {"speed": 0.4}, "super slow": {"speed": 0.4}, "very slow": {"speed": 0.4},
        "really fast": {"speed": 2.5}, "super fast": {"speed": 2.5}, "very fast": {"speed": 2.5}, "really quick": {"speed": 2.5},
        "really high": {"height": 1}, "really tall": {"height": 1}, "super tall": {"height": 1}, "complete": {}, "whole": {}, "ballerina": {}, "elegant": {}, "epic": {}, "amazing": {}, "standing": {}, "up and down": {}, "single": {"count": 1}, "super high": {"height": 1}, "really big": {"height": 1}, "super big": {"height": 1}}
DSTYLE = {w: w for w in ["robot", "twist", "floss", "macarena", "moonwalk", "chicken", "salsa", "tango", "waltz", "hula",
                         "disco", "ballet", "worm", "sprinkler", "conga", "limbo", "jig", "polka", "cancan", "shuffle",
                         "dab", "charleston", "mashed", "monkey", "penguin", "robot", "twist", "boogie", "jive",
                         "foxtrot", "rumba", "samba", "breakdance", "cha", "swim", "mambo", "running"]}
DSTYLE.update({"hokey": "hokey_pokey", "hip": "hip_hop", "chacha": "cha_cha", "cha cha": "cha_cha",
               "hokey pokey": "hokey_pokey", "hip hop": "hip_hop", "chicken dance": "chicken", "bunny hop": "bunny_hop",
               "electric slide": "electric_slide", "running man": "running_man", "moon walk": "moonwalk",
               "happy dance": "happy", "snoopy dance": "snoopy", "potty dance": "potty", "rain dance": "rain",
               "can can": "cancan", "macarena dance": "macarena"})
MNSTYLE = {w: w for w in ["silly", "happy", "funny", "crazy", "wild", "goofy", "fancy", "funky", "wiggly", "robot",
                          "disco", "victory", "excited", "sad", "spooky", "graceful", "ballet", "chicken", "snoopy",
                          "rain", "potty", "celebration", "birthday", "bedtime", "jungle", "monkey", "bear", "bunny",
                          "rabbit", "fox", "hippo", "penguin", "pirate", "cowboy", "fairy", "princess", "superhero",
                          "dinosaur", "zombie", "jelly", "wobbly", "tap", "hula", "salsa", "tango", "irish", "square",
                          "line", "belly", "break", "dizzy", "magic",
                          "wiggle", "shimmy", "jiggle", "bouncy", "boogie", "party", "rock", "pop", "star", "space",
                          "underwater", "snow", "spring", "tiny"]}

# ---------------------------------------------------------------------------------------------- objects
# [pattern, verb, flags] flags: n = needs a describing np (not a bare pronoun) unless a modifier follows,
#                               N = needs a describing np always, b = the object may be omitted (-> it),
#                               set = extra assignments
OBJ = [
    # worn items
    ["put/slip/pop/stick/get <O> on [your ears|my ears|your head ears]", "put_on", ""],
    ["put/slip/pop/stick/get/place/set/plop <O> on/onto/over your/my head/ears", "put_on", ""],   # intent c2
    ["<O> on your head/ears", "put_on", "N"],
    ["put/slip/pop on <O>", "put_on", "N"],
    ["try <O> on", "put_on", ""],
    ["try on <O>", "put_on", "N"],
    ["wear <O>", "put_on", ""],
    ["take/pull/slip/get <O> off [your head|my head|your ears]", "take_off", ""],
    ["lift/pull/take/slip/get <O> off ?of your/my head/ears", "take_off", ""],   # intent c2
    ["take/pull/slip off <O>", "take_off", "N"],
    ["remove <O>", "take_off", ""],
    # furniture / sliding (target mods: me / closer / back / away / <thing>)
    ["pull/drag/tug/yank/haul ?up/out/over <O>", "pull", ""],
    ["pull/drag/tug/yank/haul <O> up/out/over", "pull", ""],
    ["push/shove/nudge <O> ?over/in", "push", ""],
    ["push/shove/move/slide/scoot back <O>", "push", "N", {"target": "back"}],   # intent c2
    ["push/shove/move/slide away <O>", "push", "N", {"target": "away"}],
    ["get <O>", "bring", "", {"_needtarget": 1}],
    ["bring/carry <O> over", "bring", ""],
    ["bring over <O>", "bring", "N"],
    ["bring <O>", "bring", "", {"_needtarget": 1}],
    ["slide/scoot/shift/inch <O> ?over/up", "slide", ""],
    ["give <O> a ?little/big/good/hard push/shove/nudge", "push", ""],
    ["give <O> a ?little/big/good/hard pull/tug/yank", "pull", ""],
    # sitting / getting up with a thing
    ["sit/sitting/go ?down/back/yourself in/on/into/onto <O>", "sit_in", "n"],
    ["sit/plop/plonk/plunk/settle/flop/sink/perch/park/climb/hop/get ?yourself ?down/back ?yourself ?down in/on/into/onto <O>", "sit_in", "N"],   # intent c2
    ["grab/find a seat in/on <O>", "sit_in", "N"],
    ["grab/find a seat at/by <O>", "sit_at", "N"],
    ["sit/plop/settle ?yourself ?down ?right at/by <O>", "sit_at", "N"],
    ["sit/sitting ?down/back at/by <O>", "sit_at", "n"],
    ["sit back down at/by <O>", "sit_at", "n"],
    ["sit back down in/on <O>", "sit_in", "n"],
    ["sit ?down ?back up to <O>", "sit_at", "N"],
    ["take/have a seat in/on <O>", "sit_in", "N"],
    ["take/have a seat at/by <O>", "sit_at", "N"],
    ["get/stand/hop/climb/jump up from/off <O>", "stand_up", "N"],
    ["get/stand/hop/climb/jump ?up out of <O>", "stand_up", "N"],
    ["get/hop/climb/jump off ?of <O>", "stand_up", "N"],
    ["leave <O>", "stand_up", "N"],
    # open / close
    ["open/opening ?up <O> ?up", "open", ""],
    ["flip/pop <O> open", "open", ""],
    ["flip/pop open <O>", "open", "N"],
    ["open the lid of/on <O>", "open", ""],
    ["close/shut/closing ?up <O> ?up", "close", ""],
    ["close/shut the lid of/on <O>", "close", ""],
    ["fold <O> shut/closed/up", "close", ""],
    # using / working / typing
    ["get ?back to work [on <O>|at <O>]", "work_on", "b0"],
    ["go ?back to work [on <O>|at <O>]", "work_on", "b0"],
    ["start/begin/keep/continue working [on <O>|at <O>]", "work_on", "b0"],
    ["do some/your/more work [on <O>|at <O>]", "work_on", "b0"],
    ["do a little/bit ?of work [on <O>|at <O>]", "work_on", "b0"],
    ["back to work [on <O>|at <O>]", "work_on", "b0"],
    ["get some work done [on <O>|at <O>]", "work_on", "b0"],
    ["work on/at/with <O>", "work_on", ""],
    ["get typing [on <O>|at <O>|into <O>]", "type_on", "b0"],   # intent c2
    ["get ?started working [on <O>|at <O>]", "work_on", "b0"],
    ["get started [on <O>|at <O>|with <O>]", "work_on", "b0"],
    ["get started typing [on <O>|at <O>]", "type_on", "b0"],
    ["get busy [on <O>|at <O>|with <O>]", "work_on", "b0"],
    ["work", "work_on", "b0"],
    ["start/begin/keep/continue typing [on <O>|into <O>|at <O>]", "type_on", "b0"],
    ["do some typing [on <O>|at <O>]", "type_on", "b0"],
    ["type ?something/stuff/away/more [on <O>|into <O>|at <O>]", "type_on", "b0"],
    ["type on/into/at <O>", "type_on", ""],
    ["use/using <O>", "use", "n"],
    ["play with <O>", "use", ""],
    ["stop/quit working/typing/work [on <O>|at <O>]", "stop", "b0"],
    ["stop/quit using <O>", "stop", ""],
    ["stop/quit playing with <O>", "stop", ""],
    ["stop what you are doing", "stop", "b0"],
    ["pause/quit/finish ?using/with <O>", "stop", "N"],   # intent c2
    ["pause/finish working/typing [on <O>|at <O>]", "stop", "b0"],
    # hand-over (destination paw <S>)
    ["switch/swap/change/trade ?it/them ?to ?your ?the hands/paws/arms/sides/hand/paw", "hand_over", "b"],
    ["switch/swap/change/trade ?it/them over", "hand_over", "b"],
    ["switch/swap/change/trade hands/paws/arms/sides with <O>", "hand_over", ""],
    ["switch/swap/change/trade <O> to/into <S>", "hand_over", ""],
    ["give/pass/hand/move/put/switch/transfer/toss/throw/shift/swap <O> ?over/across <S>", "hand_over", ""],
    ["pass/hand <O> over/across", "hand_over", ""],
    ["give/pass/hand/move/put/switch/transfer/toss/throw/shift/swap/flip <O> from one/your @paw to/into ?the/your other ?one/@paw", "hand_over", ""],   # intent c2
    ["give/pass/hand <S> <O>", "hand_over", ""],
    # offer (to the user)
    ["give/pass/hand/offer/toss/throw <O> ?over ?here/out ?to me/us", "offer", "_to"],
    ["give/pass/hand/offer/bring/toss/throw me/us <O>", "offer", ""],
    ["hold <O> out [to me|to us|toward me|towards me]", "offer", ""],
    ["hold out <O>", "offer", ""],
    ["can/may/could i ?please have/get/hold <O>", "offer", ""],
    ["let me have/hold <O>", "offer", ""],
    ["hand <O> over", "offer", ""],
    ["give <O> here", "offer", ""],
    # show / inspect
    ["show/showing me/us <O>", "show", ""],
    ["show ?off <O> ?off", "show", ""],
    ["hold <O> up to/for/toward/towards the camera/screen", "show", ""],
    ["hold up <O> to/for/toward/towards the camera/screen", "show", ""],
    ["let/lemme me/us see <O>", "show", ""],
    ["let/lemme me/us look at <O>", "show", ""],
    ["can/could/may i/we ?please see <O>", "show", ""],
    ["can/could/may i/we ?please look at <O>", "show", ""],
    ["show me/us the back/bottom/underside/top/inside/front of <O>", "turn", "", {"amount": 0.5}],
    ["show me/us the other side of <O>", "turn", "", {"amount": 0.5}],
    ["show me/us the back/bottom/underside ?side", "turn", "b", {"amount": 0.5}],
    ["show me/us the other side", "turn", "b", {"amount": 0.5}],
    ["give <O> a quarter turn/spin", "turn", "", {"amount": 0.25}],
    ["give <O> a half turn/spin", "turn", "", {"amount": 0.5}],
    ["give <O> a full/whole turn/spin", "turn", "", {"amount": 1}],
    ["give <O> a ?little turn/spin/twirl", "turn", ""],
    ["take/have ?a ?closer/good/close/quick look at <O>", "inspect", ""],
    ["look ?closely/closer/carefully at <O>", "inspect", "N"],
    ["look at <O> ?really closely/closer/carefully", "inspect", "N"],
    ["look at <O>", "inspect", "n"],
    ["check/checking out <O>", "inspect", ""],
    ["check <O> out", "inspect", ""],
    ["inspect/examine/study/investigate/observe <O>", "inspect", ""],
    # turning / flipping it
    ["turn/rotate/spin/twirl <O> upside down", "flip", ""],
    ["hold <O> upside down", "flip", ""],
    ["flip <O> over/around", "flip", ""],
    ["turn <O> over", "flip", ""],
    ["flip/turn over <O>", "flip", ""],
    ["flip <O>", "flip", "n"],
    ["turn/rotate <O> around/round", "turn", "", {"amount": 0.5}],
    ["turn around <O>", "turn", "N", {"amount": 0.5}],
    ["spin/twirl <O> ?around/round", "turn", "", {"amount": 1}],
    ["rotate <O>", "turn", ""],
    ["turn <O>", "turn", "n"],
    # tossing
    ["toss/throw/flip/chuck <O> ?up ?high/higher [in the air|into the air]", "toss", ""],
    ["toss/throw/chuck <O> up", "toss", ""],
    ["toss/throw/chuck up <O>", "toss", "N"],
    ["toss <O>", "toss", ""],
    ["toss/throw/chuck ?up", "toss", ""],   # intent c2: "... toss then put it back" (the held one)
    ["throw <O>", "toss", ""],
    ["juggle <O>", "toss", ""],
    ["juggle", "toss", "b"],
    ["bounce <O> [on the floor|on the ground|on the table]", "toss", ""],
    ["play catch ?with <O>", "toss", "b"],
    ["catch <O>", "catch", ""],
    ["catch", "catch", "b"],
    # shake / hug / smell
    ["give <O> a ?little/good/big shake/shaking/rattle/wiggle", "shake", ""],
    ["shake/rattle/wiggle/jiggle/waggle <O> ?around/about/up", "shake", ""],
    ["wave <O> around/about", "shake", ""],
    ["wave <O>", "shake", ""],
    ["give <O> a ?big/little/huge/nice hug/squeeze/cuddle/kiss/snuggle", "hug", ""],
    ["hug/cuddle/squeeze/snuggle/kiss <O> ?tight/tightly/close", "hug", ""],
    ["cuddle/snuggle with <O>", "hug", ""],
    ["hold <O> close/tight/tightly", "hug", ""],
    ["give <O> a ?little/big/good sniff/smell/whiff", "smell", ""],
    ["take/have a ?little/big/good sniff/smell/whiff of <O>", "smell", ""],
    ["smell/sniff <O>", "smell", ""],
    # hold up
    ["hold/lift/raise/put/stick <O> up high/higher", "hold_up", ""],
    ["hold/lift/raise/put/stick <O> ?up ?high in/into the air/sky", "hold_up", ""],
    ["hold/lift/raise <O> ?up over/above your head", "hold_up", ""],
    ["hold up <O>", "hold_up", ""],
    ["hold <O> up", "hold_up", ""],
    ["raise ?up <O>", "hold_up", ""],
    ["lift <O> up high", "hold_up", ""],
    # putting things down / back / on a place
    ["put/set/lay/place <O> back down", "put_down", ""],
    ["put/set/lay/place <O> down", "put_down", ""],
    ["put/set/lay/place down <O>", "put_down", "N"],
    ["put/set/lay/place <O> back", "put_back", ""],
    ["put/set/lay/place down", "put_down", ""],   # intent c2: "... and put down" (the held one)
    ["put/set/place back", "put_back", ""],
    ["put/place back <O>", "put_back", "N"],
    ["put/set/place/tuck <O> away", "put_back", ""],
    ["put away <O>", "put_back", "N"],
    ["return/replace <O>", "put_back", ""],
    ["bring <O> back", "put_back", ""],
    ["clean/tidy up ?<O>", "put_back", "b", {"_ref": "all"}],
    ["clean/tidy <O> up", "put_back", ""],
    ["move <O>", "slide", "", {"_needtarget": 1}],
    ["put/set/place/move/leave/stick/stand/balance/lay/plop <O>", "place", "", {"_needplace": 1}],
    ["<O> on", "put_on", "N"],
    ["<O> off", "take_off", "N"],
    ["drop <O>", "drop", ""],
    ["let go of <O>", "drop", ""],
    ["let <O> go/fall/drop", "drop", ""],
    ["let go", "drop", "b"],
    # picking up (last: the most generic verbs)
    ["pick/picks/picked/pickup up <O>", "pick_up", ""],
    ["pick <O> up", "pick_up", ""],
    ["pickup <O>", "pick_up", ""],
    ["pick up", "pick_up", "b"],
    ["grab/grabbed ahold/hold of <O>", "pick_up", ""],
    ["get/take ahold/hold of <O>", "pick_up", ""],
    ["grab/grap/crab <O>", "pick_up", ""],
    ["lift ?up <O>", "pick_up", ""],
    ["lift <O> up", "pick_up", ""],
    ["hold onto/on to <O>", "pick_up", ""],
    ["take/get/fetch/carry/collect <O> ?up", "pick_up", "n"],
    ["go get <O>", "pick_up", ""],
    ["pick <O>", "pick_up", "N"],
    ["hold <O>", "pick_up", "n"],
]

# ---------------------------------------------------------------------------------------------- tail modifiers
# [pattern, kind (m = moves, o = objects, mo = both), set{}, only?]  kind of value "$N" = the captured number etc.
MODS = [
    # counts
    ["<N> times", "mo", {"count": "$N"}],
    ["<N> more times", "mo", {"count": "$N"}],
    ["<N> time", "mo", {"count": "$N"}],
    ["once", "mo", {"count": 1}], ["one time", "mo", {"count": 1}], ["twice", "mo", {"count": 2}],
    ["thrice", "mo", {"count": 3}], ["two times", "mo", {"count": 2}],
    ["over and over ?again", "mo", {"count": 5}], ["again and again", "mo", {"count": 5}],
    ["around/round and around/round", "m", {"count": 3}, ["spin", "twirl", "walk", "dance"]],
    ["in a row", "mo", {}],
    ["one after the other/another", "mo", {}], ["back to back", "mo", {}],   # intent c2
    ["just one", "mo", {"count": 1}], ["just once", "mo", {"count": 1}],
    ["?over backward/backwards", "m", {"_flip": "backflip"}, ["flip", "backflip", "frontflip"]],
    ["?over forward/forwards", "m", {"_flip": "frontflip"}, ["flip", "backflip", "frontflip"]],
    ["way up ?high/there", "m", {"height": 1}, ["jump", "hop", "bounce", "backflip", "frontflip", "flip"]],
    ["up there", "m", {"height": 0.8}, ["jump", "hop", "bounce"]],
    ["straight ?ahead", "m", {"dir": "forward"}, ["walk", "step"]],
    ["gracefully/elegantly/beautifully/nicely/neatly/quietly", "mo", {}],
    ["back", "o", {}, ["toss", "catch", "offer", "show", "hold_up", "hug"]],
    ["over", "m", {}, ["walk", "hop", "jump", "step", "bounce", "roll_over", "cartwheel", "run"]],
    ["?over/up/right to/toward/towards ?the/your <O>", "m", {"target": "$O"}, ["walk", "step", "hop", "jump", "bounce", "cartwheel", "roll_over"]],
    ["for a second/sec/moment", "mo", {}],
    ["for the audience/camera/crowd/people/kids/family", "mo", {}],
    ["<N>", "m", {"count": "$N"}, ["spin", "twirl", "backflip", "frontflip", "flip", "cartwheel", "jump", "hop", "bounce"]],
    # direction
    ["?to/toward/towards/on ?the/your @lr ?side", "mo", {"dir": "$lr"}],
    ["to/toward/towards/on my @lr ?side", "mo", {"dir": "$slr"}],
    ["@lr ward/wards", "mo", {"dir": "$lr"}],
    ["leftward/leftwards", "mo", {"dir": "left"}], ["rightward/rightwards", "mo", {"dir": "right"}],
    ["clockwise", "mo", {"dir": "cw"}],
    ["clock wise", "mo", {"dir": "cw"}],
    ["counterclockwise/anticlockwise", "mo", {"dir": "ccw"}],
    ["counter/anti clockwise", "mo", {"dir": "ccw"}],
    ["the other way ?around/round", "mo", {"dir": "ccw"}, ["spin", "twirl"]],
    ["toward/towards/to me/us", "m", {"dir": "forward"}, ["walk", "step", "jump", "hop", "bounce", "roll_over", "cartwheel"]],
    ["toward/towards the camera/screen", "m", {"dir": "forward"}],
    ["forward/forwards/ahead/closer/frontward/frontwards", "m", {"dir": "forward"}],
    ["closer to me/us", "m", {"dir": "forward"}],
    ["over here", "m", {"dir": "forward"}, ["walk", "step", "jump", "hop", "bounce", "roll_over", "cartwheel"]],
    ["backward/backwards/back", "m", {"dir": "back"}, ["walk", "step", "jump", "hop", "bounce", "roll_over", "cartwheel", "spin", "twirl"]],
    ["away [from me|from us|from the camera]", "m", {"dir": "back"}, ["walk", "step", "jump", "hop", "bounce", "roll_over", "cartwheel"]],
    ["back and forth", "m", {}, ["walk", "dance", "wiggle", "step"]],
    ["side to side", "m", {}, ["walk", "dance", "wiggle", "step"]],
    ["sideways/aside", "m", {"dir": "left"}, ["step", "walk", "hop", "jump"]],
    ["to the side", "m", {"dir": "left"}, ["step", "walk", "hop", "jump", "turn"]],
    ["up", "mo", {}],
    ["upward/upwards", "m", {"dir": "up"}, ["jump", "hop", "bounce"]],
    ["around/round/about", "m", {}],
    ["in a circle", "m", {"style": "circle"}, ["walk", "dance", "hop", "jump", "bounce", "roll_over"]],
    ["in a circle", "m", {}, ["spin", "twirl"]],
    ["around the room/place/table", "m", {"style": "circle"}, ["walk", "dance", "hop", "jump", "bounce"]],
    ["in place/position", "m", {"distance": 0}, ["walk", "step", "jump", "hop", "bounce", "dance"]],
    ["on the spot", "m", {"distance": 0}, ["walk", "step", "jump", "hop", "bounce", "dance", "spin", "twirl"]],
    ["on one/1 foot/leg/paw", "m", {"style": "one_foot"}, ["hop", "jump", "stand", "bounce", "spin", "twirl"]],
    ["on your tippy/tip toes/toe", "m", {"style": "tiptoe"}, ["walk", "dance", "stand", "spin", "twirl"]],
    ["on tiptoe/tiptoes", "m", {"style": "tiptoe"}, ["walk", "dance", "stand", "spin", "twirl"]],
    ["on the floor/ground/rug/bed", "m", {}, ["sit", "lie_down", "roll_over", "stand", "dance", "jump", "hop", "bounce"]],
    ["in the air", "m", {}, ["jump", "hop", "bounce", "backflip", "frontflip", "flip", "spin", "twirl"]],
    ["on your tummy/belly/stomach/front", "m", {"style": "front"}, ["lie_down", "roll_over"]],
    ["on your back", "m", {"style": "back"}, ["lie_down", "roll_over"]],
    # speed
    ["slowly/slow/gently/carefully/softly/slowmotion", "mo", {"speed": "slow"}],
    ["in slow motion/mo", "mo", {"speed": "slow"}],
    ["nice and slow/slowly/easy/gentle", "mo", {"speed": "slow"}],
    ["really/very/super/extra/real/so slowly/slow", "mo", {"speed": 0.4}],
    ["as slow/slowly as you can", "mo", {"speed": 0.4}],
    ["quickly/quick/fast/faster/rapidly/speedy", "mo", {"speed": "fast"}],
    ["real quick/fast", "mo", {"speed": "fast"}],
    ["really/very/super/extra/so fast/quickly/quick/faster", "mo", {"speed": 2.5}],
    ["as fast/quick/quickly as you can/possible", "mo", {"speed": 2.5}],
    ["lightning fast", "mo", {"speed": 2.5}],
    ["like lightning/crazy", "m", {"speed": 2.5}, ["spin", "twirl", "walk", "dance", "wiggle"]],
    ["at normal speed", "mo", {"speed": "normal"}],
    # height / size (jumps)
    ["high/higher/big/tall", "m", {"height": 0.8}, ["jump", "hop", "bounce", "backflip", "frontflip", "flip"]],
    ["up high", "m", {"height": 0.8}, ["jump", "hop", "bounce", "backflip", "frontflip", "flip"]],
    ["really/very/super/extra/so/real high/big/tall", "m", {"height": 1}, ["jump", "hop", "bounce", "backflip", "frontflip", "flip"]],
    ["as high as you can/possible", "m", {"height": 1}, ["jump", "hop", "bounce", "backflip", "frontflip", "flip"]],
    ["all the way up to the ceiling/sky", "m", {"height": 1}, ["jump", "hop", "bounce"]],
    ["to the ceiling/sky/moon", "m", {"height": 1}, ["jump", "hop", "bounce"]],
    ["a little/bit", "m", {"height": 0.3}, ["jump", "hop", "bounce"]],
    ["a little bit", "m", {"height": 0.3}, ["jump", "hop", "bounce"]],
    ["low/small/little/lightly", "m", {"height": 0.3}, ["jump", "hop", "bounce"]],
    ["a little/bit", "m", {}, ["walk", "dance", "wiggle", "spin", "twirl", "step"]],
    ["a little bit", "m", {}, ["walk", "dance", "wiggle", "spin", "twirl", "step"]],
    # distance
    ["<N> steps ?forward/forwards", "m", {"distance": "$N"}, ["walk", "step"]],
    ["<N> step", "m", {"distance": "$N"}, ["walk", "step"]],
    ["a step", "m", {}, ["step"]],
    ["far [away]", "m", {}, ["walk", "jump", "hop"]],
    # style
    ["like a/an @style", "m", {"style": "$style"}, ["dance", "walk", "jump", "hop", "wiggle", "spin", "twirl", "bow", "roll_over", "sit", "lie_down"]],
    ["like a/an dog/wet/puppy", "m", {}, ["shake_off"]],
    ["like @style", "m", {"style": "$style"}, ["dance", "walk", "wiggle", "spin", "twirl"]],
    ["like crazy", "m", {"style": "crazy"}, ["dance", "wiggle", "jump", "hop", "bounce"]],
    ["@mnstyle style", "m", {"style": "$mnstyle"}, ["dance"]],
    ["to the music/beat/song", "m", {}, ["dance", "wiggle", "jump", "hop", "bounce", "walk", "spin", "twirl"]],
    ["happily", "m", {"style": "happy"}, ["dance", "jump", "hop", "bounce", "spin", "twirl", "wiggle"]],
    ["silly/sillily/goofy", "m", {"style": "silly"}, ["dance", "walk", "wiggle"]],
    ["with me/us", "m", {}, ["dance", "walk", "jump", "hop", "spin", "twirl", "bounce", "sit", "lie_down", "wiggle"]],
    ["with style", "m", {}],
    # gaze
    ["without looking at me/us", "m", {"gaze": "free"}],
    ["while/and looking at me/us", "m", {"gaze": "user"}],
    ["looking at me/us", "m", {"gaze": "user"}],
    ["but/and keep looking at me/us", "m", {"gaze": "user"}],
    ["but/and dont look at me/us", "m", {"gaze": "free"}],
    ["while facing me/us", "m", {"gaze": "user"}],
    ["but keep your eyes on me/us", "m", {"gaze": "user"}],
    ["with your eyes closed", "m", {"gaze": "free"}],
    # hold / duration
    ["until i say stop/so/when", "mo", {"hold": True}],
    ["until i tell you to stop", "mo", {"hold": True}],
    ["and stay [like that|there|that way|put]", "mo", {"hold": True}],
    ["and keep it there/like that", "mo", {"hold": True}],
    ["for <N> seconds/secs/second/sec", "mo", {"duration_ms": "$Ns"}],
    ["for <N> minutes/minute", "mo", {"duration_ms": "$Nm"}],
    ["for a minute", "mo", {"duration_ms": 60000}],
    ["for a while/bit", "mo", {"duration_ms": 10000}],
    ["for a little while/bit", "mo", {"duration_ms": 10000}],
    ["forever", "mo", {"hold": True}],
    # object: furniture / slide targets
    ["?over here", "o", {"target": "me"}, ["pull", "push", "bring", "slide"]],
    ["?over to/toward/towards me/us", "o", {"target": "me"}, ["pull", "push", "bring", "slide"]],
    ["?over/right ?to/toward/towards/by/near yourself/you", "o", {"target": "me"}, ["pull", "bring", "slide"]],   # intent c2
    ["?over/right to/toward/towards/by/near my side", "o", {"target": "me"}, ["pull", "bring", "slide"]],
    ["?a little/bit/lot ?bit nearer/closer [to me|to you|to us|to yourself]", "o", {"target": "closer"}, ["pull", "push", "bring", "slide"]],
    ["away from you/yourself/me/us/<O>", "o", {"target": "away"}, ["pull", "push", "slide"]],
    ["?all/right ?the ?way ?up/over ?real/really/super close/closer to <O>", "o", {"target": "$O"}, ["pull", "push", "bring", "slide"]],
    ["all the way ?up/over to/toward/towards <O>", "o", {"target": "$O"}, ["pull", "push", "bring", "slide"]],
    ["my/this way", "o", {"target": "me"}, ["pull", "push", "bring", "slide"]],
    ["here", "o", {"target": "me"}, ["pull", "push", "bring", "slide"]],
    ["closer [to me|to you|to us]", "o", {"target": "closer"}, ["pull", "push", "bring", "slide"]],
    ["a little/bit closer", "o", {"target": "closer"}, ["pull", "push", "bring", "slide"]],
    ["a little bit closer", "o", {"target": "closer"}, ["pull", "push", "bring", "slide"]],
    ["a little/bit further/farther ?away", "o", {"target": "away"}, ["pull", "push", "bring", "slide"]],
    ["over hear", "o", {"target": "me"}, ["pull", "push", "bring", "slide"]],
    ["nearer [in]", "o", {"target": "closer"}, ["pull", "push", "bring", "slide"]],
    ["back [where it was|where it belongs|where it goes]", "o", {"target": "back"}, ["pull", "push", "bring", "slide"]],
    ["away [from me|from you|from us]", "o", {"target": "away"}, ["pull", "push", "bring", "slide"]],
    ["further/farther ?away/back", "o", {"target": "away"}, ["pull", "push", "bring", "slide"]],
    ["up/over/across/back/right ?up to/toward/towards/against/under/beside/by/near <O>", "o", {"target": "$O"}, ["pull", "push", "bring", "slide"]],
    ["?up/over/right next/close/closer/nearer to <O>", "o", {"target": "$O"}, ["pull", "push", "bring", "slide"]],
    ["?right in front of <O>", "o", {"target": "$O"}, ["pull", "push", "bring", "slide"]],
    ["to/toward/towards/against/under/beside/by <O>", "o", {"target": "$O"}, ["pull", "push", "bring", "slide"]],
    ["over/up/out/in", "o", {}, ["pull", "push", "bring", "slide"]],
    ["on your ears", "o", {}, ["put_on"]],
    # object: sides
    ["in/with/using/into ?your/the/my/ur @lr @paw", "o", {"side": "$lr"}],
    ["in/with/using/into ?your/the/my/ur @lr", "o", {"side": "$lr"}],
    ["@lr handed/pawed", "o", {"side": "$lr"}],
    ["in/with/using ?your/the/my/ur both ?your/of ?your @paw", "o", {"side": "both"}],
    ["in/with/using ?your/the/my/ur two @paw", "o", {"side": "both"}],
    ["in/with/using both", "o", {"side": "both"}],
    ["two handed/pawed", "o", {"side": "both"}],
    ["with both of them", "o", {"side": "both"}],
    ["in/with/using ?your/the/my/ur other ?@paw", "o", {"side": "other"}],
    # object: places
    ["<P>", "o", {"place": "$P"}],
    ["down <P>", "o", {"place": "$P"}],
    ["back <P>", "o", {"place": "$P", "_pb": 1}],
    ["back down <P>", "o", {"place": "$P", "_pb": 1}],
    ["next/close to <O>", "o", {"place": "near", "near": "$O"}],
    ["beside/by/near/besides <O>", "o", {"place": "near", "near": "$O"}],
    ["back where it/they belongs/belong/goes/go/was/were [from]", "o", {"_pb": 1}],
    ["back where it/they came from", "o", {"_pb": 1}],
    ["back to where it/they belongs/belong/goes/go/was/were/came [from]", "o", {"_pb": 1}],
    ["where it/they belongs/belong/goes/go/was/were/lives/live", "o", {"_pb": 1}],
    ["where it/they came from", "o", {"_pb": 1}],
    ["?back where you found it/them/that", "o", {"_pb": 1}],
    ["?back in/to its/their ?own place/spot/home/places/spots", "o", {"_pb": 1}],
    ["?back on/in its/their shelf/spot", "o", {"_pb": 1}],
    ["back home", "o", {"_pb": 1}],
    ["back where it/they started ?from/out", "o", {"_pb": 1}],   # intent c2
    ["back to where it/they started ?from/out", "o", {"_pb": 1}],
    ["?back in/to/on its/their original/usual spot/place/position/home", "o", {"_pb": 1}],
    ["?back where it/they was/were/came ?from ?before", "o", {"_pb": 1}],
    ["exactly/right", "o", {}, ["put_back", "place", "put_down"]],
    ["back", "o", {"_pb": 1}, ["put_down", "put_back", "place", "drop", "pick_up", "hold_up", "show", "offer"]],
    ["away", "o", {"_pb": 1}, ["put_down", "put_back", "place", "drop", "pick_up"]],
    ["down", "o", {"_down": 1}],
    ["over there/here", "o", {}],
    # object: recipient / camera
    ["to/at/toward/towards me/us", "o", {"_to": 1}],
    ["to/at/toward/towards the camera/screen", "o", {"_cam": 1}],
    ["to/for the user/person/viewer/audience/camera", "o", {"_cam": 1}],   # intent c2
    ["to my side", "o", {"_to": 1}],
    ["for the camera", "o", {"_cam": 1}],
    ["so i/we can see ?it/them", "o", {"_cam": 1}],
    ["for me/us to see", "o", {"_cam": 1}],
    ["to everyone/everybody", "o", {"_cam": 1}],
    # object: height / turning amounts
    ["high/higher", "o", {"_hi": 1}],
    ["up high/higher", "o", {"_hi": 1}],
    ["really/very/super high", "o", {"_hi": 1}],
    ["in/into the air/sky", "o", {"_hi": 1}],
    ["up in/into the air/sky", "o", {"_hi": 1}],
    ["over/above your head", "o", {"_hi": 1}],
    ["around/round", "o", {"amount": 0.5}, ["turn"]],
    ["all the way around/round", "o", {"amount": 1}, ["turn"]],
    ["half way ?around/round", "o", {"amount": 0.5}, ["turn"]],
    ["halfway ?around/round", "o", {"amount": 0.5}, ["turn"]],
    ["half a turn", "o", {"amount": 0.5}, ["turn"]],
    ["a quarter [of a] ?turn/way", "o", {"amount": 0.25}, ["turn"]],
    ["a little/bit", "o", {"amount": 0.25}, ["turn"]],
    ["a little bit", "o", {"amount": 0.25}, ["turn"]],
    ["90/ninety degrees", "o", {"amount": 0.25}, ["turn"]],
    ["180 degrees", "o", {"amount": 0.5}, ["turn"]],
    ["one hundred eighty degrees", "o", {"amount": 0.5}, ["turn"]],
    ["360 degrees", "o", {"amount": 1}, ["turn"]],
    ["upside down", "o", {"_flip": 1}],
    ["over", "o", {}, ["flip", "turn", "hand_over"]],
    ["closely/close/closer/carefully", "o", {}, ["inspect", "show", "hold_up", "hug"]],
    ["real/really/super close/closely", "o", {}, ["inspect", "show", "hold_up"]],   # intent c2
    ["in/with/between your/my/both paws/hands/paw/hand/arms", "o", {}],
    ["completely/fully ?around/round", "o", {"amount": 1}, ["turn"]],
    ["so i/we can see the other side/back", "o", {"amount": 0.5}, ["turn"]],
    ["in air", "o", {"_hi": 1}],
    ["here/over here", "o", {}, ["show", "offer", "hold_up"]],
    ["up close", "o", {}, ["inspect", "show", "hold_up", "hug"]],
    ["really closely/close", "o", {}, ["inspect", "show"]],
    ["tight/tightly", "o", {}, ["hug", "pick_up", "hold_up"]],
    ["and catch it/them", "o", {}, ["toss"]],
    ["and keep holding it/them", "o", {"hold": True}],
    ["and dont let go", "o", {"hold": True}],
    # fillers
    ["please/now/again/too/also/okay/ok/yeah/thanks/then/already/sometime/right", "mo", {}],
    ["instead/first/go/cool/yay/dear/kay/plz/pls", "mo", {}], ["after that", "mo", {}],   # intent c2
    ["will/would/can ya", "mo", {}], ["from you", "mo", {}], ["for us all", "mo", {}],
    ["right here", "mo", {}], ["all by yourself", "mo", {}],
    ["thank you", "mo", {}],
    ["for me/us/everyone/everybody/mommy/daddy ?please", "mo", {}],
    ["right now", "mo", {}], ["real quick", "mo", {"speed": "fast"}],
    ["one more time", "mo", {}], ["once more", "mo", {}], ["one last time", "mo", {}],
    ["if you can/want/like/could", "mo", {}],
    ["will/can/could/would you ?please", "mo", {}],
    ["as well", "mo", {}], ["this time", "mo", {}], ["just once", "mo", {"count": 1}],
    ["like you did before/earlier", "mo", {}],
    ["buddy/pal/friend/sweetie/honey/dude/man/kiddo/mister", "mo", {}],
    ["little/big buddy/guy/one/man", "mo", {}],
    ["my friend", "mo", {}],
]

# preamble frames (raw multi-word) — stripped before a verb; type: req | show | see | have | go
FRAMES = [["can you", "req"], ["could you", "req"], ["would you", "req"], ["will you", "req"], ["can u", "req"],
          ["could u", "req"], ["can you please", "req"], ["could you please", "req"], ["would you please", "req"],
          ["would you mind", "ger"], ["do you mind", "ger"], ["would ya", "req"], ["can ya", "req"],
          ["i want you to", "req"], ["i would like you to", "req"], ["id like you to", "req"], ["i need you to", "req"],
          ["i wanna see you", "req"], ["i want to see you", "req"], ["i would love to see you", "req"],
          ["id love to see you", "req"], ["id like to see you", "req"], ["i would like to see you", "req"],
          ["i wanna see", "mnp"], ["i want to see", "mnp"], ["id love to see", "mnp"], ["i would love to see", "mnp"],
          ["let me see you", "req"], ["lets see you", "req"], ["let us see you", "req"], ["show me you can", "req"],
          ["show me how you", "req"], ["show us how you", "req"], ["show me how well you can", "req"],
          ["go ahead and", "req"], ["try to", "req"], ["try and", "req"], ["try", "ger"], ["why dont you", "req"],
          ["how about you", "req"], ["what about you", "req"], ["how about", "ger"], ["what about", "ger"],
          ["do you want to", "req"], ["do you wanna", "req"], ["wanna", "req"], ["you can", "req"],
          ["you may", "req"], ["you should", "req"], ["lets", "req"], ["let us", "req"], ["let me see", "mnp"], ["lets see", "mnp"], ["let us see", "mnp"],
          ["real quick", "req"], ["quick quick", "req"], ["time to", "req"],
          ["its time to", "req"], ["time for", "mnp"], ["its time for", "mnp"], ["simon says", "req"],
          ["show me", "mnp"], ["show us", "mnp"], ["give me", "mnp"], ["give us", "mnp"], ["gimme", "mnp"],
          ["go", "req"], ["go and", "req"], ["come on", "req"], ["cmon", "req"], ["ready set", "req"],
          ["now", "req"], ["and now", "req"], ["and then", "req"], ["then", "req"], ["ok now", "req"],
          ["i dare you to", "req"], ["bet you cant", "ndg"], ["are you able to", "req"], ["please", "req"],
          ["keep", "keep"], ["keep on", "keep"], ["continue", "keep"], ["start", "keep"], ["start to", "req"],
          ["just", "req"], ["quickly", "fast"], ["slowly", "slow"], ["gently", "slow"], ["carefully", "slow"],
          ["quick", "fast"], ["first", "req"], ["next", "req"], ["also", "req"], ["again", "req"],
          # intent c2: challenge / indirect / show-me frames
          ["i bet you cant", "req"], ["i bet you can", "req"], ["bet you can", "req"], ["betcha cant", "req"],
          ["i bet you cannot", "req"], ["bet you cannot", "req"], ["i dare you", "req"], ["show me you", "ger"],
          ["show us you", "ger"], ["show me how to", "req"], ["show me how you can", "req"], ["can i see you", "ger"],
          ["can we see you", "ger"], ["may i see you", "ger"], ["let me watch you", "ger"], ["lemme see you", "ger"],
          ["lets watch you", "ger"], ["i wanna watch you", "ger"], ["i want to watch you", "ger"],
          ["would you be so kind as to", "req"], ["be so kind as to", "req"], ["do me a favor and", "req"],
          ["you gotta", "req"], ["you have to", "req"], ["you need to", "req"], ["make sure you", "req"],
          ["its your turn to", "req"], ["your turn to", "req"], ["now its time to", "req"], ["get ready to", "req"],
          ["i want to see you", "ger"], ["i wanna see you", "ger"], ["lets see you", "ger"], ["let me see you", "ger"]]
PRE = ["hey", "hi", "hello", "so", "and", "um", "uh", "uhh", "umm", "er", "erm", "hmm", "well", "alright", "okay",
       "oh", "ooh", "yeah", "yes", "ah", "right", "oops", "ok", "now", "please", "alrighty", "sure", "buddy", "pal", "friend", "kiddo", "dude",
       "sweetie", "little", "guy", "mister", "mr", "maybe", "dear", "kindly", "pls", "plz", "cmon", "like", "just", "wait"]
FILLERS = ["um", "uh", "uhh", "umm", "er", "erm", "hmm", "like", "maybe", "please", "pls", "plz", "kindly"]   # intent c2: + maybe/please anywhere   # removed anywhere in an act clause ('like' only between a verb and its object)
GERUND = {"spinning": "spin", "twirling": "twirl", "turning": "turn", "jumping": "jump", "hopping": "hop",
          "bouncing": "bounce", "dancing": "dance", "walking": "walk", "stepping": "step", "wiggling": "wiggle",
          "wobbling": "wobble", "rolling": "roll", "flipping": "flip", "facing": "face", "holding": "hold",
          "juggling": "juggle", "tossing": "toss", "throwing": "throw", "shaking": "shake", "hugging": "hug",
          "looking": "look", "cartwheeling": "cartwheel", "marching": "march", "running": "run", "jogging": "jog",
          "sitting": "sit", "lying": "lie", "laying": "lay", "standing": "stand", "bowing": "bow", "picking": "pick",
          "grabbing": "grab", "putting": "put", "showing": "show", "doing": "do", "getting": "get",
          "rotating": "rotate", "smelling": "smell", "sniffing": "sniff", "cuddling": "cuddle", "boogieing": "boogie",
          "backflipping": "backflip", "somersaulting": "somersault", "tiptoeing": "tiptoe", "leaping": "leap",
          "giving": "give", "passing": "pass", "handing": "hand", "switching": "switch", "lifting": "lift",
          "raising": "raise", "dropping": "drop", "examining": "examine", "inspecting": "inspect",
          "checking": "check", "catching": "catch", "playing": "play", "taking": "take", "making": "make",
          "trying": "try", "pirouetting": "pirouette", "jiggling": "jiggle", "grooving": "groove",
          # intent c2
          "whirling": "whirl", "pulling": "pull", "pushing": "push", "opening": "open", "closing": "close",
          "typing": "type", "working": "work", "sliding": "slide", "bringing": "bring", "moving": "move",
          "setting": "set", "placing": "place", "dragging": "drag", "removing": "remove", "wearing": "wear",
          "plopping": "plop", "settling": "settle", "waving": "wave", "rising": "rise", "using": "use",
          "stopping": "stop", "quitting": "quit", "pausing": "pause", "scooting": "scoot", "shifting": "shift",
          "nudging": "nudge", "tugging": "tug", "hauling": "haul", "carrying": "carry", "fetching": "fetch",
          "hugging": "hug", "offering": "offer", "transferring": "transfer", "swapping": "swap", "studying": "study",
          "returning": "return", "replacing": "replace", "tidying": "tidy", "cleaning": "clean", "frontflipping": "frontflip",
          "heading": "head", "stepping": "step", "flopping": "flop", "slipping": "slip", "popping": "pop"}
# words that end a request when they are what the user says after a known command ("spin around, buddy")
NEG = ["dont", "not", "never", "no", "cant", "wont", "shouldnt", "couldnt", "wouldnt", "didnt", "doesnt", "stop",
       "without", "nobody", "neither", "nor", "quit"]
LR = {"left": "left", "right": "right", "lefft": "left", "write": "right"}
SLR = {"left": "screen_left", "right": "screen_right"}
POS = {"middle": "middle", "center": "middle", "centre": "middle"}
STYLE_STOP = ["please", "now", "again", "for", "too", "okay", "ok", "and", "then", "with", "to", "in", "on", "at"]


def expand(pat: str) -> list:
    """'turn away [from me|from us]' -> ['turn away', 'turn away from me', 'turn away from us'] (engines see no groups)"""
    if "[" not in pat:
        return [pat]
    a = pat.index("[")
    b = pat.index("]", a)
    head, body, tail = pat[:a], pat[a + 1:b], pat[b + 1:]
    outs = []
    for choice in [""] + body.split("|"):
        outs += expand(" ".join((head + " " + choice + " " + tail).split()))
    return outs


def expand_all(lst: list) -> list:
    out = []
    for e in lst:
        for p in expand(e[0]):
            out.append([p] + list(e[1:]))
    return out


EVAL = {"full", "good", "nice", "real", "proper", "perfect", "cool", "awesome", "super", "fun", "great",
        "little bit", "more", "one", "sad", "excited"}


def build() -> dict:
    nouns = {}
    def addn(word, canon):
        nouns.setdefault(word, canon)
    for n in NOUN_LIST:
        addn(n, n)
    for n in NOUN_LIST:
        if n.endswith(("s", "x", "ch", "sh")):
            pl = n + "es"
        elif n.endswith("y") and n[-2] not in "aeiou":
            pl = n[:-1] + "ies"
        else:
            pl = n + "s"
        addn(pl, n + "+")       # '+' = plural marker (-> ref all unless "one of the ...")
    for k, v in PLURAL_SPECIAL.items():
        if v in nouns or v in NOUN_LIST:
            nouns[k] = v + "+" if k != v else v
    multi = {}
    def plural(w):
        if w.endswith(("s", "x", "ch", "sh")):
            return w + "es"
        if w.endswith("y") and w[-2] not in "aeiou":
            return w[:-1] + "ies"
        return w + "s"
    for k, v in NOUN_MULTI.items():
        multi[k] = v
        if not k.endswith("s"):
            multi[plural(k)] = v + "+"
    for k, v in NOUN_MULTI.items():
        if " " not in k:
            nouns.setdefault(k, v)
    for k, v in PAIR.items():
        if " " in k:
            multi[k] = v
        else:
            nouns[k] = v
    multi = {k: v for k, v in multi.items() if " " in k}
    qty = {k: v for k, v in QTY}
    return {
        "_doc": "whole-body moves + props (generated by tools/intent_bench/act_lexicon.py; see its docstring for the "
                "pattern syntax). Runs on raw clause tokens before the legacy rules; app/intent.py parse_act == app/intent.js.",
        "moves": expand_all(MOVES), "objs": expand_all(OBJ), "mods": expand_all(MODS), "frames": FRAMES, "pre": PRE, "fillers": FILLERS, "gerund": GERUND,
        "neg": NEG,
        "cls": {"madjb": {k: 1 for k, v in MADJ.items() if v and k not in EVAL}, "lr": LR, "paw": PAWS, "mn": MN, "madj": {k: 1 for k in MADJ}, "dstyle": DSTYLE,
                "mnstyle": MNSTYLE, "style": {}, "pos": POS},
        "madj": MADJ, "nums": NUMS, "qty": qty, "colors": COLORS, "sizes": SIZES, "ords": ORDS, "shades": SHADES,
        "nouns": nouns, "nouns_multi": multi, "generic": GENERIC, "anyn": ANYN, "places": PLACES,
        "style_stop": STYLE_STOP,
    }


def main():
    d = json.load(open(RULES, encoding="utf-8"))
    d["act"] = build()
    with open(RULES, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
        f.write("\n")
    a = d["act"]
    print(f"act: {len(a['moves'])} move patterns, {len(a['objs'])} object patterns, {len(a['mods'])} modifiers, "
          f"{len(a['nouns'])} nouns, {len(a['colors'])} colours")


if __name__ == "__main__":
    main()
