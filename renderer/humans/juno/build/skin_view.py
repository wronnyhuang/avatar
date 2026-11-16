"""asset — Blender look-dev viewer for a juno.asset (Cycles, 3-point light like the reference).

blender -b -P bl_view.py -- <asset_dir> <out.png> '<json opts>'
opts: {"lod":"lod0", "shapes":{"jawOpen":0.6,...}, "pose":{"upperarm_l":[x,y,z deg local euler XYZ],...},
       "cam":"face|bust|body|face34|side|hands", "res":[w,h], "samples":64, "hair":true, "bg":[r,g,b]}
Skinning + morphs are done with numpy through asset_io (the same math RENDER uses), Blender only shades.
"""
import json
import math
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
ASSET, OUT = argv[0], argv[1]
OPT = json.loads(argv[2]) if len(argv) > 2 else {}
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))
from humans import asset_io  # noqa: E402

A = asset_io.load_asset(ASSET, OPT.get("lod", "lod0"))
names = A.shape_names
w = np.zeros(len(names), np.float32)
for k, v in OPT.get("shapes", {}).items():
    w[names.index(k)] = v


def euler(d):
    x, y, z = np.radians(d)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    M = np.eye(4)
    M[:3, :3] = Rz @ Ry @ Rx
    return M


pose = OPT.get("pose", {})
hints = A.skeleton.get("skeleton_hints", {})
if "jawOpen" in OPT.get("shapes", {}) and "jaw" not in pose:
    pose["jaw"] = [OPT["shapes"]["jawOpen"] * hints.get("jaw_open_deg", 20), 0, 0]
nb = len(A.bones)
Bw = np.zeros((nb, 4, 4))
for i, b in enumerate(A.bones):
    L = euler(pose.get(b["name"], [0, 0, 0]))
    if b["parent"] < 0:
        Bw[i] = A.rest[i] @ L
    else:
        p = b["parent"]
        Bw[i] = Bw[p] @ (np.linalg.inv(A.rest[p]) @ A.rest[i]) @ L
P = A.skin(Bw.astype(np.float32), w)
C = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)   # blender <- asset
Pb = P @ C.T
UV = A.mesh["uv0"]
I = A.mesh["index"].reshape(-1, 3)

bpy.ops.wm.read_factory_settings(use_empty=True)
sc = bpy.context.scene
sc.render.engine = "CYCLES"
sc.cycles.device = "CPU"
sc.cycles.samples = OPT.get("samples", 64)
sc.cycles.use_denoising = True
res = OPT.get("res", [960, 1080])
sc.render.resolution_x, sc.render.resolution_y = res
sc.render.film_transparent = False
sc.view_settings.view_transform = "AgX" if "AgX" in [i.name for i in bpy.types.ColorManagedViewSettings.bl_rna.properties["view_transform"].enum_items] else "Filmic"
sc.view_settings.look = "None"
sc.view_settings.exposure = OPT.get("exposure", 0.0)

texcache = {}


def img(path):
    if path not in texcache:
        texcache[path] = bpy.data.images.load(path)
    return texcache[path]


def make_mat(mid):
    m = A.meta["materials"].get(mid, {})
    typ = m.get("type", "")
    mat = bpy.data.materials.new(mid)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    suf = A.lod.get("tex_suffix", "")
    if "albedo" in m:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = img(os.path.join(ASSET, m["albedo"].replace(".png", suf + ".png")))
        nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
        if m.get("type") in ("lashes",):
            nt.links.new(t.outputs["Alpha"], bsdf.inputs["Alpha"])
    if "normal" in m:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = img(os.path.join(ASSET, m["normal"].replace(".png", suf + ".png")))
        t.image.colorspace_settings.name = "Non-Color"
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = 1.0
        nt.links.new(t.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    if "packed" in m:
        t = nt.nodes.new("ShaderNodeTexImage")
        t.image = img(os.path.join(ASSET, m["packed"].replace(".png", suf + ".png")))
        t.image.colorspace_settings.name = "Non-Color"
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        nt.links.new(t.outputs["Color"], sep.inputs["Color"])
        nt.links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
    pr = m.get("params", {})
    if typ == "skin":
        bsdf.inputs["Subsurface Weight"].default_value = OPT.get("sss", 0.35)
        bsdf.inputs["Subsurface Radius"].default_value = [x / 1000 * 3 for x in pr.get("sss_radius_mm", [3.6, 1.4, 0.8])]
        bsdf.inputs["Subsurface Scale"].default_value = 1.0
    elif typ == "cornea":
        bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
        bsdf.inputs["Roughness"].default_value = 0.0
        bsdf.inputs["Transmission Weight"].default_value = 1.0
        bsdf.inputs["IOR"].default_value = pr.get("ior", 1.376)
    elif typ in ("teeth", "tongue"):
        bsdf.inputs["Subsurface Weight"].default_value = 0.2
    elif typ == "cloth":
        bsdf.inputs["Sheen Weight"].default_value = 0.5
    elif typ == "hair":
        pass
    if typ == "eye_occlusion":
        bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
    return mat


mats = {}
for s in A.submeshes:
    if s["material"] not in mats:
        mats[s["material"]] = make_mat(s["material"])
    v0, nv = s["first_vertex"], s["vertex_count"]
    tri = I[s["first_index"] // 3:(s["first_index"] + s["index_count"]) // 3] - v0
    me = bpy.data.meshes.new(s["name"])
    me.from_pydata(Pb[v0:v0 + nv].tolist(), [], tri.tolist())
    uvl = me.uv_layers.new(name="uv0")
    loops = tri.reshape(-1)
    uvl.data.foreach_set("uv", UV[v0:v0 + nv][loops].astype(np.float32).reshape(-1))
    me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
    me.update()
    ob = bpy.data.objects.new(s["name"], me)
    ob.data.materials.append(mats[s["material"]])
    sc.collection.objects.link(ob)

# hair strands as curves
if OPT.get("hair", True):
    for g, h in A.hair.items():
        gi = A.meta["hair_groups"].get(g, {})
        bone = gi.get("bone", "head")
        bi = A.bone_index.get(bone, 0)
        K = Bw[bi] @ np.linalg.inv(A.rest[bi])
        pos = h["pos"].reshape(-1, 3)
        pos = (np.c_[pos, np.ones(len(pos))] @ K.T)[:, :3] @ C.T
        pos = pos.reshape(h["S"], h["P"], 3)
        cu = bpy.data.curves.new(g, "CURVE")
        cu.dimensions = "3D"
        cu.bevel_depth = 0
        step = max(1, int(OPT.get("hair_step", 1)))
        for si in range(0, h["S"], step):
            sp = cu.splines.new("POLY")
            sp.points.add(h["P"] - 1)
            wr, wt = h["width"][si]
            for k in range(h["P"]):
                t = k / (h["P"] - 1)
                sp.points[k].co = (*pos[si, k], 1)
                sp.points[k].radius = (wr * (1 - t) + wt * t) * 0.5 * 1000
        ob = bpy.data.objects.new("hair_" + g, cu)
        cu.bevel_depth = 0.001
        cu.bevel_resolution = 0
        mat = bpy.data.materials.new("hair_" + g)
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            if n.type == "BSDF_PRINCIPLED":
                nt.nodes.remove(n)
        hb = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
        hb.parametrization = "MELANIN"
        hb.inputs["Melanin"].default_value = gi.get("melanin", 0.75)
        hb.inputs["Melanin Redness"].default_value = gi.get("redness", 0.6)
        hb.inputs["Roughness"].default_value = 0.3
        nt.links.new(hb.outputs[0], nt.nodes["Material Output"].inputs["Surface"])
        cu.materials.append(mat)
        sc.collection.objects.link(ob)

# lights + world
world = bpy.data.worlds.new("w")
sc.world = world
world.use_nodes = True
bg = OPT.get("bg", [0.32, 0.36, 0.42])
world.node_tree.nodes["Background"].inputs["Color"].default_value = (*bg, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6

head_a = A.rest[A.bone_index["head"]][:3, 3]
eye_a = np.array(A.meta["landmarks"]["eye_l"])
face_c = np.array([0, eye_a[1] - 0.03, eye_a[2]])


def light(name, kind, loc_a, energy, size, color=(1, 1, 1), target=face_c):
    L = bpy.data.lights.new(name, kind)
    L.energy = energy
    L.color = color
    if kind == "AREA":
        L.size = size
    o = bpy.data.objects.new(name, L)
    o.location = tuple(C @ np.asarray(loc_a))
    d = C @ (np.asarray(target) - np.asarray(loc_a))
    o.rotation_euler = d_to_euler(d)
    sc.collection.objects.link(o)


def d_to_euler(d):
    from mathutils import Vector
    return Vector(d).to_track_quat("-Z", "Y").to_euler()


H = A.meta["height_m"]
light("key", "AREA", face_c + np.array([-1.2, 0.6, 1.4]) * H * 0.7, 260 * H, 1.2, (1.0, 0.96, 0.9))
light("fill", "AREA", face_c + np.array([1.4, 0.1, 1.2]) * H * 0.7, 70 * H, 1.5, (0.85, 0.9, 1.0))
light("rim", "AREA", face_c + np.array([0.9, 0.7, -1.3]) * H * 0.7, 220 * H, 0.8, (1.0, 0.85, 0.65))

presets = {  # (target asset, distance m, azimuth deg, elevation deg, focal mm)
    "face": (face_c + np.array([0, 0.0, 0]), 0.75, 0, 2, 85),
    "face34": (face_c, 0.75, 32, 4, 85),
    "side": (face_c, 0.75, 90, 0, 85),
    "bust": (face_c + np.array([0, -0.12, 0]), 1.25, 0, 3, 70),
    "body": (np.array([0, H * 0.5, 0.05]), 4.2, 12, 6, 50),
    "bodyfront": (np.array([0, H * 0.5, 0.05]), 4.2, 0, 4, 50),
    "hands": (np.array(A.rest[A.bone_index["hand_r"]][:3, 3]), 0.6, -20, 10, 70),
}
cam = OPT.get("cam", "face")
tgt, dist, az, el, fl = presets[cam] if isinstance(cam, str) else cam
tgt = np.asarray(OPT.get("target", tgt), float)
az, el = math.radians(az), math.radians(el)
pos_a = tgt + dist * np.array([math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el)])
cd = bpy.data.cameras.new("cam")
cd.lens = fl
co = bpy.data.objects.new("cam", cd)
co.location = tuple(C @ pos_a)
co.rotation_euler = d_to_euler(C @ (tgt - pos_a))
sc.collection.objects.link(co)
sc.camera = co
sc.render.filepath = OUT
bpy.ops.render.render(write_still=True)
print("WROTE", OUT)
