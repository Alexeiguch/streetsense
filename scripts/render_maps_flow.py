"""Draw outputs/maps_flow.png: how StreetSense uses OpenStreetMap and Google Maps (from streetsense/maps/*)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(14, 7.5), dpi=110); ax.set_xlim(0, 14); ax.set_ylim(0, 7.5); ax.axis("off")
fig.patch.set_facecolor("#0e1015")
def box(x, y, w, h, title, body, c):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc="#1b1f27", ec=c, lw=2))
    ax.text(x + .15, y + h - .3, title, color=c, fontsize=12, weight="bold", va="top")
    ax.text(x + .15, y + h - .72, body, color="#d8dbe2", fontsize=9.2, va="top", linespacing=1.45)
def arrow(a, b, label="", c="#8a90a0"):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6))
    if label: ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + .12, label, color=c, fontsize=8.5, ha="center")
OSM, G, ME, GR = "#4cc38a", "#f5a524", "#6ea8fe", "#8a90a0"
ax.text(.3, 7.15, "StreetSense maps: who is called, when", color="white", fontsize=17, weight="bold")
ax.text(.3, 6.78, "Off unless STREETSENSE_MAPS=1 (streetsense/config.py). Default provider 'fixture' = replay of tests/fixtures/maps/, fully offline.", color=GR, fontsize=9.5)
box(.3, 3.2, 3.0, 2.9, "User asks Wonder", "\"take me to Renwick\"\n\"what's near me?\"\n\"am I on route?\"\n\nstreetsense/wonder_agent.py\n(Muse Wonder tools)", ME)
box(4.2, 3.2, 3.3, 2.9, "MapSession", "plan_route  find_nearby\nroute_status  google_search\n\nstreetsense/maps/session.py\nbuilt by maps/__init__.py\nbuild_session()", ME)
box(8.5, 4.55, 5.2, 1.95, "OpenStreetMap (no key)", "Valhalla  – walking routes, alternates, blind mode\nOverpass  – crossing tags (signals, sound, tactile)\nNominatim – place search + reverse lookup\nstreetsense/maps/osm.py", OSM)
box(8.5, 2.2, 5.2, 1.95, "Google Maps (opt-in, needs key)", "Grounding Lite MCP: search_places, compute_routes\nonly if STREETSENSE_MAPS_GOOGLE=1 + GOOGLE_MAPS_API_KEY\nmax 1 call per question, spoken disclosure\nstreetsense/maps/google_agent.py  (no key set → never called)", G)
box(4.2, .3, 3.3, 2.3, "Navigator", "crossing preview ~45 m ahead\n'at crossing' at 12 m\nstreetsense/maps/navigator.py", ME)
box(.3, .3, 3.0, 2.3, "NOT wired yet", "Pipeline.note_crossing_distance\n(pipeline.py) has no caller\nweb demo + glasses app: no maps\n(docs/MAPS.md)", "#ff5a52")
arrow((3.3, 4.65), (4.2, 4.65), "tool call")
arrow((7.5, 5.2), (8.5, 5.5), "routes, crossings,\nplaces", OSM)
arrow((7.5, 4.0), (8.5, 3.2), "only if enabled", G)
arrow((5.85, 3.2), (5.85, 2.6), "chosen route")
arrow((4.2, 1.45), (3.3, 1.45), "", "#ff5a52")
ax.text(8.5, 1.5, "Route choice weighs unsignalised crossings, crossing count, stairs and length (navigator reason string).\nExample (fixture): 536 m, 4 signalised crossings vs 919 m, 6 crossings → outputs/maps_example.png", color=GR, fontsize=9)
fig.savefig("outputs/maps_flow.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
