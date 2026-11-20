"""Per-person Weinberg K from walks of the same route.

    python scripts/fit_person_k.py

data/people.json says who walked each session and which K is the reference:
the reference person's K comes from tape-measured trials
(scripts/calibrate_weinberg.py). For everyone else, their walks of a route
both people walked are scaled so they cover the same distance as the
reference person's walks of that route:

    K_person = K_ref * mean(reference distance) / mean(person's Weinberg sum)

where a walk's Weinberg sum = logged distance / the K it was logged with.
So it is only as accurate as the reference person's distances (~3-5%).
Writes board/me193_nav/python/people.json ({name: K}), which the board app
reads for its "who's walking" selector.
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
PEOPLE = REPO / "data" / "people.json"
OUT = REPO / "board" / "me193_nav" / "python" / "people.json"


def main():
    cfg = json.loads(PEOPLE.read_text())
    ref, k_ref = cfg["reference"]["person"], cfg["reference"]["K"]
    walks = {}
    for name, info in cfg["sessions"].items():
        meta = json.loads((REPO / "data" / "raw" / f"{name}.json").read_text())
        k_logged = meta.get("weinberg_K") or k_ref
        walks.setdefault((info["person"], info["route"]), []).append(meta["totals"]["distance_m"] / k_logged)
    ks = {ref: k_ref}
    for (person, route), sums in walks.items():
        if person == ref or (ref, route) not in walks:
            continue
        ref_dist = k_ref * np.mean(walks[(ref, route)])
        ks[person] = round(float(ref_dist / np.mean(sums)), 4)
        print(f"{person}: route {route!r}, {len(sums)} walks vs {len(walks[(ref, route)])} of {ref}'s "
              f"({ref_dist:.2f} m): K = {ks[person]}")
    print(f"{ref}: K = {k_ref} (reference)")
    OUT.write_text(json.dumps(ks, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
