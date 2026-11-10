"""make_dashboard.py - Step 5: pour data/week.json into the dashboard template -> dashboard.html"""
import json, math


def clean(o):
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else round(o, 5)
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    return o


def main():
    data = clean(json.load(open("data/week.json")))
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    html = open("dashboard_template.html").read().replace("__DATA__", blob)
    open("dashboard.html", "w").write(html)
    print(f"wrote dashboard.html ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main()
