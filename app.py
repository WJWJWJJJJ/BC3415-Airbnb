"""
AI dispute assistant: Airbnb case study (BC3415 group project demo)

Workflow: 1 Read the complaint (LLM + sentiment) -> 2 Find the policy (RAG: TF-IDF + similarity search)
          -> 3 Check the case (agent tools) -> 4 Decide the route
          -> 5 Resolve automatically OR hand to a human specialist -> 6 Explain the outcome to both sides

Student project, not affiliated with Airbnb. Bookings, people and policies are
simplified samples written for the demo, not Airbnb's actual terms.
"""
import os, re, html

import gradio as gr
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# ======================= Settings =======================
AUTO_LIMIT_USD = 500      # cases at or below this amount can be resolved automatically...
MIN_CONFIDENCE = 0.75     # ...if the evidence confidence is at least this...
MIN_POLICY_MATCH = 0.10   # ...and a policy clearly matches the complaint (TF-IDF similarity)
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")   # optional: set under Environment on Render


def llm(prompt):
    """Gemini, if a key is set. Returns None otherwise so the app falls back to rules and templates."""
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        return genai.Client(api_key=GEMINI_API_KEY).models.generate_content(model=LLM_MODEL, contents=prompt).text.strip()
    except Exception as e:
        print("LLM unavailable, using rules/templates:", e)
        return None


# ======================= Sample data =======================
BOOKINGS = {
    "HM-1001": {"guest": "Alicia", "host": "Ben", "nights": 3, "nightly_rate": 150, "where": "Studio, Lisbon"},
    "HM-2002": {"guest": "Chen", "host": "Dana", "nights": 5, "nightly_rate": 220, "where": "2-bed flat, London"},
    "HM-3003": {"guest": "Elif", "host": "Faisal", "nights": 2, "nightly_rate": 180, "where": "Loft, Bangkok"},
}

# The knowledge base the RAG step searches (simplified sample policies)
POLICIES = [
    {"issue": "amenity", "title": "Essential amenity not working",
     "text": "If an essential amenity such as hot water, heating, air-conditioning, Wi-Fi or electricity stops working, "
             "the guest should report it within 72 hours with photo or video evidence. The guest may receive a refund of "
             "up to 50% of the nightly rate for each affected night."},
    {"issue": "cleanliness", "title": "Cleanliness at check-in",
     "text": "If the listing is dirty at check-in, for example unwashed sheets, rubbish, pests, mould or bad smells, the guest "
             "should report it within 24 hours with photos. The host is asked to arrange cleaning, and the guest may receive "
             "a refund of up to 30% of the nightly rate for each affected night."},
    {"issue": "access", "title": "Check-in and access problems",
     "text": "If the guest cannot get into the listing because of a wrong door code, missing keys, a locked door or an "
             "unreachable host, the guest should contact support immediately. The guest may receive a full refund for the "
             "affected night and help to rebook."},
    {"issue": "not_as_described", "title": "Listing not as described",
     "text": "If the listing is significantly different from its description or photos, such as fewer bedrooms, a different "
             "location, or advertised features like a pool or parking missing, the guest should report it within 72 hours "
             "with photos. The guest may receive a refund of up to 50% of the nightly rate for each affected night."},
    {"issue": "noise", "title": "Noise and house rules",
     "text": "Complaints about noise, loud music, parties, smoking or other house-rule breaches are first handled with a formal "
             "warning to the other party. Repeated breaches or parties are escalated to the trust and safety team."},
    {"issue": "safety", "title": "Safety concerns",
     "text": "Any safety concern, such as a gas leak, fire risk, missing smoke alarm, hidden camera or injury, is escalated "
             "immediately to the safety team. The guest is offered help to relocate and a full refund of the affected nights."},
    {"issue": "damage", "title": "Damage claims by hosts",
     "text": "Hosts must submit damage claims within 14 days of checkout, with photos of the damage and repair quotes or "
             "receipts. The guest is given a chance to respond. Claims above the automatic limit or disputed by the guest "
             "are reviewed by a specialist."},
    {"issue": "scam", "title": "Payments outside the platform",
     "text": "Requests to pay by bank transfer, cash or messaging apps outside the platform are a scam signal. The booking is "
             "protected only when paid through the platform. Suspected scams are escalated to the trust and safety team and "
             "the guest is refunded in full."},
    {"issue": "cancellation", "title": "Cancellation policies",
     "text": "Flexible: full refund up to 24 hours before check-in. Moderate: full refund up to 5 days before check-in. "
             "Strict: full refund within 48 hours of booking if check-in is at least 14 days away."},
]

# How each type of issue is handled
ISSUES = {
    "amenity":          {"label": "Amenity not working",       "payee": "guest", "calc": "half"},
    "cleanliness":      {"label": "Cleanliness",               "payee": "guest", "calc": "thirty"},
    "access":           {"label": "Check-in / access",         "payee": "guest", "calc": "one_night"},
    "not_as_described": {"label": "Listing not as described",  "payee": "guest", "calc": "half"},
    "noise":            {"label": "Noise / house rules",       "payee": None,    "calc": "none"},
    "damage":           {"label": "Damage claim",              "payee": "host",  "calc": "claimed"},
    "cancellation":     {"label": "Cancellation",              "payee": "guest", "calc": "booking"},
    "safety":           {"label": "Safety concern",            "payee": "guest", "calc": "affected", "always_human": "safety team"},
    "scam":             {"label": "Possible scam",             "payee": "guest", "calc": "booking",  "always_human": "trust & safety team"},
    "other":            {"label": "Unclear / other",           "payee": None,    "calc": "none",     "always_human": "dispute specialist"},
}

# Keyword rules used when no LLM key is set (checked in this order)
KEYWORDS = [
    ("safety", r"gas leak|smell(s|ed)? (of|like) gas|\bfire\b|smoke alarm|carbon monoxide|hidden camera|camera (in|inside)|injur|unsafe|electric shock|exposed wir"),
    ("scam", r"bank transfer|transfer (the |my )?money|whatsapp|telegram|outside (of )?the (app|platform|airbnb)|off[- ]?platform"
             r"|pay (me |you |him |her )?directly|\bcash\b|wire|paypal|western union|crypto"),
    ("damage", r"\bbroke\b|cracked|damage|smashed|ruined|destroyed|repair|broken (window|tv|lamp|chair|table|door|glass|mirror)"),
    ("cleanliness", r"dirty|filthy|unclean|not clean|stain|\bhair\b|bugs?\b|cockroach|pests?\b|mou?ld|smell|rubbish|trash|dusty"),
    ("access", r"locked out|can'?t get in|cannot get in|could(n'?t| not) get in|door code|\bkeys?\b|lock ?box|key ?box|let (us|me) in"
               r"|host (is |was )?(unreachable|not answering)|couldn'?t reach the host"),
    ("not_as_described", r"not as described|different from|(doesn|didn)'?t match|misleading|no pool|no parking|fewer|smaller than|wrong location|advertised|not like the photos"),
    ("noise", r"noise|noisy|\bloud\b|party|parties|music|smok(e|ed|ing)\b|house rules|pets? (in|inside)"),
    ("cancellation", r"cancel"),
    ("amenity", r"hot water|wi-?fi|internet|heating|heater|air[- ]?con|\ba/?c\b|electricity|\bpower\b|no water|fridge|washing machine"
                r"|not working|doesn'?t work|isn'?t working|stopped working|broken"),
]


# ======================= Step 1: Read the complaint (LLM + sentiment) =======================
def sentiment(text):
    """TextBlob polarity with the course's 3-class rule (Week 6): > +0.2 positive, < -0.2 negative, else neutral."""
    p = round(TextBlob(text).sentiment.polarity, 2)
    return p, ("positive" if p > 0.2 else "negative" if p < -0.2 else "neutral")


def classify(message):
    label = llm("Classify this Airbnb support message into exactly one of these labels: "
                f"{', '.join(ISSUES)}. Reply with the label only.\n\nMessage: {message}")
    label = (label or "").strip().lower()
    if label in ISSUES:
        return label, "LLM"
    t = message.lower()
    for issue, pattern in KEYWORDS:
        if re.search(pattern, t):
            return issue, "keyword rules"
    return "other", "keyword rules"


# ======================= Step 2: Find the policy (RAG: TF-IDF + similarity search) =======================
def _words(t):
    # lowercase, drop stop words, crude plural stemming ("rates" -> "rate")
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w
            for w in re.findall(r"[a-z]+", t.lower()) if w not in ENGLISH_STOP_WORDS]


tfidf = TfidfVectorizer(analyzer=_words)                                                  # TF-IDF (Week 7)
policy_vectors = tfidf.fit_transform([f"{p['title']}. {p['text']}" for p in POLICIES])   # the knowledge base


def find_policy(message, issue):
    query = message if issue == "other" else f"{ISSUES[issue]['label']}. {message}"
    scores = cosine_similarity(tfidf.transform([query]), policy_vectors)[0]
    best = int(scores.argmax())
    return {**POLICIES[best], "score": round(float(scores[best]), 2)}


# ======================= Step 3: Check the case (agent tools) =======================
def tool_check_evidence(evidence, other_party):
    conf, notes = 1.0, []
    if evidence:
        notes.append("evidence attached")
    else:
        conf -= 0.3; notes.append("no photos or documents")
    if other_party == "Disputes it":
        conf -= 0.4; notes.append("other party disputes it")
    elif other_party == "Confirms it":
        notes.append("other party confirms")
    else:
        notes.append("other party has not replied yet")
    return round(max(conf, 0.0), 2), notes


def tool_scam_screen(message, issue):
    return issue == "scam" or bool(re.search(KEYWORDS[1][1], message.lower()))


def tool_calculate(issue, booking, nights_affected, claimed):
    rate, n = booking["nightly_rate"], max(1, min(int(nights_affected or 1), booking["nights"]))
    calc = ISSUES[issue]["calc"]
    if calc == "half":      return 0.5 * rate * n, f"50% of US${rate}/night x {n} night(s)"
    if calc == "thirty":    return 0.3 * rate * n, f"30% of US${rate}/night x {n} night(s)"
    if calc == "one_night": return float(rate), "full refund of 1 night"
    if calc == "affected":  return float(rate * n), f"full refund of {n} affected night(s)"
    if calc == "booking":   return float(rate * booking["nights"]), f"full booking value ({booking['nights']} nights)"
    if calc == "claimed":   return float(claimed or 0), "host's claimed repair cost"
    return 0.0, "no payment involved"


# ======================= Step 4: Decide the route =======================
def decide(issue, policy, amount, confidence, fraud):
    reasons = []
    if ISSUES[issue].get("always_human"):
        reasons.append(f"{ISSUES[issue]['label'].lower()} cases always go to the {ISSUES[issue]['always_human']}")
    if policy["score"] < MIN_POLICY_MATCH or (issue != "other" and policy["issue"] != issue):
        reasons.append("no policy clearly matches this complaint")
    if fraud and issue != "scam":
        reasons.append("possible scam detected")
    if issue == "damage" and not amount:
        reasons.append("no claimed amount or repair quote provided")
    if amount > AUTO_LIMIT_USD:
        reasons.append(f"US${amount:,.0f} is above the US${AUTO_LIMIT_USD} automatic limit")
    if confidence < MIN_CONFIDENCE:
        reasons.append(f"evidence confidence {confidence:.0%} is below {MIN_CONFIDENCE:.0%}")
    return ("auto" if not reasons else "human"), reasons


# ======================= Steps 5-6: Outcome and explanation =======================
def outcome_text(issue, route, amount, b):
    team = ISSUES[issue].get("always_human") or "dispute specialist"
    if route == "human":
        return f"Escalated to the {team}"
    payee = ISSUES[issue]["payee"]
    if payee is None:
        return "Formal warning sent to the other party"
    who = b["host"] if payee == "host" else b["guest"]
    return f"{'Payout' if payee == 'host' else 'Refund'} of US${amount:,.0f} to {who}"


def explain(message, policy, route, outcome, basis, b):
    ai = llm("Write a short, neutral message (max 70 words) to both an Airbnb guest and host explaining this outcome. "
             f"Cite the policy title. Complaint: {message} Policy: {policy['title']}: {policy['text']} "
             f"Outcome: {outcome} ({basis}).")
    if ai:
        return ai
    if route == "auto":
        return (f"Hi {b['guest']} and {b['host']}, we've reviewed this case under our \"{policy['title']}\" policy. "
                f"Outcome: {outcome[0].lower() + outcome[1:]} ({basis}). This was resolved automatically, and either of you can appeal.")
    return (f"Hi {b['guest']} and {b['host']}, thanks for reporting this. Your case falls under our \"{policy['title']}\" "
            "policy and has been passed to a specialist, who will contact you both shortly. You don't need to resend any evidence.")


def specialist_brief(r):
    ai = llm("Summarise this dispute for a human Airbnb case specialist in 4 short bullet points: what happened, evidence, "
             f"applicable policy, suggested next step. Details: {r}")
    return ai or (f"{r['issue_label']} reported by the {r['raised_by']} on booking {r['booking_id']}: \"{r['message']}\" "
                  f"Evidence: {'attached' if r['evidence'] else 'none'}. Other party: {r['other_party'].lower()}. "
                  f"Policy: {r['policy']['title']}. Amount at stake: US${r['amount']:,.0f}. "
                  f"Escalated because {'; '.join(r['reasons'])}.")


def run_workflow(message, raised_by, booking_id, nights_affected, claimed, evidence, other_party):
    b = BOOKINGS[booking_id]
    raised_by = (raised_by or "guest").lower()
    issue, method = classify(message)                                                     # step 1
    if issue == "damage":
        raised_by = "host"                                                                # damage claims come from hosts
    senti = sentiment(message)
    policy = find_policy(message, issue)                                                  # step 2
    confidence, notes = tool_check_evidence(evidence, other_party)                        # step 3
    fraud = tool_scam_screen(message, issue)
    amount, basis = tool_calculate(issue, b, nights_affected, claimed)
    route, reasons = decide(issue, policy, amount, confidence, fraud)                     # step 4
    outcome = outcome_text(issue, route, amount, b)                                       # step 5
    r = {"message": message, "raised_by": raised_by, "booking_id": booking_id, "booking": b, "issue": issue,
         "issue_label": ISSUES[issue]["label"], "method": method, "sentiment": senti, "policy": policy,
         "confidence": confidence, "notes": notes, "fraud": fraud, "amount": amount, "basis": basis,
         "route": route, "reasons": reasons, "outcome": outcome, "evidence": evidence, "other_party": other_party}
    r["explanation"] = explain(message, policy, route, outcome, basis, b)                 # step 6
    r["brief"] = specialist_brief(r) if route == "human" else None
    return r


# ======================= Presentation =======================
GREEN, AMBER, GREY, INK, MUTED, LINE = "#0f7b4f", "#b45309", "#9aa1ab", "#1f2328", "#5b636e", "#e6e8eb"


def _chip(text, color):
    return (f'<span style="display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;'
            f'color:{color};background:{color}14;border:1px solid {color}33">{text}</span>')


def _tile(label, value, sub=""):
    return (f'<div style="flex:1;min-width:140px;padding:12px 14px;border:1px solid {LINE};border-radius:10px;background:#fff">'
            f'<div style="font-size:11px;letter-spacing:.04em;text-transform:uppercase;color:{MUTED}">{label}</div>'
            f'<div style="font-size:15px;font-weight:600;color:{INK};margin-top:4px">{value}</div>'
            f'<div style="font-size:12px;color:{MUTED};margin-top:2px">{sub}</div></div>')


def _step(n, title, concept, detail, state="done"):
    color = {"done": GREEN, "escalated": AMBER, "skipped": GREY}[state]
    dot = (f'<div style="flex:none;width:26px;height:26px;border-radius:50%;background:{color if state != "skipped" else "#fff"};'
           f'border:2px solid {color};color:{"#fff" if state != "skipped" else color};font-size:12px;font-weight:700;'
           f'display:flex;align-items:center;justify-content:center">{n}</div>')
    return (f'<div style="display:flex;gap:12px;padding:10px 0;border-bottom:1px solid {LINE};opacity:{1 if state != "skipped" else .55}">'
            f'{dot}<div style="flex:1"><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">'
            f'<span style="font-weight:600;color:{INK};font-size:14px">{title}</span>{_chip(concept, color)}</div>'
            f'<div style="font-size:13px;color:{MUTED};margin-top:3px;line-height:1.45">{detail}</div></div></div>')


def render(r):
    e = html.escape
    auto = r["route"] == "auto"
    accent = GREEN if auto else AMBER
    pol, slabel = r["sentiment"]
    priority = f" &nbsp;{_chip('Priority', '#b42318')}" if slabel == "negative" else ""
    scam = f" &nbsp;{_chip('Scam signal', '#b42318')}" if r["fraud"] else ""

    banner = (f'<div style="display:flex;justify-content:space-between;align-items:center;gap:16px;padding:16px 18px;'
              f'border-radius:12px;background:{accent}0f;border:1px solid {accent}40;border-left:5px solid {accent}">'
              f'<div><div style="font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:{accent}">'
              f'{"Resolved automatically" if auto else "Sent to a human specialist"}</div>'
              f'<div style="font-size:18px;font-weight:700;color:{INK};margin-top:4px">{e(r["outcome"])}</div>'
              f'<div style="font-size:13px;color:{MUTED};margin-top:2px">Booking {r["booking_id"]} &middot; {e(r["booking"]["where"])} '
              f'&middot; raised by the {r["raised_by"]}</div></div>'
              f'<div style="text-align:right;flex:none"><div style="font-size:11px;color:{MUTED};text-transform:uppercase;'
              f'letter-spacing:.04em">Amount</div><div style="font-size:24px;font-weight:700;color:{INK}">'
              f'{"US$" + format(r["amount"], ",.0f") if r["amount"] else "&ndash;"}</div></div></div>')

    tiles = (f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:14px 0">'
             + _tile("Issue type", e(r["issue_label"]), f"classified by {r['method']}")
             + _tile("Customer sentiment", f"{slabel.capitalize()}{priority}", f"TextBlob score {pol:+.2f}")
             + _tile("Policy match", e(r["policy"]["title"]), f"similarity {r['policy']['score']:.2f}")
             + _tile("Evidence confidence", f"{r['confidence']:.0%}{scam}", e("; ".join(r["notes"])))
             + '</div>')

    route_detail = (f"Auto-resolve: {'<b>yes</b>' if auto else 'no'} &nbsp;&middot;&nbsp; Human specialist: "
                    f"{'<b>yes</b> &mdash; ' + e('; '.join(r['reasons'])) if not auto else 'no'}")
    steps = ('<div style="border:1px solid ' + LINE + ';border-radius:12px;padding:4px 16px;background:#fff">'
             + _step(1, "Read the complaint", "LLM + sentiment analysis",
                     f"Labelled as <b>{e(r['issue_label'])}</b>; tone is {slabel}{' so the case is prioritised' if slabel == 'negative' else ''}.")
             + _step(2, "Find the relevant policy", "RAG: TF-IDF + similarity search",
                     f"Best match in the policy library: <b>{e(r['policy']['title'])}</b> (score {r['policy']['score']:.2f}).")
             + _step(3, "Check the case", "AI agent + tools",
                     f"Looked up booking {r['booking_id']}, checked the evidence ({e('; '.join(r['notes']))}), "
                     f"screened for scams{' (signal found)' if r['fraud'] else ' (none found)'}, calculated "
                     f"{'US$' + format(r['amount'], ',.0f') + ' (' + e(r['basis']) + ')' if r['amount'] else 'no payment'}.")
             + _step(4, "Decide the route", "Business rules",
                     f"Automatic only if &le; US${AUTO_LIMIT_USD}, confidence &ge; {MIN_CONFIDENCE:.0%}, a clear policy match and no scam or safety risk.")
             + _step(5, "Resolve automatically" if auto else "Hand to a human specialist",
                     "Automation" if auto else "Human-in-the-loop", route_detail, "done" if auto else "escalated")
             + _step(6, "Explain the outcome to both sides", "Generative AI", "Message below, citing the policy used.")
             + '</div>')

    message = (f'<div style="margin-top:14px;padding:14px 16px;border-radius:12px;background:#f6f8fa;border:1px solid {LINE}">'
               f'<div style="font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:{MUTED}">'
               f'Message sent to guest and host</div><div style="font-size:14px;color:{INK};margin-top:6px;line-height:1.5">'
               f'{e(r["explanation"])}</div></div>')
    brief = (f'<div style="margin-top:10px;padding:14px 16px;border-radius:12px;background:#fffaf2;border:1px solid {AMBER}40">'
             f'<div style="font-size:11px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;color:{AMBER}">'
             f'Case brief prepared for the specialist</div><div style="font-size:13px;color:{INK};margin-top:6px;line-height:1.5;'
             f'white-space:pre-wrap">{e(r["brief"])}</div></div>') if r["brief"] else ""

    heading = ('<div style="font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;'
               f'color:{MUTED};margin:6px 0 8px">{{}}</div>')
    details = (f'<details style="margin-top:14px"><summary style="cursor:pointer;font-size:13px;font-weight:600;color:{INK};'
               f'padding:6px 0">Show step-by-step details</summary><div style="margin-top:8px">{steps}</div></details>')
    return (f'<div style="font-family:Inter,-apple-system,Segoe UI,Roboto,sans-serif;color:{INK}">'
            f'{banner}{tiles}{heading.format("Route taken &nbsp;&middot;&nbsp; green = done, orange = sent to a human, grey = skipped")}{flowchart(r)}'
            f'{message}{brief}{details}</div>')


def flowchart(r):
    """Compact flowchart of the six steps, with the path this case took highlighted."""
    e = html.escape
    auto = r["route"] == "auto"

    def node(title, detail, state="done"):
        c = {"done": GREEN, "escalated": AMBER, "skipped": GREY}[state]
        on = state != "skipped"
        return (f'<div style="flex:1 1 120px;min-width:120px;padding:9px 10px;border-radius:10px;border:1.5px solid {c};'
                f'background:{c + "12" if on else "#fff"};opacity:{1 if on else .5}">'
                f'<div style="font-size:12px;font-weight:700;color:{INK}">{title}</div>'
                f'<div style="font-size:11.5px;color:{MUTED};margin-top:3px;line-height:1.35">{detail}</div></div>')

    arrow = f'<div style="align-self:center;color:{GREY};font-size:16px;flex:none">&rarr;</div>'
    either = f'<div style="align-self:center;color:{MUTED};font-size:11px;font-weight:700;flex:none">or</div>'
    row = '<div style="display:flex;gap:6px;align-items:stretch;flex-wrap:wrap;margin-bottom:6px">{}</div>'
    top = row.format(arrow.join([
        node("1. Read complaint", e(r["issue_label"])),
        node("2. Find policy", e(r["policy"]["title"])),
        node("3. Check case", f"US${r['amount']:,.0f} &middot; {r['confidence']:.0%} confidence" if r["amount"]
             else f"{r['confidence']:.0%} confidence"),
        node("4. Decide route", "Clear-cut: yes" if auto else "Clear-cut: no"),
    ]))
    bottom = row.format(
        node("5a. Resolve automatically", e(r["outcome"]) if auto else "Skipped", "done" if auto else "skipped")
        + either
        + node("5b. Human specialist", e(r["outcome"]) if not auto else "Skipped", "escalated" if not auto else "skipped")
        + arrow
        + node("6. Explain to both sides", "Message below"))
    return f'<div style="padding:12px;border:1px solid {LINE};border-radius:12px;background:#fff">{top}{bottom}</div>'


EMPTY = (f'<div style="font-family:Inter,-apple-system,Segoe UI,Roboto,sans-serif;border:1px dashed #cfd4da;border-radius:12px;'
         f'padding:48px 24px;text-align:center;color:{MUTED}"><div style="font-size:16px;font-weight:600;color:{INK}">No case yet</div>'
         f'<div style="font-size:14px;margin-top:6px">Describe a complaint on the left, or click one of the example cases.</div></div>')


def run(message, raised_by, booking_id, nights, claimed, evidence, other_party):
    if not message or not message.strip():
        return EMPTY
    return render(run_workflow(message.strip(), raised_by, booking_id, nights, claimed, evidence, other_party))


# ======================= App layout =======================
EXAMPLES = [
    ["This is terrible. There has been no hot water since we checked in and the boiler shows an error light.",
     "Guest", "HM-1001", 3, 0, True, "Confirms it"],
    ["The guest cracked the TV screen and left wine stains on the sofa. Repair quotes attached.",
     "Host", "HM-2002", 1, 5000, True, "Disputes it"],
    ["The host messaged me on WhatsApp asking me to pay by bank transfer to get a discount.",
     "Guest", "HM-3003", 1, 0, True, "No reply yet"],
    ["The room was filthy when we arrived, with hair on the sheets and rubbish left in the kitchen.",
     "Guest", "HM-1001", 1, 0, True, "No reply yet"],
    ["We were locked out at midnight because the door code didn't work and the host wasn't answering.",
     "Guest", "HM-3003", 1, 0, True, "No reply yet"],
    ["The flat is nothing like the photos. It's much smaller and the advertised pool doesn't exist.",
     "Guest", "HM-2002", 2, 0, True, "Disputes it"],
    ["The guests threw a loud party until 3am and the neighbours complained.",
     "Host", "HM-2002", 1, 0, True, "No reply yet"],
    ["I can smell gas in the kitchen and there is no smoke alarm.",
     "Guest", "HM-1001", 2, 0, False, "No reply yet"],
]
EXAMPLE_LABELS = ["No hot water", "Damaged TV (US$5,000)", "Bank transfer request", "Dirty room",
                  "Locked out", "Not as described", "Party / noise", "Gas smell"]

THEME = gr.themes.Soft(primary_hue="rose", neutral_hue="slate", font=[gr.themes.GoogleFont("Inter"), "sans-serif"]).set(
    block_title_background_fill="transparent", block_title_text_color="*neutral_700",
    block_label_background_fill="transparent", block_label_text_color="*neutral_700",
    block_title_text_weight="600")
CSS = """
.gradio-container {max-width: 1240px !important; margin: 0 auto !important;}
#hero {padding: 8px 4px 4px;}
#run-btn {font-weight: 600;}
footer {display: none !important;}
"""
FORCE_LIGHT = """() => { const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'light') { u.searchParams.set('__theme', 'light'); window.location.replace(u.href); } }"""

with gr.Blocks(title="AI Dispute Assistant | Airbnb case study") as demo:
    gr.HTML(f"""
    <div id="hero" style="font-family:Inter,-apple-system,Segoe UI,Roboto,sans-serif">
      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap">
        <span style="font-size:24px;font-weight:700;color:{INK}">AI Dispute Assistant</span>
        {_chip('Prototype', '#e11d48')} {_chip('Case study: Airbnb', MUTED)}
      </div>
      <div style="font-size:14px;color:{MUTED};margin-top:6px;max-width:820px;line-height:1.5">
        Resolves guest&ndash;host disputes in six steps: it reads the complaint, finds the matching policy, checks the evidence,
        and then either settles the case automatically or hands it to a human specialist with a ready-made brief.
      </div>
    </div>""")

    with gr.Row(equal_height=False):
        with gr.Column(scale=2, min_width=340):
            with gr.Group():
                message = gr.Textbox(label="Complaint", lines=4,
                                     placeholder="Describe what happened, e.g. 'The Wi-Fi hasn't worked since we arrived.'")
                with gr.Row():
                    raised_by = gr.Radio(["Guest", "Host"], value="Guest", label="Raised by")
                    booking = gr.Dropdown(list(BOOKINGS), value="HM-1001", label="Booking")
                with gr.Row():
                    nights = gr.Number(value=1, minimum=1, maximum=5, precision=0, label="Nights affected")
                    claimed = gr.Number(value=0, minimum=0, label="Claimed amount (US$, damage only)")
                evidence = gr.Checkbox(value=True, label="Photos or documents attached")
                other_party = gr.Radio(["No reply yet", "Confirms it", "Disputes it"], value="No reply yet",
                                       label="Other party's response")
            go = gr.Button("Analyse case", variant="primary", size="lg", elem_id="run-btn")
            examples_slot = gr.Column()
        with gr.Column(scale=3, min_width=420):
            result = gr.HTML(EMPTY)

    inputs = [message, raised_by, booking, nights, claimed, evidence, other_party]
    with examples_slot:
        gr.Examples(EXAMPLES, inputs=inputs, outputs=result, fn=run, run_on_click=True, cache_examples=False,
                    example_labels=EXAMPLE_LABELS, label="Example cases (click to run)")
    go.click(run, inputs, result)
    message.submit(run, inputs, result)

    gr.HTML(f'<div style="font-family:Inter,sans-serif;font-size:12px;color:{MUTED};text-align:center;padding:18px 0 4px">'
            'BC3415 student project. Not affiliated with Airbnb. Bookings, names and policies are simplified samples.</div>')

if __name__ == "__main__":
    # Render passes the port in PORT; SHARE=1 gives a temporary public link (e.g. from Colab)
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),
                share=bool(os.environ.get("SHARE")), theme=THEME, css=CSS, js=FORCE_LIGHT)
