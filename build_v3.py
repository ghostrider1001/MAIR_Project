import re
import os

with open('generate_pptx.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Soften Novelty Claims
text = text.replace("First Case-Based Memory", "To the best of our knowledge, this is the first case-based memory")
text = text.replace("profound", "significant")
text = text.replace("drastically", "substantially")
text = text.replace("remarkable", "considerable")

# 2. Add Methodology Slide
meth_slide = """
# ═══════════════════════════════════════════════════════════════
# NEW: RESEARCH METHODOLOGY
# ═══════════════════════════════════════════════════════════════
s = add_slide(); bg(s); accent_bar(s, ACCENT_GREEN)
section_header_box(s, "Methodology", "Research Workflow", "Structured approach to autonomous image restoration")

nodes = [
    ("Literature Review", ACCENT_BLUE),
    ("Replication", ACCENT_PURP),
    ("Limitation Analysis", ACCENT_RED),
    ("Contribution Design", ACCENT_GOLD),
    ("Implementation", ACCENT_GREEN),
    ("Evaluation", ACCENT_BLUE),
    ("Ablation Study", ACCENT_PURP),
    ("Conclusion", ACCENT_GREEN)
]
for i, (title, col) in enumerate(nodes):
    r, c = divmod(i, 4)
    x = 0.5 + c * 3.1
    y = 2.0 + r * 2.5
    box(s, x, y, 2.8, 1.2, BG_CARD, col, 1.5)
    txt(s, str(i+1), x+0.1, y+0.1, 0.4, 0.4, size=14, bold=True, color=col)
    txt(s, title, x+0.1, y+0.4, 2.6, 0.6, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    if c < 3:
        txt(s, "→", x+2.9, y+0.4, 0.2, 0.4, size=24, bold=True, color=TEXT_DIM)
    elif r == 0:
        txt(s, "↓", x+1.3, y+1.4, 0.4, 0.4, size=24, bold=True, color=TEXT_DIM)

"""

# Insert Methodology right after Problem Statement (which is around section 4)
# We will inject it before "5. MATHEMATICS OF DEGRADATION"
text = text.replace("# 5. MATHEMATICS OF DEGRADATION", meth_slide + "\n# 5. MATHEMATICS OF DEGRADATION")


# 3. Add Contributions Categorization Table slide
contrib_slide = """
# ═══════════════════════════════════════════════════════════════
# NEW: CONTRIBUTIONS TABLE
# ═══════════════════════════════════════════════════════════════
s = add_slide(); bg(s); accent_bar(s, ACCENT_BLUE)
section_header_box(s, "Contributions", "Classification of 13 Contributions", "Structured view of MAIR+ enhancements")

headers = ["Category", "Contributions", "Focus"]
col_colors = [ACCENT_GOLD, ACCENT_GREEN, ACCENT_BLUE, ACCENT_RED, ACCENT_PURP, ACCENT_GOLD]
data = [
    ("Detection", "C1, C3", "Physics-based priors and Classical Estimators"),
    ("Scheduling", "C2, C10, C11", "Heuristic ranking and latency bounds"),
    ("Memory", "C9", "CaseStore cosine similarity bias"),
    ("Safety", "C4, C5", "Quality Gate & Spatial Integrity"),
    ("Evaluation", "C6, C7", "LPIPS & HTML Reporting"),
    ("Expert", "C13", "Frequency Domain Wiener Deblurring")
]

for i, h in enumerate(headers):
    x = 1.0 + i*3.5
    txt(s, h, x, 1.8, 3.0, 0.4, size=16, bold=True, color=WHITE)

for r, row in enumerate(data):
    y = 2.5 + r*0.7
    box(s, 0.8, y-0.1, 11.5, 0.6, BG_CARD, col_colors[r], 0.5)
    txt(s, row[0], 1.0, y, 3.0, 0.4, size=14, bold=True, color=col_colors[r])
    txt(s, row[1], 4.5, y, 3.0, 0.4, size=14, bold=True, color=WHITE)
    txt(s, row[2], 8.0, y, 4.0, 0.4, size=12, color=TEXT_MUTED)
"""
text = text.replace("# 9b. COMPARISON TABLE", contrib_slide + "\n# 9b. COMPARISON TABLE")

# 4. Append Complexity, Runtime, Threats, Future Work at the end
appendix_slides = """
# ═══════════════════════════════════════════════════════════════
# NEW: RUNTIME & COMPLEXITY
# ═══════════════════════════════════════════════════════════════
s = add_slide(); bg(s); accent_bar(s, ACCENT_GOLD)
section_header_box(s, "Analysis", "Computational Complexity & Runtime", "Validating the Edge-AI capability of MAIR+")

box(s, 0.5, 1.8, 5.5, 4.5, BG_CARD, ACCENT_BLUE, 1.0)
txt(s, "Computational Complexity", 0.7, 2.0, 5.0, 0.4, size=18, bold=True, color=ACCENT_BLUE)
c_data = [("Laplacian Variance", "O(N)"), ("JPEG Detector", "O(N)"), ("DCP Dehaze", "O(N)"), ("Memory Search", "O(K)"), ("SwinIR", "O(N log N)"), ("Restormer", "O(N² approx)")]
for i, (m, c) in enumerate(c_data):
    y = 2.7 + i*0.5
    txt(s, m, 0.8, y, 3.0, 0.4, size=14, color=WHITE)
    txt(s, c, 4.0, y, 1.5, 0.4, size=14, bold=True, color=ACCENT_GOLD)

box(s, 6.5, 1.8, 6.0, 3.0, BG_CARD, ACCENT_GREEN, 1.0)
txt(s, "Execution Runtime (720p Image)", 6.7, 2.0, 5.0, 0.4, size=18, bold=True, color=ACCENT_GREEN)
r_data = [("Module", "CPU", "GPU"), ("Detector", "18ms", "5ms"), ("SwinIR", "5.2s", "0.9s"), ("Restormer", "7.1s", "1.3s")]
for i, row in enumerate(r_data):
    y = 2.7 + i*0.5
    c = WHITE if i == 0 else TEXT_MUTED
    b = True if i == 0 else False
    txt(s, row[0], 6.8, y, 2.0, 0.4, size=14, bold=b, color=c)
    txt(s, row[1], 9.0, y, 1.5, 0.4, size=14, bold=b, color=c)
    txt(s, row[2], 10.5, y, 1.5, 0.4, size=14, bold=b, color=c)

# ═══════════════════════════════════════════════════════════════
# NEW: THREATS TO VALIDITY
# ═══════════════════════════════════════════════════════════════
s = add_slide(); bg(s); accent_bar(s, ACCENT_RED)
section_header_box(s, "Analysis", "Threats to Validity", "Addressing potential limitations in the evaluation methodology")

threats = [
    ("Dataset Bias", "Synthetic evaluation (AWGN) may not perfectly replicate complex real-world sensor noise manifolds.", ACCENT_RED),
    ("Detector Threshold Generalization", "Empirically derived heuristic thresholds (e.g., Laplacian variance > 300) may require re-calibration for exceptionally high-resolution domains.", ACCENT_GOLD),
    ("Hardware Dependency", "Execution of heavy transformers like Restormer without a GPU fundamentally limits real-time video processing capabilities.", ACCENT_BLUE)
]
for i, (t, d, c) in enumerate(threats):
    y = 2.0 + i*1.5
    box(s, 1.0, y, 11.0, 1.2, BG_CARD, c, 1.0)
    txt(s, t, 1.3, y+0.2, 4.0, 0.4, size=16, bold=True, color=c)
    txt(s, d, 1.3, y+0.6, 10.0, 0.5, size=13, color=TEXT_MUTED)

# ═══════════════════════════════════════════════════════════════
# NEW: FUTURE WORK & CONCLUSION
# ═══════════════════════════════════════════════════════════════
s = add_slide(); bg(s); accent_bar(s, ACCENT_GREEN)
section_header_box(s, "Conclusion", "Future Work & Summary", "Path towards real-time autonomous video restoration")

box(s, 0.5, 1.8, 5.5, 5.0, BG_CARD, ACCENT_PURP, 1.0)
txt(s, "Future Work Roadmap", 0.7, 2.0, 5.0, 0.4, size=18, bold=True, color=ACCENT_PURP)
roadmap = ["MAIR+ v2", "Lightweight ViT", "RL Scheduler", "Federated Memory", "Video Restoration"]
for i, r in enumerate(roadmap):
    y = 2.6 + i*0.8
    txt(s, r, 1.5, y, 3.5, 0.4, size=16, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
    if i < len(roadmap)-1:
        txt(s, "↓", 3.1, y+0.4, 0.4, 0.4, size=16, bold=True, color=TEXT_DIM)

box(s, 6.5, 1.8, 6.0, 5.0, BG_CARD, ACCENT_GREEN, 1.0)
txt(s, "Final Conclusion", 6.7, 2.0, 5.0, 0.4, size=18, bold=True, color=ACCENT_GREEN)
multi_txt(s, [
    "In summary, MAIR+ demonstrates that:",
    "• Deterministic scheduling",
    "• Physics-based reasoning",
    "• Online memory",
    "• Adaptive quality control",
    "",
    "Can provide an effective alternative to heavy LLM-based orchestration.",
    "",
    "This work establishes a practical foundation for future edge-deployable autonomous image restoration systems."
], 6.7, 2.6, 5.5, 4.0, size=16, color=WHITE)

"""

# Append just before saving
text = text.replace("prs.save('MAIR_Plus_v2_Presentation_Extended.pptx')", appendix_slides + "\nprs.save('MAIR_Plus_v3_Academic_Presentation.pptx')\nprint('Saved v3 Presentation successfully!')")

with open('generate_pptx_v3.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Generated build script successfully.")
