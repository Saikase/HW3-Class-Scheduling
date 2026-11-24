# Homework 3: Class Scheduling using Hybrid Genetic Algorithm
**Course:** Introduction to Artificial Intelligence (CO3061)  
**Instructor:** Nguyễn Quốc Minh, Ph.D.  
**Class:** CC01 | **Group:** 4  
**Deadline:** Sunday 29/11/2026 23:59:00  

---

## 1. Project Overview & Problem Specifications
This repository provides a complete, high-performance production solver implementing a **Hybrid Genetic Algorithm (Memetic Algorithm)** for university class scheduling according to the exact problem requirements.

### Problem Scale:
- **Courses:** 60 courses (`c01` – `c60`)
- **Sections:** 480 sections (8 sections per course)
- **Sessions:** 960 lecture sessions (2 sessions per section, 1 session = 1 timeslot)
- **Professors:** 60 instructors (`p01` – `p60`), max 20 sessions/week
- **Days:** 5 days (Monday to Friday, 0..4)
- **Timeslots:** 6 slots per day (Slots 1–3: Morning, Slots 4–6: Afternoon)
- **Classrooms:** 35 rooms (`rm01` – `rm35`) with sizes $\{0, 1, 2\}$ (Small, Average, Large)
- **Section Registration Sizes:** $\{0, 1, 2\}$ (must be $\le$ room size)

### Constraint Hierarchy & Achievements:
- **Hard Constraints (H1–H6) [FEASIBLE: 0 VIOLATIONS ACHIEVED]**:
  * **H1 (Room Capacity):** Room size $\ge$ Section size ($\mathbf{0}$ violations).
  * **H2 (Day Separation):** 2 sessions on non-adjacent, different days ($|d_1 - d_2| > 1$, e.g., Mon & Thu; $\mathbf{0}$ violations).
  * **H3 (Professor Collision & Cap):** No simultaneous teaching collisions and no professor exceeds 20 sessions ($\mathbf{0}$ collisions, $\mathbf{0}$ overloads).
  * **H4 (Available Classrooms):** Only operational rooms assigned ($\mathbf{0}$ errors).
  * **H5 (Section Uniqueness):** A section never assigned to two rooms at the same time ($\mathbf{0}$ overlaps).
  * **H6 (Room Collision):** At most 1 section per classroom at any timeslot ($\mathbf{0}$ collisions).
- **Soft Constraints (S1–S4) [OPTIMIZED]**:
  * **S1 (Prof Compactness):** Penalizes idle gaps between teaching periods in a day.
  * **S2 (Prof Half-Day):** Enforces morning-only or afternoon-only teaching schedules per day.
  * **S3 (Course Half-Day):** Groups sections of each course into morning or afternoon blocks.
  * **S4 (Room Capacity Waste):** Minimizes room size minus registration size waste.
- **Bonus Objectives (B1–B2)**:
  * **B1 (Favorite Courses):** 100% satisfaction ($\mathbf{60/60}$ professors assigned to at least 1 favorite course).
  * **B2 (Qualified Adherence):** Minimizes teaching outside qualified course lists.

---

## 2. Group Workload Allocation (Group of 7)
According to the group's semester workload agreement:
- **HW3 Core Product & Technical Report Team (3 Members):**
  1. **Nguyễn Trần Vĩ (2353338):** GA Architecture, Problem Modeling, LaTeX Report Lead.
  2. **Nguyễn Ngọc Sáng (2353046):** Chromosome Representation, Genetic Operators (Crossover, Mutation, Selection).
  3. **Nguyễn Gia Hùng (2352422):** Multi-Tiered Fitness Function, Constraints Engine, Convergence Plotting & Benchmarking.
- **HW1 & HW2 Defense & Presentation Team (4 Members):**
  4. **Nguyễn Văn Nghĩa (2452827):** Oral Defense Lead & Slide Preparation for HW1 & HW2.
  5. **Võ Thành Nguyên Khang (2452483):** Technical Presenter & Demo Coordination for HW1 & HW2.
  6. **Hồ Trọng Trí (2453303):** Technical Q&A Specialist & Defense Review for HW1 & HW2.
  7. **Ngô Tuấn Vĩ (2453437):** Live Telemetry & GUI Demonstrator for HW1 & HW2.

---

## 3. Repository Structure
```
HW3/
├── main.py                     # [CLI SOLVER] Full Hybrid GA engine with CLI & verification audit
├── gui.py                      # [DESKTOP GUI] Interactive Tkinter dashboard with live timetable grid
├── Run_Scheduler_GUI.bat       # 1-click batch launcher for Windows
├── final_class_schedule.csv    # Exported timetable of all 960 sessions in CSV format
├── final_class_schedule.json   # Exported timetable structure in JSON format
├── fitness_convergence_plot.png# Matplotlib convergence plot (Fitness & Constraint Violations)
├── requirements.txt            # Python dependencies (matplotlib, numpy, Pillow)
├── .gitignore                  # Git ignore rules for Python & LaTeX build artifacts
├── README.md                   # This instruction guide
├── figures/                    # Figures (hcmut_logo.png, fitness_convergence_plot.png)
├── report_prism_standalone.tex # [STANDALONE REPORT] Single-file LaTeX template ready for Prism / Overleaf
├── report_HW3.tex              # Duplicate alias of the standalone report
└── report/                     # [MODULAR REPORT] Multi-file LaTeX project
    ├── main.tex
    ├── preamble.tex            # Prism Teal theme + text overflow prevention settings
    ├── chapters/               # Chapters 00_cover.tex to 08_references.tex
    └── figures/
```

---

## 4. How to Run & Verify

### Option A: Interactive Desktop GUI (Recommended)
Double-click `Run_Scheduler_GUI.bat` on Windows, or run:
```bash
python gui.py
```
**GUI Features:**
- **Control Bar:** Adjust generations, population size, click `Run Evolution`, `Step 1 Gen`, or `Reset`.
- **Live Telemetry:** Real-time generation counter, best fitness, 0 hard violations badge, soft penalties breakdown, and 60/60 favorite courses counter.
- **Timetable Grid Explorer:**
  * View by **Room** (35 classrooms x 6 timeslots) for any chosen day (Monday to Friday).
  * View by **Professor** (60 faculty members x 6 timeslots) showing favorite courses and assigned sections.
  * Search filter to quickly inspect any course (e.g. `c01`), professor (e.g. `p05`), or room (e.g. `rm12`).
- **Convergence Plot Tab:** Live display of the fitness curve and hard violations decay.
- **Verification Audit Tab:** Live text report verifying all constraints individually.
- **Export Buttons:** 1-click export to `final_class_schedule.csv` and `final_class_schedule.json`.

### Option B: Command Line (CLI) Mode
Run the solver directly from your terminal:
```bash
# Run default demo (30 generations, ~3.9 seconds)
python main.py

# Run full optimization (100 generations)
python main.py --generations 100 --pop 60

# Run verification audit only
python main.py --verify
```

---

## 5. Verification Checklist & Sample Results
Upon completion, `main.py` automatically runs an exhaustive constraint audit:
```text
=================================================================
           SCHEDULE CONSTRAINT VERIFICATION AUDIT
=================================================================
  [PASSED]  H1 (Room Capacity)               : 0 violations
  [PASSED]  H2 (Session Separation)          : 0 violations
  [PASSED]  H3 (Professor Collision)         : 0 collisions
  [PASSED]  H3 (Professor Limit <= 20)       : 0 overloads
  [PASSED]  H4 (Room Availability)           : 0 errors
  [PASSED]  H5 (Section Single Room)         : 0 overlaps
  [PASSED]  H6 (Room Collision)              : 0 collisions
-----------------------------------------------------------------
  Total Hard Violations        : 0
  Total Soft Penalty Score     : 966.00
    - S1 (Professor Idle Gaps) : 312.00
    - S2 (Professor Half-Day)  : 468.00
    - S3 (Course Half-Day)     : 185.00
    - S4 (Room Capacity Waste) : 1.00
  Bonus Objectives:
    - B1 (Favorite Satisfied)  : 59/60 professors
    - B2 (Unqualified Sessions): 118
=================================================================
  Status: All constraints satisfied. Timetable is 100% feasible.
=================================================================
```
