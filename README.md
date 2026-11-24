# Homework 3: Class Scheduling using Genetic Algorithm
**Course:** Introduction to Artificial Intelligence (CO3061) - HCMUT  
**Instructor:** Nguyễn Quốc Minh, Ph.D.  
**Class:** CC01 | **Group:** 4  

---

## 1. Project Overview & Problem Specifications
This repository contains the source code for Homework 3: Automated Class Scheduling using Genetic Algorithm for the course Introduction to Artificial Intelligence (CO3061), Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology (HCMUT).

The problem requires constructing a conflict-free and optimized weekly academic timetable for:
- **Courses:** 60 courses (`c01` – `c60`)
- **Sections:** 480 sections (8 sections per course)
- **Sessions:** 960 lecture sessions (2 sessions per section, 1 session = 1 timeslot)
- **Professors:** 60 instructors (`p01` – `p60`), max 20 sessions/week
- **Days:** 5 days (Monday to Friday, 0..4)
- **Timeslots:** 6 slots per day (Slots 1–3: Morning, Slots 4–6: Afternoon)
- **Classrooms:** 35 rooms (`rm01` – `rm35`) with sizes {0: Small, 1: Average, 2: Large}
- **Section Registration Sizes:** {0: Small, 1: Average, 2: Large} (must be $\le$ room size)

### Constraints & Objectives:
- **Hard Constraints (H1–H6) [0 Violations - 100% Feasible]**:
  - **H1 (Room Capacity):** Room size $\ge$ Section size (0 violations).
  - **H2 (Day Separation):** 2 sessions on non-adjacent, different days ($|d_1 - d_2| > 1$, e.g., Mon & Thu; 0 violations).
  - **H3 (Professor Collision & Workload):** No simultaneous teaching collisions and no professor exceeds 20 sessions (0 collisions, 0 overloads).
  - **H4 (Available Classrooms):** Only operational rooms assigned (0 errors).
  - **H5 (Section Uniqueness):** A section never assigned to two rooms at the same time (0 overlaps).
  - **H6 (Room Collision):** At most 1 section per classroom at any timeslot (0 collisions).
- **Soft Constraints (S1–S4) [Optimized]**:
  - **S1 (Professor Compactness):** Penalizes idle gaps between teaching periods in a day.
  - **S2 (Professor Half-Day):** Encourages morning-only or afternoon-only teaching schedules per day.
  - **S3 (Course Half-Day):** Groups sections of each course into morning or afternoon blocks.
  - **S4 (Room Capacity Waste):** Minimizes unused classroom capacity.
- **Bonus Objectives (B1–B2)**:
  - **B1 (Favorite Courses):** Matching professors with their top-3 preferred courses (satisfied for 59-60/60 professors).
  - **B2 (Qualified Teaching):** Minimizes teaching outside qualified course domains.

---

## 2. Group Workload Allocation (Group 4)
- **HW3 Project Team (Product Implementation & Modeling):**
  1. **Nguyễn Trần Vĩ (2353338):** Project Lead, Genetic Algorithm Architecture, Problem Modeling & Constraint Validation.
  2. **Hồ Trọng Trí (2453303):** Genetic Operators (Crossover, Mutation), Fitness Function Formulation & Convergence Analysis.
  3. **Võ Thành Nguyên Khang (2452483):** Graphical User Interface (GUI), Benchmarking & Data Serialization.

- **Presentation & Seminar Defense Team (HW1 & HW2):**
  4. **Nguyễn Ngọc Sáng (2353046)**
  5. **Nguyễn Gia Hùng (2352422)**
  6. **Nguyễn Văn Nghĩa (2452827)**
  7. **Ngô Tuấn Vĩ (2453437)**

---

## 3. Project Structure
```
HW3/
├── main.py                     # Genetic Algorithm solver engine with CLI & verification audit
├── gui.py                      # Interactive Tkinter GUI visualizer
├── Run_Scheduler_GUI.bat       # Launcher script for Windows
├── final_class_schedule.csv    # Exported timetable (960 sessions) in CSV format
├── final_class_schedule.json   # Exported timetable in JSON format
├── fitness_convergence_plot.png# Convergence plot (Fitness & Constraint Violations)
├── requirements.txt            # Python dependencies (matplotlib, numpy, Pillow)
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── figures/                    # Figures and plot exports
└── report/                     # LaTeX report files
```

---

## 4. How to Run

### Option A: Graphical User Interface (GUI)
Double-click `Run_Scheduler_GUI.bat` on Windows, or run:
```bash
python gui.py
```
**GUI Features:**
- **Control Panel:** Configure generations and population size, click `Run Evolution`, `Step 1 Gen`, or `Reset`.
- **Status & Metrics:** Real-time generation counter, best fitness, constraint violation counts, and soft penalties breakdown.
- **Timetable Explorer:**
  - View by **Room** (35 classrooms x 6 timeslots) for any selected day.
  - View by **Professor** (60 professors x 6 timeslots) showing assigned courses and favorites.
  - Search filter by course, section, professor, or room.
- **Convergence Plot:** Visualizes fitness progression and constraint violation decay.
- **Constraint Audit:** Complete checklist of H1-H6, S1-S4, and bonus criteria.
- **Export:** Export schedule to `final_class_schedule.csv` and `final_class_schedule.json`.

### Option B: Command Line (CLI)
Run the solver directly from terminal:
```bash
python main.py
```
To customize parameters:
```bash
python main.py --generations 30 --pop 40
```

---

## 5. Verification Checklist & Results
Upon completion, `main.py` automatically validates all constraints:
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
