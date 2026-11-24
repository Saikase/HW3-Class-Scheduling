"""
University Class Scheduling using Genetic Algorithm
Desktop GUI Application
Course: Introduction to Artificial Intelligence (CO3061) - HCMUT
Instructor: Nguyen Quoc Minh, Ph.D.
Class: CC01 | Group: 4
HW3 Team Members:
- Nguyễn Trần Vĩ (2353338)
- Hồ Trọng Trí (2453303)
- Võ Thành Nguyên Khang (2452483)
"""

import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional

from main import (
    ProblemInstance,
    HybridGeneticAlgorithm,
    Individual,
    DAY_NAMES,
    NUM_DAYS,
    NUM_TIMESLOTS,
    NUM_ROOMS,
    NUM_PROFESSORS,
    NUM_COURSES,
    export_schedule_to_csv,
    export_schedule_to_json,
    plot_fitness_history,
    verify_schedule
)

# Optional image display using PIL if installed
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class SchedulerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("University Class Scheduling - Genetic Algorithm | HCMUT CO3061")
        self.root.geometry("1200x780")
        self.root.minsize(1050, 680)

        # Style & Color Palette
        self.primary_teal = "#1A6B72"
        self.light_teal = "#D6EDEF"
        self.bg_gray = "#F8FAFC"
        self.accent_green = "#10B981"
        self.accent_red = "#EF4444"

        # State Variables
        self.instance: Optional[ProblemInstance] = None
        self.ga: Optional[HybridGeneticAlgorithm] = None
        self.current_schedule: Optional[Individual] = None
        self.is_running = False
        self.target_generations = 30
        self.pop_size = 40
        self.seed = 42

        self._init_theme()
        self._build_layout()
        self._reset_problem()

    def _init_theme(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TFrame", background=self.bg_gray)
        style.configure("TLabelframe", background=self.bg_gray, foreground=self.primary_teal)
        style.configure("TLabelframe.Label", font=("Helvetica", 10, "bold"), foreground=self.primary_teal)

        style.configure("Primary.TButton", font=("Helvetica", 9, "bold"),
                        background=self.primary_teal, foreground="white")
        style.map("Primary.TButton",
                  background=[("active", "#134E54"), ("pressed", "#0E3B3F")])

        style.configure("Success.TButton", font=("Helvetica", 9, "bold"),
                        background=self.accent_green, foreground="white")

        style.configure("Treeview.Heading", font=("Helvetica", 9, "bold"),
                        background=self.light_teal, foreground=self.primary_teal)
        style.configure("Treeview", font=("Consolas", 9), rowheight=24)

    def _build_layout(self):
        # 1. Header Banner
        header = tk.Frame(self.root, bg=self.primary_teal, height=60)
        header.pack(fill="x", side="top")

        title_lbl = tk.Label(
            header,
            text="UNIVERSITY CLASS SCHEDULING - GENETIC ALGORITHM",
            font=("Helvetica", 13, "bold"),
            fg="white",
            bg=self.primary_teal
        )
        title_lbl.pack(side="left", padx=20, pady=8)

        sub_lbl = tk.Label(
            header,
            text="CO3061 - Introduction to AI | HCMUT | Group 4 (CC01)",
            font=("Helvetica", 10),
            fg="#CCFBF1",
            bg=self.primary_teal
        )
        sub_lbl.pack(side="right", padx=20, pady=10)

        # 2. Main Content Split
        main_container = ttk.Frame(self.root)
        main_container.pack(fill="both", expand=True, padx=12, pady=10)

        # Left Column: Controls & Telemetry Dashboard (width ~340)
        left_pane = ttk.Frame(main_container, width=340)
        left_pane.pack(side="left", fill="y", padx=(0, 10))

        self._build_control_panel(left_pane)
        self._build_telemetry_panel(left_pane)

        # Right Column: Notebook Tabs (Timetable Explorer, Convergence Plot, Audit Log)
        right_pane = ttk.Frame(main_container)
        right_pane.pack(side="right", fill="both", expand=True)

        self.notebook = ttk.Notebook(right_pane)
        self.notebook.pack(fill="both", expand=True)

        # Tab 1: Timetable Explorer
        self.tab_timetable = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_timetable, text="Timetable Explorer")
        self._build_timetable_tab()

        # Tab 2: Convergence Plot
        self.tab_plot = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_plot, text="Convergence Plot")
        self._build_plot_tab()

        # Tab 3: Verification Audit Log
        self.tab_log = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_log, text="Verification Audit")
        self._build_log_tab()

    def _build_control_panel(self, parent):
        ctrl_frame = ttk.LabelFrame(parent, text="Algorithm Controls", padding=10)
        ctrl_frame.pack(fill="x", pady=(0, 10))

        # Parameters row
        row1 = ttk.Frame(ctrl_frame)
        row1.pack(fill="x", pady=2)
        ttk.Label(row1, text="Generations:").pack(side="left")
        self.gen_var = tk.StringVar(value="30")
        ttk.Entry(row1, textvariable=self.gen_var, width=6).pack(side="right")

        row2 = ttk.Frame(ctrl_frame)
        row2.pack(fill="x", pady=2)
        ttk.Label(row2, text="Population Size:").pack(side="left")
        self.pop_var = tk.StringVar(value="40")
        ttk.Entry(row2, textvariable=self.pop_var, width=6).pack(side="right")

        row3 = ttk.Frame(ctrl_frame)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="Random Seed:").pack(side="left")
        self.seed_var = tk.StringVar(value="42")
        ttk.Entry(row3, textvariable=self.seed_var, width=6).pack(side="right")

        # Buttons
        btn_frame = ttk.Frame(ctrl_frame)
        btn_frame.pack(fill="x", pady=(8, 2))

        self.btn_run = ttk.Button(btn_frame, text="Run Evolution", style="Primary.TButton", command=self._start_evolution_thread)
        self.btn_run.pack(side="left", expand=True, fill="x", padx=2)

        self.btn_step = ttk.Button(btn_frame, text="Step 1 Gen", command=self._step_once)
        self.btn_step.pack(side="left", expand=True, fill="x", padx=2)

        self.btn_reset = ttk.Button(btn_frame, text="Reset", command=self._reset_problem)
        self.btn_reset.pack(side="left", expand=True, fill="x", padx=2)

        # Action exports
        act_frame = ttk.Frame(ctrl_frame)
        act_frame.pack(fill="x", pady=(6, 0))

        ttk.Button(act_frame, text="Export CSV", command=self._export_csv).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(act_frame, text="Export JSON", command=self._export_json).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(act_frame, text="Verify Audit", command=self._run_audit).pack(side="left", expand=True, fill="x", padx=2)

    def _build_telemetry_panel(self, parent):
        tele_frame = ttk.LabelFrame(parent, text="Optimization Status", padding=10)
        tele_frame.pack(fill="both", expand=True)

        # Status badge
        self.status_badge = tk.Label(
            tele_frame, text="FEASIBLE: 0 HARD VIOLATIONS",
            font=("Helvetica", 10, "bold"),
            bg="#D1FAE5", fg="#065F46", relief="solid", bd=1, pady=4
        )
        self.status_badge.pack(fill="x", pady=(0, 10))

        # Metric grid
        grid_frame = ttk.Frame(tele_frame)
        grid_frame.pack(fill="both", expand=True)

        self.lbl_gen = self._create_metric_row(grid_frame, "Generation:", "0 / 30")
        self.lbl_fitness = self._create_metric_row(grid_frame, "Best Fitness:", "0.0000")
        self.lbl_hard = self._create_metric_row(grid_frame, "Hard Violations:", "0", highlight=True)
        self.lbl_h1 = self._create_metric_row(grid_frame, "  • H1 (Room/Section Size):", "0")
        self.lbl_h2 = self._create_metric_row(grid_frame, "  • H2 (Day Separation):", "0")
        self.lbl_h3 = self._create_metric_row(grid_frame, "  • H3 (Prof Conflict/Cap):", "0")
        self.lbl_h6 = self._create_metric_row(grid_frame, "  • H6 (Room Collisions):", "0")
        self.lbl_soft = self._create_metric_row(grid_frame, "Soft Penalty Total:", "0.0")
        self.lbl_s1 = self._create_metric_row(grid_frame, "  • S1 (Prof Idle Gaps):", "0.0")
        self.lbl_s2 = self._create_metric_row(grid_frame, "  • S2 (Prof Half-Day):", "0.0")
        self.lbl_s3 = self._create_metric_row(grid_frame, "  • S3 (Course Half-Day):", "0.0")
        self.lbl_s4 = self._create_metric_row(grid_frame, "  • S4 (Capacity Waste):", "0.0")
        self.lbl_fav = self._create_metric_row(grid_frame, "Favorites (Bonus 1):", "60 / 60")
        self.lbl_time = self._create_metric_row(grid_frame, "Elapsed Compute Time:", "0.0s")

    def _create_metric_row(self, parent, label_text: str, default_val: str, highlight: bool = False):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        font = ("Helvetica", 9, "bold") if highlight else ("Helvetica", 9)
        fg = self.primary_teal if highlight else "black"
        ttk.Label(row, text=label_text, font=font).pack(side="left")
        val_lbl = tk.Label(row, text=default_val, font=font, fg=fg, bg=self.bg_gray)
        val_lbl.pack(side="right")
        return val_lbl

    def _build_timetable_tab(self):
        filter_bar = ttk.Frame(self.tab_timetable, padding=6)
        filter_bar.pack(fill="x", side="top")

        ttk.Label(filter_bar, text="Day:", font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 4))
        self.day_combo = ttk.Combobox(filter_bar, values=DAY_NAMES, state="readonly", width=8)
        self.day_combo.set(DAY_NAMES[0])
        self.day_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_timetable_view())
        self.day_combo.pack(side="left", padx=(0, 15))

        ttk.Label(filter_bar, text="View Mode:", font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 4))
        self.view_mode = ttk.Combobox(filter_bar, values=["By Room (35 rooms)", "By Professor (60 profs)", "By Course (60 courses)"], state="readonly", width=22)
        self.view_mode.set("By Room (35 rooms)")
        self.view_mode.bind("<<ComboboxSelected>>", lambda e: self._refresh_timetable_view())
        self.view_mode.pack(side="left", padx=(0, 15))

        ttk.Label(filter_bar, text="Search Filter:").pack(side="left", padx=(0, 4))
        self.search_var = tk.StringVar()
        self.search_var.trace("w", lambda *args: self._refresh_timetable_view())
        ttk.Entry(filter_bar, textvariable=self.search_var, width=15).pack(side="left")

        # Table Grid
        cols = ["Item", "Type", "Slot 1 (M)", "Slot 2 (M)", "Slot 3 (M)", "Slot 4 (A)", "Slot 5 (A)", "Slot 6 (A)"]
        self.tree = ttk.Treeview(self.tab_timetable, columns=cols, show="headings", selectmode="browse")

        for col in cols:
            w = 110 if col.startswith("Slot") else (130 if col == "Item" else 70)
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")

        tree_scroll = ttk.Scrollbar(self.tab_timetable, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)

        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")

    def _build_plot_tab(self):
        container = ttk.Frame(self.tab_plot, padding=10)
        container.pack(fill="both", expand=True)

        top_ctrl = ttk.Frame(container)
        top_ctrl.pack(fill="x", pady=(0, 8))
        ttk.Button(top_ctrl, text="Reload Plot Image", command=self._load_plot_image).pack(side="left")

        self.plot_label = tk.Label(container, text="[Convergence plot will render here once evolution completes]", bg="white", relief="sunken")
        self.plot_label.pack(fill="both", expand=True)

    def _build_log_tab(self):
        container = ttk.Frame(self.tab_log, padding=10)
        container.pack(fill="both", expand=True)

        self.log_text = tk.Text(container, font=("Consolas", 10), bg="#0F172A", fg="#38BDF8", wrap="word")
        log_scroll = ttk.Scrollbar(container, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)

        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")

    # -------------------------------------------------------------------------
    # CORE LOGIC BINDINGS
    # -------------------------------------------------------------------------

    def _reset_problem(self):
        try:
            self.target_generations = int(self.gen_var.get())
            self.pop_size = int(self.pop_var.get())
            self.seed = int(self.seed_var.get())
        except ValueError:
            self.target_generations = 30
            self.pop_size = 40
            self.seed = 42

        self.instance = ProblemInstance.generate(seed=self.seed)
        self.ga = HybridGeneticAlgorithm(
            instance=self.instance,
            pop_size=self.pop_size,
            seed=self.seed
        )
        self.ga.initialize_population()
        self.ga.history = []
        self.current_schedule = self.ga.population[0]
        self._update_telemetry(gen=0, elapsed=0.0)
        self._refresh_timetable_view()
        self._run_audit()

    def _start_evolution_thread(self):
        if self.is_running:
            return
        self.is_running = True
        self.btn_run.config(state="disabled")
        threading.Thread(target=self._run_evolution_loop, daemon=True).start()

    def _run_evolution_loop(self):
        start_time = time.time()
        gens = self.target_generations
        self.ga.history = []

        for g in range(gens):
            best = self.ga.population[0]
            avg_fit = sum(c.fitness for c in self.ga.population) / len(self.ga.population)
            self.ga.history.append({
                "generation": g + 1,
                "best_fitness": best.fitness,
                "avg_fitness": avg_fit,
                "hard_violations": best.metrics.total_hard,
                "soft_penalty": best.metrics.total_soft,
                "fav_satisfied": best.metrics.b1_fav_satisfied,
            })
            self.current_schedule = best
            elapsed = time.time() - start_time
            self.root.after(0, self._update_telemetry, g + 1, elapsed)

            if g < gens - 1:
                self.ga.step(current_gen=g + 1)
            time.sleep(0.02)

        self.is_running = False
        self.root.after(0, self._on_evolution_finish)

    def _step_once(self):
        if self.is_running:
            return
        cur_gen = len(self.ga.history) + 1
        self.ga.step(current_gen=cur_gen)
        self.current_schedule = self.ga.population[0]
        best = self.current_schedule
        avg_fit = sum(c.fitness for c in self.ga.population) / len(self.ga.population)
        self.ga.history.append({
            "generation": cur_gen,
            "best_fitness": best.fitness,
            "avg_fitness": avg_fit,
            "hard_violations": best.metrics.total_hard,
            "soft_penalty": best.metrics.total_soft,
            "fav_satisfied": best.metrics.b1_fav_satisfied,
        })
        self._update_telemetry(gen=cur_gen, elapsed=0.0)
        self._refresh_timetable_view()
        self._run_audit()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(base_dir, "fitness_convergence_plot.png")
        plot_fitness_history(self.ga.history, save_path=plot_path)
        self._load_plot_image()

    def _on_evolution_finish(self):
        self.btn_run.config(state="normal")
        self._refresh_timetable_view()
        self._run_audit()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(base_dir, "fitness_convergence_plot.png")
        plot_fitness_history(self.ga.history, save_path=plot_path)
        fig_dir = os.path.join(base_dir, "figures")
        if os.path.exists(fig_dir):
            import shutil
            try:
                shutil.copy2(plot_path, os.path.join(fig_dir, "fitness_convergence_plot.png"))
            except Exception:
                pass
        self._load_plot_image()
        status_text = "100% FEASIBLE (0 VIOLATIONS)" if self.current_schedule.metrics.total_hard == 0 else f"{self.current_schedule.metrics.total_hard} VIOLATIONS"
        messagebox.showinfo("Optimization Complete", f"Evolution finished {self.target_generations} generations!\nResult: {status_text}\nBest Fitness: {self.current_schedule.fitness:.4f}")

    def _update_telemetry(self, gen: int, elapsed: float):
        if not self.current_schedule:
            return
        m = self.current_schedule.metrics
        self.lbl_gen.config(text=f"{gen} / {self.target_generations}")
        self.lbl_fitness.config(text=f"{self.current_schedule.fitness:.4f}")
        self.lbl_hard.config(text=f"{m.total_hard}")
        self.lbl_h1.config(text=f"{m.h1_room_size}")
        self.lbl_h2.config(text=f"{m.h2_day_separation}")
        self.lbl_h3.config(text=f"{m.h3_prof_conflict + m.h3_prof_overload}")
        self.lbl_h6.config(text=f"{m.h6_room_conflict}")
        self.lbl_soft.config(text=f"{m.total_soft:.1f}")
        self.lbl_s1.config(text=f"{m.s1_prof_scatter:.1f}")
        self.lbl_s2.config(text=f"{m.s2_prof_half_day:.1f}")
        self.lbl_s3.config(text=f"{m.s3_course_half_day:.1f}")
        self.lbl_s4.config(text=f"{m.s4_room_waste:.1f}")
        self.lbl_fav.config(text=f"{m.b1_fav_satisfied} / 60")
        self.lbl_time.config(text=f"{elapsed:.2f}s")

        if m.total_hard == 0:
            self.status_badge.config(text="FEASIBLE: 0 HARD VIOLATIONS", bg="#D1FAE5", fg="#065F46")
        else:
            self.status_badge.config(text=f"CONFLICTS: {m.total_hard} HARD VIOLATIONS", bg="#FEE2E2", fg="#991B1B")

    def _refresh_timetable_view(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        if not self.current_schedule or not self.instance:
            return

        chosen_day = DAY_NAMES.index(self.day_combo.get())
        mode = self.view_mode.get()
        search = self.search_var.get().lower().strip()

        # Build lookup table: item_id -> [slot0, slot1, slot2, slot3, slot4, slot5]
        if "Room" in mode:
            # 35 rooms
            grid = {r: ["" for _ in range(NUM_TIMESLOTS)] for r in range(NUM_ROOMS)}
            for sec_id, gene in enumerate(self.current_schedule.genes):
                sec = self.instance.sections[sec_id]
                p_name = self.instance.professors[gene.prof_id].name
                label = f"c{sec.course_id+1:02d}.s{sec.section_num} ({p_name})"
                if gene.day1 == chosen_day:
                    grid[gene.room1][gene.slot1] = label
                if gene.day2 == chosen_day:
                    grid[gene.room2][gene.slot2] = label

            for r_id in range(NUM_ROOMS):
                r_obj = self.instance.rooms[r_id]
                sz_str = ["Small", "Avg", "Large"][r_obj.size]
                row_vals = [r_obj.name, sz_str] + grid[r_id]
                if not search or any(search in str(v).lower() for v in row_vals):
                    self.tree.insert("", "end", values=row_vals)

        elif "Professor" in mode:
            # 60 professors
            grid = {p: ["" for _ in range(NUM_TIMESLOTS)] for p in range(NUM_PROFESSORS)}
            for sec_id, gene in enumerate(self.current_schedule.genes):
                sec = self.instance.sections[sec_id]
                p = gene.prof_id
                if gene.day1 == chosen_day:
                    grid[p][gene.slot1] = f"c{sec.course_id+1:02d} in {self.instance.rooms[gene.room1].name}"
                if gene.day2 == chosen_day:
                    grid[p][gene.slot2] = f"c{sec.course_id+1:02d} in {self.instance.rooms[gene.room2].name}"

            for p_id in range(NUM_PROFESSORS):
                p_obj = self.instance.professors[p_id]
                fav_str = f"Favs:{[f'c{c+1:02d}' for c in p_obj.favorites]}"
                row_vals = [p_obj.name, fav_str] + grid[p_id]
                if not search or any(search in str(v).lower() for v in row_vals):
                    self.tree.insert("", "end", values=row_vals)

    def _load_plot_image(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(base_dir, "fitness_convergence_plot.png")
        if not os.path.exists(plot_path) or not HAS_PIL:
            return

        try:
            img = Image.open(plot_path)
            # Resize to fit frame
            w, h = 800, 420
            img = img.resize((w, h), Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(img)
            self.plot_label.config(image=self.tk_img, text="")
        except Exception as e:
            self.plot_label.config(text=f"Error displaying image: {e}")

    def _run_audit(self):
        if not self.current_schedule:
            return
        m = self.current_schedule.metrics
        self.log_text.delete("1.0", tk.END)

        audit_str = (
            "======================================================================\n"
            "           SCHEDULE CONSTRAINT VERIFICATION AUDIT\n"
            "======================================================================\n"
            f"  [{'PASSED' if m.h1_room_size==0 else 'FAILED'}]  H1 (Room Capacity)               : {m.h1_room_size} violations\n"
            f"  [{'PASSED' if m.h2_day_separation==0 else 'FAILED'}]  H2 (Session Separation)          : {m.h2_day_separation} violations\n"
            f"  [{'PASSED' if m.h3_prof_conflict==0 else 'FAILED'}]  H3 (Professor Collision)         : {m.h3_prof_conflict} collisions\n"
            f"  [{'PASSED' if m.h3_prof_overload==0 else 'FAILED'}]  H3 (Professor Limit <= 20)       : {m.h3_prof_overload} overloads\n"
            f"  [{'PASSED' if m.h4_room_availability==0 else 'FAILED'}]  H4 (Room Availability)           : {m.h4_room_availability} errors\n"
            f"  [{'PASSED' if m.h5_section_overlap==0 else 'FAILED'}]  H5 (Section Single Room)         : {m.h5_section_overlap} overlaps\n"
            f"  [{'PASSED' if m.h6_room_conflict==0 else 'FAILED'}]  H6 (Room Collision)              : {m.h6_room_conflict} collisions\n"
            "----------------------------------------------------------------------\n"
            f"  Total Hard Violations        : {m.total_hard}\n"
            f"  Total Soft Penalty Score     : {m.total_soft:.2f}\n"
            f"    - S1 (Professor Idle Gaps) : {m.s1_prof_scatter:.2f}\n"
            f"    - S2 (Professor Half-Day)  : {m.s2_prof_half_day:.2f}\n"
            f"    - S3 (Course Half-Day)     : {m.s3_course_half_day:.2f}\n"
            f"    - S4 (Room Capacity Waste) : {m.s4_room_waste:.2f}\n"
            "  Bonus Objectives:\n"
            f"    - B1 (Favorite Satisfied)  : {m.b1_fav_satisfied} / 60 professors\n"
            f"    - B2 (Unqualified Sessions): {m.b2_unqualified_count} sessions\n"
            "======================================================================\n"
        )
        if m.total_hard == 0:
            audit_str += "  Status: All constraints satisfied. Timetable is 100% feasible.\n"
            audit_str += "======================================================================\n"
        else:
            audit_str += f"  Status: Infeasible ({m.total_hard} hard violations remaining).\n"
            audit_str += "======================================================================\n"

        self.log_text.insert(tk.END, audit_str)

    def _export_csv(self):
        if not self.current_schedule:
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "final_class_schedule.csv")
        export_schedule_to_csv(self.current_schedule, self.instance, filepath=path)
        messagebox.showinfo("Export Successful", f"Saved schedule to:\n{path}")

    def _export_json(self):
        if not self.current_schedule:
            return
        base_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_dir, "final_class_schedule.json")
        export_schedule_to_json(self.current_schedule, self.instance, filepath=path)
        messagebox.showinfo("Export Successful", f"Saved schedule to:\n{path}")


def launch_gui():
    root = tk.Tk()
    app = SchedulerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
