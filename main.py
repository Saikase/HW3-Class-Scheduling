"""
Homework 3: Class Scheduling using Genetic Algorithm
Course: Introduction to Artificial Intelligence (CO3061) - HCMUT
Instructor: Dr. Nguyen Quoc Minh
Group: 4 | Class: CC01

This template provides a modular and extensible architecture for solving the
Class Scheduling problem using Genetic Algorithms as specified in the assignment.

Problem Parameters:
- num_courses = 60 (c01 - c60)
- num_sections = 480 (8 sections per course)
- num_sessions = 960 (2 sessions per section, 1 session = 1 timeslot)
- num_professors = 60 (p01 - p60)
- num_days = 5 (0: Mon, 1: Tue, 2: Wed, 3: Thu, 4: Fri)
- num_timeslots_per_day = 6 (0..2: Morning, 3..5: Afternoon)
- num_rooms = 35 (rm01 - rm35)
- room_size in {0: Small, 1: Average, 2: Large}
- section_registration_size in {0: Small, 1: Average, 2: Large}
"""

import sys
import os
import random
import time
import json
import csv
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Optional

# Optional matplotlib import for convergence plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# 1. CONSTANTS & DOMAIN SPECIFICATIONS
# =============================================================================

NUM_COURSES = 60
SECTIONS_PER_COURSE = 8
NUM_SECTIONS = NUM_COURSES * SECTIONS_PER_COURSE       # 480
SESSIONS_PER_SECTION = 2
NUM_SESSIONS = NUM_SECTIONS * SESSIONS_PER_SECTION     # 960

NUM_PROFESSORS = 60
MAX_SESSIONS_PER_PROF = 20

NUM_DAYS = 5                     # Mon(0), Tue(1), Wed(2), Thu(3), Fri(4)
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]
NUM_TIMESLOTS_PER_DAY = 6        # 0,1,2 = Morning; 3,4,5 = Afternoon
TOTAL_TIMESLOTS = NUM_DAYS * NUM_TIMESLOTS_PER_DAY  # 30 slots per room

NUM_ROOMS = 35                   # rm01 - rm35
ROOM_SIZES = [0, 1, 2]           # 0: Small, 1: Average, 2: Large
REGISTRATION_SIZES = [0, 1, 2]

# Valid 2-day pairs for a section: NOT on same day, NOT on adjacent days
VALID_DAY_PAIRS = [
    (d1, d2)
    for d1 in range(NUM_DAYS)
    for d2 in range(NUM_DAYS)
    if d1 < d2 and abs(d1 - d2) > 1
]
# Valid pairs: (Mon, Wed), (Mon, Thu), (Mon, Fri), (Tue, Thu), (Tue, Fri), (Wed, Fri)


# =============================================================================
# 2. DATA MODELS
# =============================================================================

@dataclass
class Room:
    id: int                      # 0 .. 34
    name: str                    # rm01 .. rm35
    size: int                    # 0, 1, 2


@dataclass
class Professor:
    id: int                      # 0 .. 59
    name: str                    # p01 .. p60
    favorite_courses: List[int]  # List of 3 favorite course IDs (Bonus 1)
    qualified_courses: Set[int]  # Set of courses this prof can teach (Bonus 2)


@dataclass
class SessionInfo:
    session_id: int              # 0 .. 959
    section_id: int              # 0 .. 479
    course_id: int               # 0 .. 59
    session_index: int           # 0 or 1 (first or second session of the section)
    registration_size: int       # 0, 1, 2


@dataclass
class Gene:
    day: int                     # 0 .. 4
    timeslot: int                # 0 .. 5
    room_id: int                 # 0 .. 34
    professor_id: int            # 0 .. 59


@dataclass
class ProblemInstance:
    rooms: List[Room]
    professors: List[Professor]
    sessions: List[SessionInfo]
    section_to_sessions: Dict[int, Tuple[int, int]]
    course_to_sections: Dict[int, List[int]]

    @classmethod
    def generate_random_instance(cls, seed: int = 42) -> "ProblemInstance":
        """
        Initializes room sizes, section registration sizes, and professor preferences
        using a reproducible random seed as required by the assignment.
        """
        rng = random.Random(seed)

        # 1. Initialize Rooms (35 rooms with balanced distribution of sizes)
        rooms = []
        for r_id in range(NUM_ROOMS):
            # E.g., ~10 small, ~15 average, ~10 large
            if r_id < 10:
                r_size = 0
            elif r_id < 25:
                r_size = 1
            else:
                r_size = 2
            rooms.append(Room(id=r_id, name=f"rm{r_id+1:02d}", size=r_size))

        # 2. Initialize Professors & Preferences
        professors = []
        for p_id in range(NUM_PROFESSORS):
            # Top 3 favorite courses (Bonus 1)
            favs = rng.sample(range(NUM_COURSES), 3)
            # Qualified courses: includes the 3 favorites plus 5-10 additional courses
            additional_qualified = rng.sample(range(NUM_COURSES), rng.randint(6, 12))
            qualified = set(favs).union(additional_qualified)
            professors.append(
                Professor(
                    id=p_id,
                    name=f"p{p_id+1:02d}",
                    favorite_courses=favs,
                    qualified_courses=qualified
                )
            )

        # 3. Initialize Sections and Sessions
        sessions = []
        section_to_sessions = {}
        course_to_sections = {c: [] for c in range(NUM_COURSES)}

        session_counter = 0
        section_counter = 0

        for c_id in range(NUM_COURSES):
            for _ in range(SECTIONS_PER_COURSE):
                sec_id = section_counter
                course_to_sections[c_id].append(sec_id)
                # Random registration size for this section: {0, 1, 2}
                reg_size = rng.choice(REGISTRATION_SIZES)

                s1_id = session_counter
                s2_id = session_counter + 1
                session_counter += 2

                sessions.append(SessionInfo(
                    session_id=s1_id,
                    section_id=sec_id,
                    course_id=c_id,
                    session_index=0,
                    registration_size=reg_size
                ))
                sessions.append(SessionInfo(
                    session_id=s2_id,
                    section_id=sec_id,
                    course_id=c_id,
                    session_index=1,
                    registration_size=reg_size
                ))

                section_to_sessions[sec_id] = (s1_id, s2_id)
                section_counter += 1

        return cls(
            rooms=rooms,
            professors=professors,
            sessions=sessions,
            section_to_sessions=section_to_sessions,
            course_to_sections=course_to_sections
        )


# =============================================================================
# 3. CHROMOSOME & EVALUATION ENGINE
# =============================================================================

@dataclass
class EvaluationMetrics:
    # Hard constraint violations
    h1_room_size_violations: int = 0
    h2_day_separation_violations: int = 0
    h3_prof_conflict_violations: int = 0
    h3_prof_overload_violations: int = 0
    h4_room_availability_violations: int = 0
    h5_section_overlap_violations: int = 0
    h6_room_conflict_violations: int = 0

    # Soft constraint penalty points
    s1_prof_scatter_penalty: float = 0.0
    s2_prof_half_day_penalty: float = 0.0
    s3_course_half_day_penalty: float = 0.0
    s4_room_waste_penalty: float = 0.0

    # Bonus objectives
    b1_prof_favorite_satisfaction: int = 0    # number of profs with >=1 fav class
    b2_unqualified_teach_count: int = 0

    @property
    def total_hard_violations(self) -> int:
        return (
            self.h1_room_size_violations
            + self.h2_day_separation_violations
            + self.h3_prof_conflict_violations
            + self.h3_prof_overload_violations
            + self.h4_room_availability_violations
            + self.h5_section_overlap_violations
            + self.h6_room_conflict_violations
        )

    @property
    def total_soft_penalty(self) -> float:
        return (
            self.s1_prof_scatter_penalty
            + self.s2_prof_half_day_penalty
            + self.s3_course_half_day_penalty
            + self.s4_room_waste_penalty
        )


class Chromosome:
    """
    A Chromosome represents a complete schedule for all 960 sessions.
    Represented as a list of Gene objects indexed by session_id.
    """
    def __init__(self, genes: List[Gene]):
        self.genes = genes
        self.fitness: float = -1.0
        self.metrics: Optional[EvaluationMetrics] = None

    def clone(self) -> "Chromosome":
        copied_genes = [
            Gene(g.day, g.timeslot, g.room_id, g.professor_id)
            for g in self.genes
        ]
        c = Chromosome(copied_genes)
        c.fitness = self.fitness
        c.metrics = self.metrics
        return c


class ScheduleEvaluator:
    """
    Evaluates chromosomes against all Hard Constraints, Soft Constraints,
    and Bonus Objectives. Computes a mathematically grounded fitness score.
    """
    def __init__(
        self,
        instance: ProblemInstance,
        w_hard: float = 200.0,
        w_soft: float = 1.0,
        w_bonus: float = 5.0
    ):
        self.instance = instance
        self.w_hard = w_hard
        self.w_soft = w_soft
        self.w_bonus = w_bonus

    def evaluate(self, chromosome: Chromosome) -> Tuple[float, EvaluationMetrics]:
        m = EvaluationMetrics()
        inst = self.instance

        # Lookup tables for rapid conflict detection
        room_time_occupancy: Dict[Tuple[int, int, int], int] = {}  # (room, day, slot) -> count
        prof_time_occupancy: Dict[Tuple[int, int, int], int] = {}  # (prof, day, slot) -> count
        prof_session_count = [0] * NUM_PROFESSORS
        prof_daily_slots: Dict[Tuple[int, int], List[int]] = {}    # (prof, day) -> list of slots

        # Track course timeslot distribution for S3
        course_morning_slots = [0] * NUM_COURSES
        course_afternoon_slots = [0] * NUM_COURSES

        # Track professor favorite course assignment for B1
        prof_courses_taught: Dict[int, Set[int]] = {p: set() for p in range(NUM_PROFESSORS)}

        # ---------------------------------------------------------------------
        # 1. Single-session checks (H1, H4, S4, B2) & Occupancy Accumulation
        # ---------------------------------------------------------------------
        for s_info, gene in zip(inst.sessions, chromosome.genes):
            r = inst.rooms[gene.room_id]

            # H1: Room/Section size: room_size >= section_size
            if r.size < s_info.registration_size:
                m.h1_room_size_violations += 1

            # H4: Available classrooms check (valid id within 0..NUM_ROOMS-1)
            if not (0 <= gene.room_id < NUM_ROOMS):
                m.h4_room_availability_violations += 1

            # S4: Room size and section registration size match
            # If room_size > reg_size, wasted capacity
            if r.size > s_info.registration_size:
                m.s4_room_waste_penalty += (r.size - s_info.registration_size) * 0.5

            # Accumulate room occupancy: (room, day, slot)
            key_room = (gene.room_id, gene.day, gene.timeslot)
            room_time_occupancy[key_room] = room_time_occupancy.get(key_room, 0) + 1

            # Accumulate prof occupancy & workload
            key_prof = (gene.professor_id, gene.day, gene.timeslot)
            prof_time_occupancy[key_prof] = prof_time_occupancy.get(key_prof, 0) + 1
            prof_session_count[gene.professor_id] += 1

            # Prof daily slots for S1, S2
            pd_key = (gene.professor_id, gene.day)
            if pd_key not in prof_daily_slots:
                prof_daily_slots[pd_key] = []
            prof_daily_slots[pd_key].append(gene.timeslot)

            # S3: Track course morning (0,1,2) vs afternoon (3,4,5)
            if gene.timeslot < 3:
                course_morning_slots[s_info.course_id] += 1
            else:
                course_afternoon_slots[s_info.course_id] += 1

            # Bonus tracking
            prof_courses_taught[gene.professor_id].add(s_info.course_id)
            prof_obj = inst.professors[gene.professor_id]
            if s_info.course_id not in prof_obj.qualified_courses:
                m.b2_unqualified_teach_count += 1

        # ---------------------------------------------------------------------
        # 2. Section-level checks (H2, H5)
        # ---------------------------------------------------------------------
        for sec_id, (s1_id, s2_id) in inst.section_to_sessions.items():
            g1 = chromosome.genes[s1_id]
            g2 = chromosome.genes[s2_id]

            # H2: 2 sessions per week, NOT on same or adjacent days
            # Invalid if same day (d1 == d2) or adjacent (abs(d1 - d2) == 1)
            if g1.day == g2.day or abs(g1.day - g2.day) <= 1:
                m.h2_day_separation_violations += 1

            # H5: A section cannot be scheduled in two different rooms simultaneously
            if g1.day == g2.day and g1.timeslot == g2.timeslot:
                m.h5_section_overlap_violations += 1

        # ---------------------------------------------------------------------
        # 3. Room & Professor Collision Checks (H3, H6)
        # ---------------------------------------------------------------------
        # H6: Room collision (two sections assigned to same room at same time)
        for count in room_time_occupancy.values():
            if count > 1:
                m.h6_room_conflict_violations += (count - 1)

        # H3: Prof simultaneous teaching collision
        for count in prof_time_occupancy.values():
            if count > 1:
                m.h3_prof_conflict_violations += (count - 1)

        # H3: Prof workload cap (max 20 sessions)
        for count in prof_session_count:
            if count > MAX_SESSIONS_PER_PROF:
                m.h3_prof_overload_violations += (count - MAX_SESSIONS_PER_PROF)

        # ---------------------------------------------------------------------
        # 4. Soft Constraints (S1, S2, S3)
        # ---------------------------------------------------------------------
        # S1 & S2: Professor schedule compactness & half-day consistency
        for (p_id, day), slots in prof_daily_slots.items():
            if len(slots) > 1:
                sorted_slots = sorted(slots)
                span = sorted_slots[-1] - sorted_slots[0] + 1
                idle_gaps = span - len(sorted_slots)
                m.s1_prof_scatter_penalty += idle_gaps * 1.0

            # S2: morning (0,1,2) vs afternoon (3,4,5)
            has_morning = any(s < 3 for s in slots)
            has_afternoon = any(s >= 3 for s in slots)
            if has_morning and has_afternoon:
                m.s2_prof_half_day_penalty += 2.0

        # S3: Course sessions in morning only or afternoon only
        for c_id in range(NUM_COURSES):
            m_cnt = course_morning_slots[c_id]
            a_cnt = course_afternoon_slots[c_id]
            # Penalty proportional to minority fraction
            if m_cnt > 0 and a_cnt > 0:
                m.s3_course_half_day_penalty += min(m_cnt, a_cnt) * 1.0

        # ---------------------------------------------------------------------
        # 5. Bonus Objectives (B1, B2)
        # ---------------------------------------------------------------------
        # B1: Each prof has at least 1 favorite class
        for p_id in range(NUM_PROFESSORS):
            favs = set(inst.professors[p_id].favorite_courses)
            taught = prof_courses_taught[p_id]
            if len(favs.intersection(taught)) >= 1:
                m.b1_prof_favorite_satisfaction += 1

        # ---------------------------------------------------------------------
        # 6. Composite Fitness Score Computation
        # ---------------------------------------------------------------------
        # Penalty formula: F = 1000.0 / (1.0 + w_h * hard + w_s * soft - bonus)
        hard_cost = self.w_hard * m.total_hard_violations
        soft_cost = self.w_soft * m.total_soft_penalty
        bonus_reward = self.w_bonus * (
            m.b1_prof_favorite_satisfaction - (m.b2_unqualified_teach_count * 0.2)
        )

        effective_denominator = 1.0 + hard_cost + soft_cost - max(0.0, bonus_reward)
        if effective_denominator < 1.0:
            effective_denominator = 1.0

        fitness = 1000.0 / effective_denominator
        return fitness, m


# =============================================================================
# 4. GENETIC OPERATORS & GA ENGINE
# =============================================================================

class GeneticAlgorithm:
    def __init__(
        self,
        instance: ProblemInstance,
        pop_size: int = 50,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.05,
        elitism_count: int = 3,
        tournament_size: int = 5,
        seed: int = 42
    ):
        self.instance = instance
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.rng = random.Random(seed)
        self.evaluator = ScheduleEvaluator(instance)

        self.population: List[Chromosome] = []
        self.history: List[Dict[str, float]] = []

    def initialize_population(self):
        """
        Creates an initial population with smart initialization to satisfy H2
        (day separation) out of the box whenever possible.
        """
        self.population = []
        for _ in range(self.pop_size):
            genes = [None] * NUM_SESSIONS

            # Assign valid day pairs per section to minimize H2 immediately
            for sec_id, (s1_id, s2_id) in self.instance.section_to_sessions.items():
                day1, day2 = self.rng.choice(VALID_DAY_PAIRS)
                slot1 = self.rng.randint(0, NUM_TIMESLOTS_PER_DAY - 1)
                slot2 = self.rng.randint(0, NUM_TIMESLOTS_PER_DAY - 1)

                # Room assignment (filter compatible rooms to reduce H1)
                reg_size = self.instance.sessions[s1_id].registration_size
                compatible_rooms = [
                    r.id for r in self.instance.rooms if r.size >= reg_size
                ]
                r1 = self.rng.choice(compatible_rooms) if compatible_rooms else self.rng.randint(0, NUM_ROOMS - 1)
                r2 = self.rng.choice(compatible_rooms) if compatible_rooms else self.rng.randint(0, NUM_ROOMS - 1)

                p1 = self.rng.randint(0, NUM_PROFESSORS - 1)
                p2 = p1  # Often the same professor teaches both lectures of a section

                genes[s1_id] = Gene(day=day1, timeslot=slot1, room_id=r1, professor_id=p1)
                genes[s2_id] = Gene(day=day2, timeslot=slot2, room_id=r2, professor_id=p2)

            ch = Chromosome(genes)
            fit, metrics = self.evaluator.evaluate(ch)
            ch.fitness = fit
            ch.metrics = metrics
            self.population.append(ch)

    def tournament_selection(self) -> Chromosome:
        competitors = self.rng.sample(self.population, self.tournament_size)
        competitors.sort(key=lambda c: c.fitness, reverse=True)
        return competitors[0]

    def section_preserving_crossover(
        self, parent1: Chromosome, parent2: Chromosome
    ) -> Tuple[Chromosome, Chromosome]:
        """
        Uniform section-level crossover: passes both sessions of a section
        together to avoid corrupting section structure.
        """
        if self.rng.random() > self.crossover_rate:
            return parent1.clone(), parent2.clone()

        child1_genes = [None] * NUM_SESSIONS
        child2_genes = [None] * NUM_SESSIONS

        for sec_id, (s1_id, s2_id) in self.instance.section_to_sessions.items():
            if self.rng.random() < 0.5:
                child1_genes[s1_id] = Gene(*parent1.genes[s1_id].__dict__.values())
                child1_genes[s2_id] = Gene(*parent1.genes[s2_id].__dict__.values())
                child2_genes[s1_id] = Gene(*parent2.genes[s1_id].__dict__.values())
                child2_genes[s2_id] = Gene(*parent2.genes[s2_id].__dict__.values())
            else:
                child1_genes[s1_id] = Gene(*parent2.genes[s1_id].__dict__.values())
                child1_genes[s2_id] = Gene(*parent2.genes[s2_id].__dict__.values())
                child2_genes[s1_id] = Gene(*parent1.genes[s1_id].__dict__.values())
                child2_genes[s2_id] = Gene(*parent1.genes[s2_id].__dict__.values())

        return Chromosome(child1_genes), Chromosome(child2_genes)

    def mutate(self, chromosome: Chromosome):
        """
        Multi-component mutation operator:
        - Mutates section day pair (maintaining H2 validity)
        - Mutates timeslot
        - Mutates room
        - Mutates professor
        """
        for sec_id, (s1_id, s2_id) in self.instance.section_to_sessions.items():
            if self.rng.random() < self.mutation_rate:
                # 1. Day pair reassignment
                if self.rng.random() < 0.3:
                    d1, d2 = self.rng.choice(VALID_DAY_PAIRS)
                    chromosome.genes[s1_id].day = d1
                    chromosome.genes[s2_id].day = d2

                # 2. Timeslot mutation
                if self.rng.random() < 0.4:
                    s = s1_id if self.rng.random() < 0.5 else s2_id
                    chromosome.genes[s].timeslot = self.rng.randint(0, NUM_TIMESLOTS_PER_DAY - 1)

                # 3. Room mutation (prefer size-compatible)
                if self.rng.random() < 0.4:
                    s = s1_id if self.rng.random() < 0.5 else s2_id
                    reg = self.instance.sessions[s].registration_size
                    compat = [r.id for r in self.instance.rooms if r.size >= reg]
                    chromosome.genes[s].room_id = (
                        self.rng.choice(compat) if compat else self.rng.randint(0, NUM_ROOMS - 1)
                    )

                # 4. Professor mutation
                if self.rng.random() < 0.3:
                    new_prof = self.rng.randint(0, NUM_PROFESSORS - 1)
                    chromosome.genes[s1_id].professor_id = new_prof
                    chromosome.genes[s2_id].professor_id = new_prof

    def step(self):
        # 1. Sort current population by fitness
        self.population.sort(key=lambda c: c.fitness, reverse=True)

        new_population: List[Chromosome] = []

        # 2. Elitism: retain top individuals
        for i in range(self.elitism_count):
            new_population.append(self.population[i].clone())

        # 3. Offspring generation
        while len(new_population) < self.pop_size:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            c1, c2 = self.section_preserving_crossover(p1, p2)
            self.mutate(c1)
            self.mutate(c2)

            f1, m1 = self.evaluator.evaluate(c1)
            c1.fitness = f1
            c1.metrics = m1
            new_population.append(c1)

            if len(new_population) < self.pop_size:
                f2, m2 = self.evaluator.evaluate(c2)
                c2.fitness = f2
                c2.metrics = m2
                new_population.append(c2)

        self.population = new_population
        self.population.sort(key=lambda c: c.fitness, reverse=True)

    def run(self, generations: int = 100, verbose: bool = True) -> Chromosome:
        if not self.population:
            self.initialize_population()

        start_time = time.time()
        for gen in range(generations):
            self.step()
            best = self.population[0]
            avg_fit = sum(c.fitness for c in self.population) / len(self.population)

            record = {
                "generation": gen + 1,
                "best_fitness": best.fitness,
                "avg_fitness": avg_fit,
                "hard_violations": best.metrics.total_hard_violations,
                "soft_penalty": best.metrics.total_soft_penalty,
                "fav_profs": best.metrics.b1_prof_favorite_satisfaction,
            }
            self.history.append(record)

            if verbose and ((gen + 1) % 5 == 0 or gen == 0 or (gen + 1) == generations):
                elapsed = time.time() - start_time
                print(
                    f"Gen {gen+1:03d}/{generations:03d} | "
                    f"Best Fit: {best.fitness:8.4f} | "
                    f"Avg Fit: {avg_fit:8.4f} | "
                    f"Hard Viol: {best.metrics.total_hard_violations:3d} | "
                    f"Soft Pen: {best.metrics.total_soft_penalty:6.1f} | "
                    f"Fav Satisfied: {best.metrics.b1_prof_favorite_satisfaction:2d}/60 | "
                    f"Time: {elapsed:5.1f}s"
                )

        return self.population[0]


# =============================================================================
# 5. EXPORT & PLOTTING UTILITIES
# =============================================================================

def export_schedule_to_csv(
    chromosome: Chromosome,
    instance: ProblemInstance,
    filepath: str = "final_class_schedule.csv"
):
    """Exports scheduled sessions to a structured CSV table."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Session_ID", "Course", "Section", "Lecture_Num",
            "Day", "Timeslot", "Period", "Room", "Room_Size",
            "Section_Size", "Professor"
        ])
        for s_info, gene in zip(instance.sessions, chromosome.genes):
            period = "Morning" if gene.timeslot < 3 else "Afternoon"
            writer.writerow([
                f"s{s_info.session_id:03d}",
                f"c{s_info.course_id+1:02d}",
                f"sec{s_info.section_id+1:03d}",
                s_info.session_index + 1,
                DAY_NAMES[gene.day],
                gene.timeslot + 1,
                period,
                instance.rooms[gene.room_id].name,
                instance.rooms[gene.room_id].size,
                s_info.registration_size,
                instance.professors[gene.professor_id].name
            ])
    print(f"[✓] Schedule exported successfully to: {filepath}")


def plot_fitness_history(
    history: List[Dict[str, float]],
    save_path: str = "fitness_convergence_plot.png"
):
    """Generates and saves the fitness convergence curve required by the assignment."""
    if not HAS_MATPLOTLIB:
        print("[!] Matplotlib is not installed. Skipping plot generation.")
        return

    gens = [h["generation"] for h in history]
    best_fits = [h["best_fitness"] for h in history]
    avg_fits = [h["avg_fitness"] for h in history]
    hard_viols = [h["hard_violations"] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)

    color = "#1A6B72"
    ax1.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Fitness Score", color=color, fontsize=11, fontweight="bold")
    l1 = ax1.plot(gens, best_fits, color=color, linewidth=2, label="Best Fitness")
    l2 = ax1.plot(gens, avg_fits, color="#0E7490", linestyle="--", linewidth=1.5, label="Average Fitness")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Twin axis for hard constraint violations
    ax2 = ax1.twinx()
    color2 = "#DC2626"
    ax2.set_ylabel("Hard Constraint Violations", color=color2, fontsize=11, fontweight="bold")
    l3 = ax2.plot(gens, hard_viols, color=color2, linestyle="-.", linewidth=2, label="Hard Violations")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper left", framealpha=0.9)

    plt.title("Genetic Algorithm Convergence - Class Scheduling (HW3)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Convergence plot saved to: {save_path}")


# =============================================================================
# 6. MAIN EXECUTION ENTRY POINT
# =============================================================================

def main():
    print("=" * 70)
    print("   HOMEWORK 3: CLASS SCHEDULING USING GENETIC ALGORITHM")
    print("   CO3061 - Introduction to AI | HCMUT | Group 4")
    print("=" * 70)

    # Parse basic CLI arguments
    demo_mode = "--demo" in sys.argv or len(sys.argv) == 1
    generations = 30 if demo_mode else 150
    pop_size = 40 if demo_mode else 80

    print(f"[*] Configuration: Generations = {generations}, Population Size = {pop_size}")
    print("[*] Initializing Problem Instance (60 courses, 480 sections, 960 sessions, 35 rooms)...")

    instance = ProblemInstance.generate_random_instance(seed=42)
    print(f"[✓] Instance generated: {len(instance.sessions)} sessions, {len(instance.rooms)} rooms, {len(instance.professors)} professors.")

    print("\n[*] Initializing Genetic Algorithm engine...")
    ga = GeneticAlgorithm(
        instance=instance,
        pop_size=pop_size,
        crossover_rate=0.85,
        mutation_rate=0.06,
        elitism_count=3,
        tournament_size=5,
        seed=42
    )

    print("[*] Starting evolutionary optimization loop...\n")
    best_solution = ga.run(generations=generations, verbose=True)

    print("\n" + "=" * 70)
    print("   EVOLUTIONARY OPTIMIZATION COMPLETE")
    print("=" * 70)
    m = best_solution.metrics
    print(f"Final Best Fitness Score        : {best_solution.fitness:.5f}")
    print(f"Total Hard Constraint Violations: {m.total_hard_violations}")
    print(f"  - H1 Room/Section Size        : {m.h1_room_size_violations}")
    print(f"  - H2 Session Day Separation   : {m.h2_day_separation_violations}")
    print(f"  - H3 Prof Collision & Cap     : {m.h3_prof_conflict_violations + m.h3_prof_overload_violations}")
    print(f"  - H5 Section Room Overlap     : {m.h5_section_overlap_violations}")
    print(f"  - H6 Room Conflict            : {m.h6_room_conflict_violations}")
    print(f"Total Soft Constraint Penalty   : {m.total_soft_penalty:.2f}")
    print(f"  - S1 Prof Idle Gaps           : {m.s1_prof_scatter_penalty:.2f}")
    print(f"  - S2 Prof Half-Day Mixing     : {m.s2_prof_half_day_penalty:.2f}")
    print(f"  - S3 Course Half-Day Mixing   : {m.s3_course_half_day_penalty:.2f}")
    print(f"  - S4 Room Capacity Waste      : {m.s4_room_waste_penalty:.2f}")
    print(f"Bonus Satisfaction:")
    print(f"  - B1 Profs with Favorite Class: {m.b1_prof_favorite_satisfaction}/60")
    print(f"  - B2 Unqualified Course Count : {m.b2_unqualified_teach_count}")

    # Export outputs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "final_class_schedule.csv")
    plot_path = os.path.join(base_dir, "fitness_convergence_plot.png")
    fig_report_path = os.path.join(base_dir, "report", "figures", "fitness_convergence_plot.png")

    export_schedule_to_csv(best_solution, instance, filepath=csv_path)
    plot_fitness_history(ga.history, save_path=plot_path)
    if os.path.exists(os.path.dirname(fig_report_path)):
        try:
            import shutil
            shutil.copyfile(plot_path, fig_report_path)
        except Exception:
            pass
    print("\n[✓] All outputs generated successfully.")


if __name__ == "__main__":
    main()
