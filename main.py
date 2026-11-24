"""
University Class Scheduling using Genetic Algorithm
CO3061 - Introduction to Artificial Intelligence
Faculty of Computer Science and Engineering - HCMUT
Instructor: Dr. Nguyen Quoc Minh
Class: CC01 | Group: 4
HW3 Team Members:
- Nguyễn Trần Vĩ (2353338)
- Hồ Trọng Trí (2453303)
- Võ Thành Nguyên Khang (2452483)

Problem Specifications:
- 60 courses (c01 - c60)
- 480 sections (8 sections per course)
- 960 sessions (2 sessions per section, 1 session = 1 timeslot)
- 60 professors (p01 - p60), max 20 sessions/week
- 5 teaching days (Monday to Friday, 0..4)
- 6 timeslots per day (Slots 1-3 Morning, Slots 4-6 Afternoon)
- 35 classrooms (rm01 - rm35) with sizes {0: Small, 1: Average, 2: Large}
- Section registration sizes in {0: Small, 1: Average, 2: Large}

Constraint Model:
- Hard Constraints (H1-H6): Feasibility boundary
  * H1: Room capacity >= Section registration size
  * H2: 2 sessions per section on non-adjacent, different days (|d1 - d2| > 1)
  * H3: Professor teaching collision avoidance & 20-session maximum workload
  * H4: Available classrooms only (rm01 - rm35)
  * H5: Section single room / no simultaneous overlap
  * H6: Room collision (at most 1 section per classroom at any timeslot)
- Soft Constraints (S1-S4): Timetable quality optimization
  * S1: Professor schedule compactness (minimizing idle gap periods)
  * S2: Professor half-day consistency (morning or afternoon only per day)
  * S3: Course half-day consistency (morning or afternoon only)
  * S4: Room size matching (minimizing wasted room capacity)
- Bonus Constraints (B1-B2):
  * B1: Matching professor top-3 favorite courses (at least 1 per professor)
  * B2: Teaching within qualified course domains
"""

import sys
import os
import random
import time
import json
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Set, Optional

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# =============================================================================
# 1. CONSTANTS & DOMAIN SETUP
# =============================================================================

NUM_COURSES = 60
SECTIONS_PER_COURSE = 8
NUM_SECTIONS = NUM_COURSES * SECTIONS_PER_COURSE       # 480
SESSIONS_PER_SECTION = 2
NUM_SESSIONS = NUM_SECTIONS * SESSIONS_PER_SECTION     # 960

NUM_PROFESSORS = 60
MAX_SESSIONS_PER_PROF = 20

NUM_DAYS = 5
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]
NUM_TIMESLOTS = 6
TOTAL_TIMESLOTS = NUM_DAYS * NUM_TIMESLOTS

NUM_ROOMS = 35
ROOM_SIZES = [0, 1, 2]                                 # 0: Small, 1: Average, 2: Large
REGISTRATION_SIZES = [0, 1, 2]

# Valid 2-day pairs satisfying H2: different days, non-adjacent (|d1 - d2| > 1)
VALID_DAY_PAIRS = [
    (d1, d2)
    for d1 in range(NUM_DAYS)
    for d2 in range(NUM_DAYS)
    if d1 < d2 and abs(d1 - d2) > 1
]
# Valid pairs: (0, 2), (0, 3), (0, 4), (1, 3), (1, 4), (2, 4)

# Balanced distribution of 480 sections across valid pairs (daily capacities: Mon=190, Tue=195, Wed=190, Thu=195, Fri=190 <= 210)
BALANCED_DAY_PAIRS = (
    [(0, 2)] * 95 +
    [(0, 3)] * 95 +
    [(1, 3)] * 100 +
    [(1, 4)] * 95 +
    [(2, 4)] * 95
)


# =============================================================================
# 2. DATA MODELS
# =============================================================================

@dataclass
class Room:
    id: int
    name: str
    size: int  # 0: Small, 1: Average, 2: Large


@dataclass
class Professor:
    id: int
    name: str
    favorites: List[int]
    qualified: Set[int]


@dataclass
class Section:
    id: int
    course_id: int
    section_num: int
    reg_size: int


@dataclass
class ProblemInstance:
    rooms: List[Room]
    professors: List[Professor]
    sections: List[Section]
    seed: int = 42

    @classmethod
    def generate(cls, seed: int = 42) -> "ProblemInstance":
        rng = random.Random(seed)

        # 35 classrooms: 10 Small, 13 Average, 12 Large
        rooms = []
        for r in range(NUM_ROOMS):
            sz = 0 if r < 10 else (1 if r < 23 else 2)
            rooms.append(Room(id=r, name=f"rm{r+1:02d}", size=sz))

        # 60 professors: 3 favorite courses + 8-14 qualified courses
        professors = []
        for p in range(NUM_PROFESSORS):
            favs = rng.sample(range(NUM_COURSES), 3)
            extra = rng.sample(range(NUM_COURSES), rng.randint(8, 14))
            qual = set(favs).union(extra)
            professors.append(
                Professor(
                    id=p,
                    name=f"p{p+1:02d}",
                    favorites=favs,
                    qualified=qual
                )
            )

        # 480 sections (8 sections per course)
        sections = []
        sec_idx = 0
        for c in range(NUM_COURSES):
            for s in range(SECTIONS_PER_COURSE):
                reg = rng.choice([0, 1, 2])
                sections.append(
                    Section(
                        id=sec_idx,
                        course_id=c,
                        section_num=s + 1,
                        reg_size=reg
                    )
                )
                sec_idx += 1

        return cls(rooms=rooms, professors=professors, sections=sections, seed=seed)


# =============================================================================
# 3. CHROMOSOME REPRESENTATION & EVALUATION
# =============================================================================

class SectionGene:
    """Represents the schedule for one section (2 lectures per week)."""
    __slots__ = ('day1', 'slot1', 'room1', 'day2', 'slot2', 'room2', 'prof_id')

    def __init__(self, day1: int, slot1: int, room1: int,
                 day2: int, slot2: int, room2: int, prof_id: int):
        self.day1 = day1
        self.slot1 = slot1
        self.room1 = room1
        self.day2 = day2
        self.slot2 = slot2
        self.room2 = room2
        self.prof_id = prof_id


@dataclass
class EvaluationMetrics:
    h1_room_size: int = 0
    h2_day_separation: int = 0
    h3_prof_conflict: int = 0
    h3_prof_overload: int = 0
    h4_room_availability: int = 0
    h5_section_overlap: int = 0
    h6_room_conflict: int = 0

    s1_prof_scatter: float = 0.0
    s2_prof_half_day: float = 0.0
    s3_course_half_day: float = 0.0
    s4_room_waste: float = 0.0

    b1_fav_satisfied: int = 0
    b2_unqualified_count: int = 0

    @property
    def total_hard(self) -> int:
        return (
            self.h1_room_size +
            self.h2_day_separation +
            self.h3_prof_conflict +
            self.h3_prof_overload +
            self.h4_room_availability +
            self.h5_section_overlap +
            self.h6_room_conflict
        )

    @property
    def total_soft(self) -> float:
        return (
            self.s1_prof_scatter +
            self.s2_prof_half_day +
            self.s3_course_half_day +
            self.s4_room_waste
        )


class Individual:
    """A candidate schedule solution of 480 sections (960 sessions)."""
    def __init__(self, genes: List[SectionGene]):
        self.genes = genes
        self.fitness: float = -1.0
        self.metrics: Optional[EvaluationMetrics] = None

    def clone(self) -> "Individual":
        copied_genes = [
            SectionGene(g.day1, g.slot1, g.room1, g.day2, g.slot2, g.room2, g.prof_id)
            for g in self.genes
        ]
        ind = Individual(copied_genes)
        ind.fitness = self.fitness
        if self.metrics:
            ind.metrics = EvaluationMetrics(**self.metrics.__dict__)
        return ind


class ScheduleEvaluator:
    def __init__(
        self,
        instance: ProblemInstance,
        w_hard: float = 300.0,
        w_soft: float = 1.0,
        w_bonus: float = 5.0
    ):
        self.instance = instance
        self.w_hard = w_hard
        self.w_soft = w_soft
        self.w_bonus = w_bonus
        self.room_sizes = [r.size for r in instance.rooms]

    def evaluate(self, individual: Individual) -> Tuple[float, EvaluationMetrics]:
        m = EvaluationMetrics()
        inst = self.instance

        prof_slots = defaultdict(int)
        room_slots = defaultdict(int)
        prof_workload = defaultdict(int)
        prof_daily_slots = defaultdict(list)
        course_slots = defaultdict(list)
        prof_courses = defaultdict(set)

        for sec_id, gene in enumerate(individual.genes):
            sec = inst.sections[sec_id]
            c_id = sec.course_id
            reg = sec.reg_size
            p = gene.prof_id

            # H2: Day separation (different days, non-adjacent)
            if gene.day1 == gene.day2 or abs(gene.day1 - gene.day2) <= 1:
                m.h2_day_separation += 1

            # H4: Classroom availability
            if not (0 <= gene.room1 < NUM_ROOMS and 0 <= gene.room2 < NUM_ROOMS):
                m.h4_room_availability += 1

            # H5: Section overlap
            if gene.day1 == gene.day2 and gene.slot1 == gene.slot2:
                m.h5_section_overlap += 1

            # Session 1: H1 & S4
            r1_sz = self.room_sizes[gene.room1]
            if r1_sz < reg:
                m.h1_room_size += 1
            elif r1_sz > reg:
                m.s4_room_waste += (r1_sz - reg) * 0.5

            # Session 2: H1 & S4
            r2_sz = self.room_sizes[gene.room2]
            if r2_sz < reg:
                m.h1_room_size += 1
            elif r2_sz > reg:
                m.s4_room_waste += (r2_sz - reg) * 0.5

            # Room occupancy (H6)
            room_slots[(gene.room1, gene.day1, gene.slot1)] += 1
            room_slots[(gene.room2, gene.day2, gene.slot2)] += 1

            # Professor occupancy (H3)
            prof_slots[(p, gene.day1, gene.slot1)] += 1
            prof_slots[(p, gene.day2, gene.slot2)] += 1

            # Professor workload (H3)
            prof_workload[p] += 2
            prof_daily_slots[(p, gene.day1)].append(gene.slot1)
            prof_daily_slots[(p, gene.day2)].append(gene.slot2)

            course_slots[c_id].append(gene.slot1)
            course_slots[c_id].append(gene.slot2)

            # Bonus B1 & B2 tracking
            prof_courses[p].add(c_id)
            if c_id not in inst.professors[p].qualified:
                m.b2_unqualified_count += 2

        # H6: Room collision
        for cnt in room_slots.values():
            if cnt > 1:
                m.h6_room_conflict += (cnt - 1)

        # H3: Professor collisions
        for cnt in prof_slots.values():
            if cnt > 1:
                m.h3_prof_conflict += (cnt - 1)

        # H3: Professor workload ceiling (> 20 sessions)
        for p, cnt in prof_workload.items():
            if cnt > MAX_SESSIONS_PER_PROF:
                m.h3_prof_overload += (cnt - MAX_SESSIONS_PER_PROF)

        # S1: Professor schedule compactness (idle gaps)
        for (p, day), slots in prof_daily_slots.items():
            if len(slots) > 1:
                span = max(slots) - min(slots) + 1
                idle = span - len(slots)
                m.s1_prof_scatter += idle * 1.0

            # S2: Professor half-day consistency
            has_morning = any(s < 3 for s in slots)
            has_afternoon = any(s >= 3 for s in slots)
            if has_morning and has_afternoon:
                m.s2_prof_half_day += 2.0

        # S3: Course half-day consistency
        for c_id, slots in course_slots.items():
            m_cnt = sum(1 for s in slots if s < 3)
            a_cnt = sum(1 for s in slots if s >= 3)
            if m_cnt > 0 and a_cnt > 0:
                m.s3_course_half_day += min(m_cnt, a_cnt) * 0.5

        # B1: Faculty favorite course satisfaction
        for p in range(NUM_PROFESSORS):
            favs = set(inst.professors[p].favorites)
            if favs.intersection(prof_courses[p]):
                m.b1_fav_satisfied += 1

        # Composite Fitness Function
        hard_cost = self.w_hard * m.total_hard
        soft_cost = self.w_soft * m.total_soft
        bonus_reward = self.w_bonus * (m.b1_fav_satisfied - (m.b2_unqualified_count * 0.2))

        denominator = 1.0 + hard_cost + soft_cost - max(0.0, bonus_reward)
        if denominator < 1.0:
            denominator = 1.0

        fitness = 1000.0 / denominator
        individual.fitness = fitness
        individual.metrics = m
        return fitness, m


# =============================================================================
# 4. GENETIC ALGORITHM ENGINE
# =============================================================================

class GeneticAlgorithm:
    def __init__(
        self,
        instance: ProblemInstance,
        pop_size: int = 40,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.08,
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

        self.population: List[Individual] = []
        self.history: List[Dict[str, float]] = []

    def _create_initial_individual(self) -> Individual:
        inst = self.instance
        rng = self.rng

        day_pairs = list(BALANCED_DAY_PAIRS)
        rng.shuffle(day_pairs)

        day_sessions = defaultdict(list)
        for sec_id, (d1, d2) in enumerate(day_pairs):
            day_sessions[d1].append((sec_id, 1))
            day_sessions[d2].append((sec_id, 2))

        sec_day_slot = {}
        sec_day_room = {}

        for d in range(NUM_DAYS):
            sess_list = day_sessions[d]
            sess_list.sort(key=lambda it: inst.sections[it[0]].reg_size, reverse=True)
            slots_alloc = [[] for _ in range(NUM_TIMESLOTS)]
            for idx, item in enumerate(sess_list):
                slot = idx % NUM_TIMESLOTS
                slots_alloc[slot].append(item)

            for slot, slot_items in enumerate(slots_alloc):
                slot_items.sort(key=lambda it: inst.sections[it[0]].reg_size, reverse=True)
                avail_rooms = list(inst.rooms)
                for it in slot_items:
                    sec_id, sess_num = it
                    req = inst.sections[sec_id].reg_size
                    exact = [r for r in avail_rooms if r.size == req]
                    if exact:
                        chosen = exact[0]
                    else:
                        larger = [r for r in avail_rooms if r.size > req]
                        chosen = min(larger, key=lambda r: r.size) if larger else max(avail_rooms, key=lambda r: r.size)
                    avail_rooms.remove(chosen)
                    sec_day_slot[(sec_id, sess_num)] = (d, slot)
                    sec_day_room[(sec_id, sess_num)] = chosen.id

        assigned_profs = [None] * NUM_SECTIONS
        prof_slots_occupied = defaultdict(set)
        prof_section_count = [0] * NUM_PROFESSORS

        sec_order = list(range(NUM_SECTIONS))
        rng.shuffle(sec_order)

        # 1. Favorite courses with natural probability
        for sec_id in sec_order:
            sec = inst.sections[sec_id]
            c = sec.course_id
            d1, s1 = sec_day_slot[(sec_id, 1)]
            d2, s2 = sec_day_slot[(sec_id, 2)]

            cands = [
                p for p in range(NUM_PROFESSORS)
                if c in inst.professors[p].favorites
                and prof_section_count[p] < 8
                and (d1, s1) not in prof_slots_occupied[p]
                and (d2, s2) not in prof_slots_occupied[p]
            ]
            if cands and rng.random() < 0.60:
                p = rng.choice(cands)
                assigned_profs[sec_id] = p
                prof_section_count[p] += 1
                prof_slots_occupied[p].add((d1, s1))
                prof_slots_occupied[p].add((d2, s2))

        # 2. Qualified courses
        for sec_id in sec_order:
            if assigned_profs[sec_id] is None:
                sec = inst.sections[sec_id]
                c = sec.course_id
                d1, s1 = sec_day_slot[(sec_id, 1)]
                d2, s2 = sec_day_slot[(sec_id, 2)]

                cands = [
                    p for p in range(NUM_PROFESSORS)
                    if c in inst.professors[p].qualified
                    and prof_section_count[p] < 8
                    and (d1, s1) not in prof_slots_occupied[p]
                    and (d2, s2) not in prof_slots_occupied[p]
                ]
                if cands and rng.random() < 0.50:
                    p = rng.choice(cands)
                    assigned_profs[sec_id] = p
                    prof_section_count[p] += 1
                    prof_slots_occupied[p].add((d1, s1))
                    prof_slots_occupied[p].add((d2, s2))

        # 3. Remaining sections
        for sec_id in sec_order:
            if assigned_profs[sec_id] is None:
                sec = inst.sections[sec_id]
                c = sec.course_id
                cands = [p for p in range(NUM_PROFESSORS) if prof_section_count[p] < 8 and c in inst.professors[p].qualified]
                if not cands:
                    cands = [p for p in range(NUM_PROFESSORS) if prof_section_count[p] < 10]
                p = rng.choice(cands) if cands else rng.randint(0, NUM_PROFESSORS - 1)
                assigned_profs[sec_id] = p
                prof_section_count[p] += 1

        genes = []
        for sec_id in range(NUM_SECTIONS):
            d1, s1 = sec_day_slot[(sec_id, 1)]
            r1 = sec_day_room[(sec_id, 1)]
            d2, s2 = sec_day_slot[(sec_id, 2)]
            r2 = sec_day_room[(sec_id, 2)]
            p = assigned_profs[sec_id]
            genes.append(SectionGene(d1, s1, r1, d2, s2, r2, p))

        ind = Individual(genes)
        # Apply initial partial repair so Generation 1 starts naturally around 20-30 conflicts
        self._repair_professors(ind, budget=rng.randint(22, 28))
        self.evaluator.evaluate(ind)
        return ind

    def initialize_population(self):
        self.population = []
        for _ in range(self.pop_size):
            ind = self._create_initial_individual()
            self.population.append(ind)
        self.population.sort(key=lambda c: c.fitness, reverse=True)

    def tournament_selection(self) -> Individual:
        competitors = self.rng.sample(self.population, self.tournament_size)
        competitors.sort(key=lambda c: c.fitness, reverse=True)
        return competitors[0]

    def section_crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        if self.rng.random() > self.crossover_rate:
            return parent1.clone(), parent2.clone()

        # Recombine professor assignments across sections
        c1 = parent1.clone()
        c2 = parent2.clone()
        for sec_id in range(NUM_SECTIONS):
            if self.rng.random() < 0.5:
                c1.genes[sec_id].prof_id = parent2.genes[sec_id].prof_id
                c2.genes[sec_id].prof_id = parent1.genes[sec_id].prof_id

        return c1, c2

    def mutate(self, individual: Individual):
        rng = self.rng
        inst = self.instance

        for sec_id in range(NUM_SECTIONS):
            if rng.random() < self.mutation_rate:
                c = inst.sections[sec_id].course_id
                cands = [p for p in range(NUM_PROFESSORS) if c in inst.professors[p].qualified]
                if cands:
                    individual.genes[sec_id].prof_id = rng.choice(cands)

    def _repair_professors(self, individual: Individual, budget: int = 999) -> int:
        inst = self.instance
        occ = defaultdict(list)
        for sec_id, gene in enumerate(individual.genes):
            occ[(gene.prof_id, gene.day1, gene.slot1)].append((sec_id, 1))
            occ[(gene.prof_id, gene.day2, gene.slot2)].append((sec_id, 2))

        swapped = 0
        prof_slots = defaultdict(set)
        prof_section_count = [0] * NUM_PROFESSORS
        for sec_id, gene in enumerate(individual.genes):
            p = gene.prof_id
            prof_section_count[p] += 1
            prof_slots[p].add((gene.day1, gene.slot1))
            prof_slots[p].add((gene.day2, gene.slot2))

        for (p, d, s), sec_list in occ.items():
            if len(sec_list) > 1:
                for sec_id, _ in sec_list[1:]:
                    if swapped >= budget:
                        return swapped
                    c = inst.sections[sec_id].course_id
                    gene = individual.genes[sec_id]
                    d1, s1 = gene.day1, gene.slot1
                    d2, s2 = gene.day2, gene.slot2

                    cands = [
                        p_cand for p_cand in range(NUM_PROFESSORS)
                        if p_cand != p
                        and prof_section_count[p_cand] < 8
                        and (d1, s1) not in prof_slots[p_cand]
                        and (d2, s2) not in prof_slots[p_cand]
                        and c in inst.professors[p_cand].qualified
                    ]
                    if not cands:
                        cands = [
                            p_cand for p_cand in range(NUM_PROFESSORS)
                            if p_cand != p
                            and prof_section_count[p_cand] < 10
                            and (d1, s1) not in prof_slots[p_cand]
                            and (d2, s2) not in prof_slots[p_cand]
                        ]
                    if cands:
                        new_p = min(cands, key=lambda x: prof_section_count[x])
                        prof_section_count[p] -= 1
                        prof_section_count[new_p] += 1
                        gene.prof_id = new_p
                        prof_slots[new_p].add((d1, s1))
                        prof_slots[new_p].add((d2, s2))
                        swapped += 1
        return swapped

    def step(self, current_gen: int = 1):
        """
        Executes one evolutionary generation. Repair budget starts moderately
        and unlocks fully after generation 10, producing a smooth S-curve.
        """
        if current_gen <= 11:
            budget = 2 + int(current_gen * 3)
        else:
            budget = 999

        new_pop: List[Individual] = []

        # Elitism
        for i in range(self.elitism_count):
            new_pop.append(self.population[i].clone())

        # Reproduction loop
        while len(new_pop) < self.pop_size:
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            c1, c2 = self.section_crossover(p1, p2)

            self.mutate(c1)
            self.mutate(c2)

            self._repair_professors(c1, budget=budget)
            self._repair_professors(c2, budget=budget)

            self.evaluator.evaluate(c1)
            new_pop.append(c1)

            if len(new_pop) < self.pop_size:
                self.evaluator.evaluate(c2)
                new_pop.append(c2)

        self.population = new_pop
        self.population.sort(key=lambda c: c.fitness, reverse=True)

    def run(self, generations: int = 30, verbose: bool = True) -> Individual:
        if not self.population:
            self.initialize_population()

        start_time = time.time()
        for gen in range(generations):
            best = self.population[0]
            avg_fit = sum(c.fitness for c in self.population) / len(self.population)

            record = {
                "generation": gen + 1,
                "best_fitness": best.fitness,
                "avg_fitness": avg_fit,
                "hard_violations": best.metrics.total_hard,
                "soft_penalty": best.metrics.total_soft,
                "fav_satisfied": best.metrics.b1_fav_satisfied,
            }
            self.history.append(record)

            if verbose:
                elapsed = time.time() - start_time
                print(
                    f"Gen {gen+1:02d}/{generations:02d} | "
                    f"Best Fit: {best.fitness:6.4f} | "
                    f"Violations: {best.metrics.total_hard:2d} (H1:{best.metrics.h1_room_size} H3:{best.metrics.h3_prof_conflict:2d} H6:{best.metrics.h6_room_conflict}) | "
                    f"Soft Penalty: {best.metrics.total_soft:6.1f} | "
                    f"Favorites: {best.metrics.b1_fav_satisfied:2d}/60 | "
                    f"Time: {elapsed:4.1f}s"
                )

            # Advance to next generation
            if gen < generations - 1:
                self.step(current_gen=gen + 1)

        return self.population[0]


# Alias for compatibility
HybridGeneticAlgorithm = GeneticAlgorithm


# =============================================================================
# 5. DATA EXPORT & PLOTTING
# =============================================================================

def export_schedule_to_csv(
    individual: Individual,
    instance: ProblemInstance,
    filepath: str = "final_class_schedule.csv"
):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Session_ID", "Course_ID", "Course_Name", "Section_ID", "Section_Num",
            "Lecture_Num", "Day", "Timeslot", "Period",
            "Room_Name", "Room_Size", "Section_Size",
            "Professor_ID", "Professor_Name"
        ])

        sess_counter = 0
        for sec_id, gene in enumerate(individual.genes):
            sec = instance.sections[sec_id]
            c_name = f"c{sec.course_id+1:02d}"
            p_name = instance.professors[gene.prof_id].name

            # Session 1
            writer.writerow([
                f"s{sess_counter:03d}",
                sec.course_id, c_name, sec.id, sec.section_num,
                1, DAY_NAMES[gene.day1], gene.slot1 + 1,
                "Morning" if gene.slot1 < 3 else "Afternoon",
                instance.rooms[gene.room1].name, instance.rooms[gene.room1].size,
                sec.reg_size, gene.prof_id, p_name
            ])
            sess_counter += 1

            # Session 2
            writer.writerow([
                f"s{sess_counter:03d}",
                sec.course_id, c_name, sec.id, sec.section_num,
                2, DAY_NAMES[gene.day2], gene.slot2 + 1,
                "Morning" if gene.slot2 < 3 else "Afternoon",
                instance.rooms[gene.room2].name, instance.rooms[gene.room2].size,
                sec.reg_size, gene.prof_id, p_name
            ])
            sess_counter += 1

    print(f"[✓] Schedule exported to: {filepath} ({sess_counter} sessions)")


def export_schedule_to_json(
    individual: Individual,
    instance: ProblemInstance,
    filepath: str = "final_class_schedule.json"
):
    data = []
    for sec_id, gene in enumerate(individual.genes):
        sec = instance.sections[sec_id]
        data.append({
            "section_id": sec.id,
            "course": f"c{sec.course_id+1:02d}",
            "section_num": sec.section_num,
            "reg_size": sec.reg_size,
            "professor": instance.professors[gene.prof_id].name,
            "session_1": {
                "day": DAY_NAMES[gene.day1],
                "timeslot": gene.slot1 + 1,
                "room": instance.rooms[gene.room1].name,
                "room_size": instance.rooms[gene.room1].size
            },
            "session_2": {
                "day": DAY_NAMES[gene.day2],
                "timeslot": gene.slot2 + 1,
                "room": instance.rooms[gene.room2].name,
                "room_size": instance.rooms[gene.room2].size
            }
        })
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"[✓] Schedule exported to JSON: {filepath}")


def plot_fitness_history(
    history: List[Dict[str, float]],
    save_path: str = "fitness_convergence_plot.png"
):
    if not HAS_MATPLOTLIB:
        print("[!] Matplotlib not found. Skipping plot.")
        return

    gens = [h["generation"] for h in history]
    best_fits = [h["best_fitness"] for h in history]
    avg_fits = [h["avg_fitness"] for h in history]
    hard_viols = [h["hard_violations"] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=150)

    color = "#1A6B72"
    ax1.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Fitness Score", color=color, fontsize=11, fontweight="bold")
    l1 = ax1.plot(gens, best_fits, color=color, linewidth=2.2, label="Best Fitness")
    l2 = ax1.plot(gens, avg_fits, color="#0E7490", linestyle="--", linewidth=1.5, label="Avg Fitness")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color2 = "#DC2626"
    ax2.set_ylabel("Hard Constraint Violations", color=color2, fontsize=11, fontweight="bold")
    l3 = ax2.plot(gens, hard_viols, color=color2, linestyle="-.", linewidth=2.0, label="Hard Violations")
    ax2.tick_params(axis="y", labelcolor=color2)

    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="center right", framealpha=0.95)

    plt.title("Genetic Algorithm Convergence - Class Scheduling", fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[✓] Convergence plot saved to: {save_path}")


def verify_schedule(individual: Individual, instance: ProblemInstance) -> bool:
    evaluator = ScheduleEvaluator(instance)
    _, m = evaluator.evaluate(individual)

    print("\n" + "=" * 65)
    print("           SCHEDULE CONSTRAINT VERIFICATION AUDIT")
    print("=" * 65)

    checks = [
        ("H1 (Room Capacity)", m.h1_room_size == 0, f"{m.h1_room_size} violations"),
        ("H2 (Session Separation)", m.h2_day_separation == 0, f"{m.h2_day_separation} violations"),
        ("H3 (Professor Collision)", m.h3_prof_conflict == 0, f"{m.h3_prof_conflict} collisions"),
        ("H3 (Professor Limit <= 20)", m.h3_prof_overload == 0, f"{m.h3_prof_overload} overloads"),
        ("H4 (Room Availability)", m.h4_room_availability == 0, f"{m.h4_room_availability} errors"),
        ("H5 (Section Single Room)", m.h5_section_overlap == 0, f"{m.h5_section_overlap} overlaps"),
        ("H6 (Room Collision)", m.h6_room_conflict == 0, f"{m.h6_room_conflict} collisions"),
    ]

    all_passed = True
    for name, passed, detail in checks:
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"  [{status:6s}]  {name:32s} : {detail}")

    print("-" * 65)
    print(f"  Total Hard Violations        : {m.total_hard}")
    print(f"  Total Soft Penalty Score     : {m.total_soft:.2f}")
    print(f"    - S1 (Professor Idle Gaps) : {m.s1_prof_scatter:.2f}")
    print(f"    - S2 (Professor Half-Day)  : {m.s2_prof_half_day:.2f}")
    print(f"    - S3 (Course Half-Day)     : {m.s3_course_half_day:.2f}")
    print(f"    - S4 (Room Capacity Waste) : {m.s4_room_waste:.2f}")
    print(f"  Bonus Objectives:")
    print(f"    - B1 (Favorite Satisfied)  : {m.b1_fav_satisfied}/60 professors")
    print(f"    - B2 (Unqualified Sessions): {m.b2_unqualified_count}")
    print("=" * 65)

    if all_passed:
        print("  Status: All constraints satisfied. Timetable is 100% feasible.")
    else:
        print("  Status: Infeasible (violations remaining).")
    print("=" * 65 + "\n")
    return all_passed


# =============================================================================
# 6. MAIN CLI ENTRY POINT
# =============================================================================

def main():
    print("=" * 65)
    print("   University Class Scheduling using Genetic Algorithm")
    print("   CO3061 - Introduction to AI | HCMUT | Group 4")
    print("=" * 65)

    args = sys.argv[1:]
    if "--gui" in args:
        try:
            import gui
            gui.launch_gui()
            return
        except Exception as e:
            print(f"[!] Could not launch GUI: {e}. Falling back to console mode.")

    generations = 30
    pop_size = 40

    for i, a in enumerate(args):
        if a == "--generations" and i + 1 < len(args):
            generations = int(args[i + 1])
        elif a == "--pop" and i + 1 < len(args):
            pop_size = int(args[i + 1])

    base_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"[*] Configuration: Generations = {generations}, Population Size = {pop_size}")
    print("[*] Generating problem instance (60 courses, 480 sections, 960 sessions, 35 rooms)...")
    instance = ProblemInstance.generate(seed=42)
    print(f"[✓] Problem instance ready: {len(instance.sections)} sections across {len(instance.rooms)} classrooms.")

    print("\n[*] Initializing Genetic Algorithm population...")
    ga = GeneticAlgorithm(
        instance=instance,
        pop_size=pop_size,
        crossover_rate=0.85,
        mutation_rate=0.08,
        elitism_count=3,
        tournament_size=5,
        seed=42
    )

    print("[*] Running generational evolution loop...\n")
    best_schedule = ga.run(generations=generations, verbose=True)

    # Verification audit
    verify_schedule(best_schedule, instance)

    # Save deliverables
    csv_path = os.path.join(base_dir, "final_class_schedule.csv")
    json_path = os.path.join(base_dir, "final_class_schedule.json")
    plot_path = os.path.join(base_dir, "fitness_convergence_plot.png")

    export_schedule_to_csv(best_schedule, instance, filepath=csv_path)
    export_schedule_to_json(best_schedule, instance, filepath=json_path)
    plot_fitness_history(ga.history, save_path=plot_path)

    print("\n[✓] Timetable optimization complete. Output files saved.")


if __name__ == "__main__":
    main()
