import collections

from ortools.sat.python import cp_model

from datetime import datetime

class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    """Print intermediate solutions."""

    def __init__(self):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.__solution_count = 0

    def on_solution_callback(self):
        """Called at each new solution."""
        #print('Solution %i, time = %f s, objective = %i' %
              #(self.__solution_count, self.WallTime(), self.ObjectiveValue()))
        self.__solution_count += 1


def solve_fjsp(jobs, operations, seconds = 15):
    jobs_or = []
    list_machines = []
    for job in jobs:
        job_info = []
        for o_id in job:
            ops_info = []
            for i in range(len(operations[o_id])):
                o = operations[o_id][i]
                if o != 0:
                    ops_info.append((int(o), i))
                    list_machines.append(i)
            job_info.append(ops_info)
        if job_info != []:
            jobs_or.append(job_info)

    return flexible_jobshop(jobs_or, len(operations[o_id]), seconds, [0]*len(operations[o_id]), [0]*len(jobs))


def flexible_jobshop(jobs, num_machines, seconds, machines_starts, job_starts):
    """Solve a small flexible jobshop problem."""
    # Data part.
    jobs = jobs

    num_jobs = len(jobs)
    all_jobs = range(num_jobs)

    num_machines = num_machines
    all_machines = range(num_machines)

    # Model the flexible jobshop problem.
    model = cp_model.CpModel()
    

    horizon = max(machines_starts)
    for job in jobs:
        for task in job:
            max_task_duration = 0
            for alternative in task:
                max_task_duration = max(max_task_duration, alternative[0])
            horizon += max_task_duration


    intervals_per_resources = collections.defaultdict(list)
    starts = {}  # indexed by (job_id, task_id).
    presences = {}  # indexed by (job_id, task_id, alt_id).
    job_ends = []

    var_machines = []
    for machine_id in all_machines:
        start_m = model.NewIntVar(machines_starts[machine_id], horizon, 'start_m' + str(machine_id))
        model.Add(start_m >= machines_starts[machine_id])
        var_machines.append(start_m)

    # Scan the jobs and create the relevant variables and intervals.
    for job_id in all_jobs:
        job = jobs[job_id]
        num_tasks = len(job)
        previous_end = None
        for task_id in range(num_tasks):
            task = job[task_id]

            min_duration = task[0][0]
            max_duration = task[0][0]

            num_alternatives = len(task)
            all_alternatives = range(num_alternatives)

            for alt_id in range(1, num_alternatives):
                alt_duration = task[alt_id][0]
                min_duration = min(min_duration, alt_duration)
                max_duration = max(max_duration, alt_duration)

            # Create main interval for the task.
            suffix_name = '_j%i_t%i' % (job_id, task_id)
            start = model.NewIntVar(0, horizon, 'start' + suffix_name)


            duration = model.NewIntVar(min_duration, max_duration,
                                       'duration' + suffix_name)
            end = model.NewIntVar(0, horizon, 'end' + suffix_name)
            interval = model.NewIntervalVar(start, duration, end,
                                            'interval' + suffix_name)

            #print((job_id, task_id))
            # Store the start for the solution.
            starts[(job_id, task_id)] = start

            model.Add(start >= job_starts[job_id])

            if num_alternatives==1:
                model.Add(start >= machines_starts[task[0][1]])

            # Add precedence with previous task in the same job.
            if previous_end is not None:
                model.Add(start >= previous_end)
                
            previous_end = end

            # Create alternative intervals.
            if num_alternatives > 1:
                l_presences = []
                for alt_id in all_alternatives:
                    alt_suffix = '_j%i_t%i_a%i' % (job_id, task_id, alt_id)
                    l_presence = model.NewBoolVar('presence' + alt_suffix)

                    if task_id == 0:
                        l_start = model.NewIntVar(machines_starts[task[alt_id][1]], horizon, 'start' + alt_suffix)
                    else:
                        l_start = model.NewIntVar(machines_starts[task[alt_id][1]], horizon, 'start' + alt_suffix)

                    model.Add(l_start >= machines_starts[task[alt_id][1]])

                    l_duration = task[alt_id][0]
                    l_end = model.NewIntVar(machines_starts[task[alt_id][1]] + l_duration, horizon, 'end' + alt_suffix)
                    l_interval = model.NewOptionalIntervalVar(
                        l_start, l_duration, l_end, l_presence,
                        'interval' + alt_suffix)
                    l_presences.append(l_presence)
                    
                    # Link the master variables with the local ones.
                    model.Add(start == l_start).OnlyEnforceIf(l_presence)
                    model.Add(duration == l_duration).OnlyEnforceIf(l_presence)
                    model.Add(end == l_end).OnlyEnforceIf(l_presence)

                    model.Add(start >= machines_starts[task[alt_id][1]]).OnlyEnforceIf(l_presence)                    

                    # Add the local interval to the right machine.
                    intervals_per_resources[task[alt_id][1]].append(l_interval)

                    # Store the presences for the solution.
                    presences[(job_id, task_id, alt_id)] = l_presence

                # Select exactly one presence variable.
                model.AddExactlyOne(l_presences)
            else:
                intervals_per_resources[task[0][1]].append(interval)
                presences[(job_id, task_id, 0)] = model.NewConstant(1)

        job_ends.append(previous_end)

    # Create machines constraints.
    for machine_id in all_machines:
        intervals = intervals_per_resources[machine_id]
        if len(intervals) > 1:
            model.AddNoOverlap(intervals)

    # Makespan objective
    makespan = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(makespan, job_ends)
    model.Minimize(makespan)

    # Solve model.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds

    solution_printer = SolutionPrinter()
    status = solver.Solve(model, solution_printer)


    list_dicts = []
    for job_id in all_jobs:
        for task_id in range(len(jobs[job_id])):
            start_value = solver.Value(starts[(job_id, task_id)])

            machine = -1
            duration = -1
            selected = -1
            for alt_id in range(len(jobs[job_id][task_id])):
                if solver.Value(presences[(job_id, task_id, alt_id)]):
                    duration = jobs[job_id][task_id][alt_id][0]
                    machine = jobs[job_id][task_id][alt_id][1]
                    selected = alt_id


            list_dicts.append({"job_id":str(job_id), "task_id": task_id, "start": start_value, "end": start_value+duration, "machine": str(machine)})  

    return solver.StatusName(status), solver.ObjectiveValue(), list_dicts