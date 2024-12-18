import random
from pathlib import Path

cwd = Path().cwd()

PATHS = {"data": cwd / "data"}


class CaseGenerator:
    """
    FJSP instance generator
    """

    def __init__(
        self,
        job_init,
        num_mas,
        opes_per_job_min,
        opes_per_job_max,
        nums_ope=None,
        meq = 3,
        min_processing = 1,
        max_processing = 5,
        desv_proc = 0.2,
        path=PATHS["data"],
        flag_same_opes=True,
        flag_doc=False,
    ):
        if nums_ope is None:
            nums_ope = []

        self.flag_doc = flag_doc
        self.flag_same_opes = flag_same_opes
        self.nums_ope = nums_ope
        self.path = path
        self.job_init = job_init
        self.num_mas = num_mas

        self.mas_per_ope_min = 1
        self.mas_per_ope_max = meq

        # self.mas_per_ope_min = int(num_mas/4)
        # self.mas_per_ope_max = int(num_mas/1.5)

        self.opes_per_job_min = opes_per_job_min
        self.opes_per_job_max = opes_per_job_max
        self.proctime_per_ope_min = min_processing
        self.proctime_per_ope_max = max_processing
        self.proctime_dev = desv_proc

    def get_case(self, idx=0):
        """
        Generate FJSP instance
        :param idx: The instance number
        """
        self.num_jobs = self.job_init
        if not self.flag_same_opes:
            self.nums_ope = [
                random.randint(self.opes_per_job_min, self.opes_per_job_max)
                for _ in range(self.num_jobs)
            ]
        self.num_opes = sum(self.nums_ope)
        self.nums_option = [
            random.randint(self.mas_per_ope_min, self.mas_per_ope_max)
            for _ in range(self.num_opes)
        ]
        self.num_options = sum(self.nums_option)
        self.ope_ma = []
        for val in self.nums_option:
            self.ope_ma = self.ope_ma + sorted(random.sample(range(1, self.num_mas + 1), val))
        self.proc_time = []
        self.proc_times_mean = [
            random.randint(self.proctime_per_ope_min, self.proctime_per_ope_max)
            for _ in range(self.num_opes)
        ]
        for i in range(len(self.nums_option)):
            low_bound = max(
                self.proctime_per_ope_min, round(self.proc_times_mean[i] * (1 - self.proctime_dev))
            )
            high_bound = min(
                self.proctime_per_ope_max, round(self.proc_times_mean[i] * (1 + self.proctime_dev))
            )
            proc_time_ope = [
                random.randint(10, 30) for _ in range(self.nums_option[i])
                #random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])
            ]
            self.proc_time = self.proc_time + proc_time_ope
        self.num_ope_biass = [sum(self.nums_ope[0:i]) for i in range(self.num_jobs)]
        self.num_ma_biass = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]
        line0 = f"{self.num_jobs}\t{self.num_mas}\t{self.num_options / self.num_opes}\n"
        lines = []
        lines_doc = []
        lines.append(line0)
        lines_doc.append(f"{self.num_jobs}\t{self.num_mas}\t{self.num_options / self.num_opes}")
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1
            idx_ma = 0
            line = []
            option_max = sum(
                self.nums_option[
                    self.num_ope_biass[i] : (self.num_ope_biass[i] + self.nums_ope[i])
                ]
            )
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_ope[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_ma = 0
                    flag_new_ope += self.nums_option[self.num_ope_biass[i] + idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_biass[i] + idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(
                        self.ope_ma[self.num_ma_biass[self.num_ope_biass[i] + idx_ope] + idx_ma]
                    )
                    flag += 1
                    flag_time = 1
                else:
                    line.append(
                        self.proc_time[self.num_ma_biass[self.num_ope_biass[i] + idx_ope] + idx_ma]
                    )
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_ma += 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + "\n")
                    lines_doc.append(str_line)
                    break
        lines.append("\n")
        if self.flag_doc:
            doc = open(
                self.path + f"{self.num_jobs}j_{self.num_mas}m_{str.zfill(str(idx + 1), 3)}.fjs",
                "a",
            )
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=doc)
            doc.close()
        return lines, self.num_jobs, self.num_jobs


def generate_instance_list(
    n_cases: int = 1,
    range_jobs: tuple[int, int] = (2, 3),
    range_machines: tuple[int, int] = (2, 3),
    range_op_per_job: tuple[int, int] = (2, 3),
    meq: int = 3,
    max_processing: int = 5,
    desv_proc: float =0.2
) -> list[str]:

    list_instances = []
    for _ in range(n_cases):
        max_machines = random.randint(10, 11)
        meq = random.randint(2, max_machines)
        max_opers = random.randint(max_machines, max_machines+1)
        desv_opers = random.randint(0, 2)
        min_processing = random.randint(2, 3)
        max_processing = random.randint(9, 10)
        desv_proc = 0.2 #random.uniform(0.05,0.2)

        train_config = {
                "n_cases": n_cases,
                "range_jobs": (19, 20),
                "range_machines": (max_machines, max_machines+1),
                "range_op_per_job": (max_opers - desv_opers, max_opers),
                "meq": meq,
                "min_processing": min_processing,
                "max_processing": max_processing,
                "desv_proc": desv_proc
        }

        range_jobs = (19, 20)
        range_machines =  (max_machines, max_machines+1)
        range_op_per_job = (max_opers - desv_opers, max_opers)

        n_jobs = random.randint(*range_jobs)
        n_machines = random.randint(*range_machines)

        n_operations = [random.randint(*range_op_per_job) for _ in range(n_jobs)]

        case_generator = CaseGenerator(
            n_jobs,
            n_machines,
            *range_op_per_job,
            nums_ope=n_operations,
            meq = meq,
            min_processing=min_processing,
            max_processing = max_processing,
            desv_proc = desv_proc
        )

        instance = case_generator.get_case()[0]
        instance = "".join(instance)
        list_instances.append(instance)

    return list_instances
