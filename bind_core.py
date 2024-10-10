import subprocess
import argparse
import os
import time
import logging
from datetime import datetime
from datetime import timezone


class PathManager:
    DATA_FILE_AUTHORITY = 0o640

    @classmethod
    def create_file_safety(cls, path: str):
        base_name = os.path.basename(path)
        msg = f"Failed to create file: {base_name}"
        if os.path.islink(path):
            raise RuntimeError(msg)
        if os.path.exists(path):
            return
        try:
            os.close(os.open(path, os.O_WRONLY | os.O_CREAT, cls.DATA_FILE_AUTHORITY))
        except Exception as err:
            raise RuntimeError(msg) from err


class BindCoreManager():
    DEFAULT_FIND_RUNNING_PID_TIMES = 5

    def __init__(self):
        self.npu_id_list = []
        self.running_pid_on_npu = {}
        self.find_running_pid_times = self.DEFAULT_FIND_RUNNING_PID_TIMES
        self.npu_affinity_cpu_dict = {}
        self.log_file = ''
        self._init_log_file()

    @staticmethod
    def _launch_process(cmd: list):
        logging.info('Start to execute cmd: %s', cmd)
        try:
            subprocess.Popen(cmd.split(), shell=False)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Failed to run cmd: {cmd}') from e

    def get_running_pid_on_npu(self) -> bool:
        no_running_pids_on_npu_msg = '[INFO] Now there is no running process on all NPUs, stop bind cores'
        logging.info('Begin to find running process on all NPUs')
        # get running process on NPUs
        for _ in range(self.find_running_pid_times):
            running_pid_on_npu = {}
            for npu_id in self.npu_id_list:
                get_npu_pids_cmd = 'npu-smi info -t proc-mem -i {} -c 0'.format(npu_id)
                get_npu_pids_process = subprocess.run(get_npu_pids_cmd.split(), shell=False, capture_output=True)
                res = get_npu_pids_process.stdout.decode('utf-8').split()
                pid_list = []
                for value in res:
                    if value.startswith('id:'):
                        pid = value.split(':')[1]
                        pid_list.append(pid)
                if pid_list:
                    running_pid_on_npu[npu_id] = list(set(pid_list))

            if len(running_pid_on_npu) < len(self.npu_id_list):
                time.sleep(5)
                continue

            if len(self.running_pid_on_npu.keys()) == len(running_pid_on_npu.keys()) and running_pid_on_npu:
                self.running_pid_on_npu = running_pid_on_npu
                break

            self.running_pid_on_npu = running_pid_on_npu

        # delete repeat pid
        for npu_id in self.npu_id_list:
            if npu_id not in self.running_pid_on_npu:
                continue
            pids_on_npu = self.running_pid_on_npu[npu_id]
            for npu_id_with_pids, pids in self.running_pid_on_npu.items():
                if npu_id == npu_id_with_pids:
                    continue
                pids_on_npu = list(set(pids_on_npu) - set(pids))
            self.running_pid_on_npu[npu_id] = pids_on_npu

        if_running_process = False
        for npu_id, pids in self.running_pid_on_npu.items():
            if not pids:
                logging.info('There is no running process on NPU %d', npu_id)
            else:
                logging.info('Succeed to find running process %s on NPU %d', pids, npu_id)
                if_running_process = True
        if not if_running_process:
            print(no_running_pids_on_npu_msg)
        return if_running_process

    def get_npu_info(self) -> bool:
        try:
            self._get_all_npu_id()
            if not self._get_npu_affinity():
                return False
        except subprocess.CalledProcessError:
            return False
        return True

    def run_bind_core(self):
        if not self.running_pid_on_npu:
            return
        for npu, pid_list in self.running_pid_on_npu.items():
            if npu not in self.npu_affinity_cpu_dict.keys():
                logging.warning('Cannot find affinity cpu for npu: %d', npu)
                continue
            affinity_cpu = self.npu_affinity_cpu_dict.get(npu)
            for pid in pid_list:
                try:
                    logging.info('Begin to bind cores for process %s on NPU %d', pid, npu)
                    set_affinity_cpu_cmd = 'taskset -pc {} {}'.format(affinity_cpu, pid)
                    p = subprocess.run(set_affinity_cpu_cmd.split(), shell=False, capture_output=True)
                    logging.info(p.stdout.decode('utf-8'))
                except subprocess.CalledProcessError:
                    print('[ERROR] Failed to bind process {} on NPU {} with cpu cores list {}'.format(pid, npu, affinity_cpu))

                logging.info('Succeed to bind process %s on NPU %d with cpu cores list %s', pid, npu, affinity_cpu)

    def args_parse(self):
        parser = argparse.ArgumentParser(description='This is a affinity cpu core bind script.')
        parser.add_argument('-t', '--time', type=int, metavar='', help='Wait time before bind cores that you want to set. The unit is \'s\'.')
        parser.add_argument('-app', '--application', metavar='', nargs='+', help='Training or inference command that you want to run.')
        args = parser.parse_args()
        if args.application:
            application_cmd = ' '.join(args.application)
            self._launch_process(application_cmd)
            time.sleep(2)
        # if time is set, wait for setting time before bind cores
        if args.time:
            time.sleep(args.time)

    def _init_log_file(self):
        now_time = datetime.now(tz=timezone.utc)
        time_stamp = str(now_time.year) + '_' + \
                     str(now_time.month) + '_' + \
                     str(now_time.day) + '_' + \
                     str(now_time.hour) + '_' + \
                     str(now_time.minute) + '_' + \
                     str(now_time.second)
        log_file_name = 'bind_core_' + time_stamp + '.log'
        msg = f"Failed to create file: {log_file_name}"
        try:
            PathManager.create_file_safety(os.path.join(os.getcwd(), log_file_name))
        except RuntimeError as err:
            raise RuntimeError(msg) from err
        self.log_file = log_file_name
        logging.basicConfig(filename=self.log_file,
                            level=logging.INFO,
                            filemode='w',
                            format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')

    def _get_all_npu_id(self) -> None:
        get_npu_info_cmd = 'npu-smi info -l'
        get_npu_info_process = subprocess.run(get_npu_info_cmd.split(), shell=False, capture_output=True)
        get_npu_id_cmd = 'grep ID'
        get_npu_id_process = subprocess.run(get_npu_id_cmd.split(), shell=False, input=get_npu_info_process.stdout, capture_output=True)
        res = get_npu_id_process.stdout.decode('utf-8').split()
        for i in res:
            if i.isdigit():
                self.npu_id_list.append(int(i))
        logging.info('NPU total id list: %s', self.npu_id_list)

    def _get_npu_affinity(self) -> bool:
        cpu_num = os.cpu_count()
        cpu_num_for_each_npu = cpu_num // len(self.npu_id_list)
        get_npu_topo_cmd = 'npu-smi info -t topo'
        p = subprocess.run(get_npu_topo_cmd.split(), shell=False, capture_output=True)
        res = p.stdout.decode('utf-8').split()
        if not res:
            print('[ERROR] Failed to run get npu affinity info, please check if driver version support cmd npu-smi info -t topo')
            return False

        index = 0
        for v in res:
            if '-' in v:
                affinity_cpus = []
                cpu_lists = v.split(',')
                for cpu_list in cpu_lists:
                    cpus = cpu_list.split('-')
                    if len(cpus) != 2:
                        continue
                    if int(cpus[1]) - int(cpus[0]) == cpu_num_for_each_npu - 1:
                        cpus[1] = str(int(cpus[1]) + cpu_num_for_each_npu)
                    affinity_cpus.append(cpus[0] + '-' + cpus[1])
                if index < len(self.npu_id_list):
                    self.npu_affinity_cpu_dict[self.npu_id_list[index]] = ','.join(affinity_cpu for affinity_cpu in affinity_cpus)
                    index += 1
                else:
                    print('[ERROR] Get affinity_cpu_list for {} npus, more than real npu num: {}'.format(index + 1, len(self.npu_id_list)))
                    return False

        for k in self.npu_affinity_cpu_dict.keys():
            logging.info('Affinity CPU list [%s] for NPU %d', self.npu_affinity_cpu_dict[k], k)
        return True


if __name__ == '__main__':
    print('[INFO] Begin to run bind-cores script...')
    bind_core_manager = BindCoreManager()
    bind_core_manager.args_parse()

    if not bind_core_manager.get_npu_info():
        print('[ERROR] Failed to get current npus info')
        exit()

    if not bind_core_manager.get_running_pid_on_npu():
        exit()
    bind_core_manager.run_bind_core()
    print('[INFO] End to run bind-cores script, the log is saved in {}'.format(bind_core_manager.log_file))