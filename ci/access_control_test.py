import argparse
import os
from pathlib import Path


# =============================
# ST test, run with shell
# =============================

class UT_Test:

    def __init__(self):

        base_dir = Path(__file__).absolute().parent.parent
        test_dir = os.path.join(base_dir, 'tests')
        self.ut_file = os.path.join(test_dir, "ut")
    
    def run_ut(self):
        command = f"python3.8 -m pytest -k 'not allocator' {self.ut_file}"
        ut_exitcode = os.system(command)
        if ut_exitcode == 0:
            print("UT test success")
        else:
            print("UT failed")
            exit(1)


class ST_Test:
    
    def __init__(self):

        base_dir = Path(__file__).absolute().parent.parent
        test_dir = os.path.join(base_dir, 'tests')

        st_dir = "st"
        test_shell_file = os.path.join(
            test_dir, st_dir, "test_st_demo.sh")

        self.st_file_list = [
            test_shell_file
        ]

    def run_st(self):
        all_success = True
        for shell_file in self.st_file_list:
            command = f"sh {shell_file}"
            st_exitcode = os.system(command)
            if st_exitcode != 0:
                all_success = False
                print(f"ST run {shell_file} failed")
                break

        if all_success:
            print("ST test success")
        else:
            print("ST failed")
            exit(1)


# ===============================================
# UT test, run with pytest, waiting for more ...
# ===============================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control needed test cases")
    parser.add_argument("--type", type=str, default="all", 
                        help='Test cases type. `all`: run all test cases; `ut`: run ut case,' '`st`: run st cases;')
    options = parser.parse_args()
    print(f"options: {options}")
    if options.type == "st":
        st = ST_Test()
        st.run_st()
    elif options.type == "ut":
        ut = UT_Test()
        ut.run_ut()
    elif options.type == "all":
        st = ST_Test()
        st.run_st()
        ut = UT_Test()
        ut.run_ut()
    else:
        raise ValueError(f"TEST CASE TYPE ERROR: no type `{options.type}`")