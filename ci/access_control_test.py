import argparse
import os
from pathlib import Path


def read_files_from_txt(txt_file):
    with open(txt_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def is_examples(file):
    return file.startswith("example/")


def is_pipecase(file):
    return file.startswith("tests/pipeline")


def is_markdown(file):
    return file.endswith(".md")


def skip_ci_file(files, skip_cond):
    for file in files:
        if not any(condition(file) for condition in skip_cond):
            return False
    return True


def alter_skip_ci():
    parent_dir = Path(__file__).absolute().parents[2]
    raw_txt_file = os.path.join(parent_dir, "modify.txt")

    if not os.path.exists(raw_txt_file):
        return False
    
    file_list = read_files_from_txt(raw_txt_file)
    skip_conds = [
        is_examples,
        is_pipecase,
        is_markdown
    ]

    return skip_ci_file(file_list, skip_conds)


def acquire_exitcode(command):
    exitcode = os.system(command)
    real_code = os.WEXITSTATUS(exitcode)
    return real_code


# =============================
# UT test, run with pytest
# =============================

class UT_Test:

    def __init__(self):

        base_dir = Path(__file__).absolute().parent.parent
        test_dir = os.path.join(base_dir, 'tests')
        self.ut_file = os.path.join(test_dir, "ut")
    
    def run_ut(self):
        command = f"pytest -x {self.ut_file}"
        code = acquire_exitcode(command)
        if code == 0:
            print("UT test success")
        else:
            print("UT failed")
        return code


# ===============================================
# ST test, run with sh.
# ===============================================

class ST_Test:
    
    def __init__(self):

        base_dir = Path(__file__).absolute().parent.parent
        test_dir = os.path.join(base_dir, 'tests')

        st_dir = "st"
        self.st_shell = os.path.join(
            test_dir, st_dir, "st_run.sh"
        )

    def run_st(self):
        command = f"bash {self.st_shell}"
        code = acquire_exitcode(command)
        
        if code == 0:
            print("ST test success")
        else:
            print("ST failed")
        return code


def run_ut_tests():
    ut = UT_Test()
    return ut.run_ut()


def run_st_tests():
    st = ST_Test()
    return st.run_st()


def run_tests(options):
    if options.type == "st":
        return run_st_tests()
    elif options.type == "ut":
        return run_ut_tests()
    elif options.type == "all":
        code = run_ut_tests()
        if code == 0:
            return run_st_tests()
        return code
    else:
        raise ValueError(f"TEST CASE TYPE ERROR: no type `{options.type}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Control needed test cases")
    parser.add_argument("--type", type=str, default="all", 
                        help='Test cases type. `all`: run all test cases; `ut`: run ut case,' '`st`: run st cases;')
    args = parser.parse_args()
    print(f"options: {args}")
    if alter_skip_ci():
        print("Skipping CI")
    else:
        exit_code = run_tests(args)
        if exit_code != 0:
            exit(exit_code)