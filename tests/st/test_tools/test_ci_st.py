import pytest
from test_tools.acquire_json import transfer_logs_as_json, read_json, check_is_valid


WARM_UP = 5


class TestCIST:

    margin_loss = 0.01 # loss可允许误差范围
    margin_time_percent = 0.05 # 性能可允许波动百分比
    margin_memory_percent = 0.1 # 内存可允许波动百分比

    def _get_baseline(self, baseline_json):
        # acquire expected results
        self.expected = read_json(baseline_json)
        self.warm_up = self.expected.get("warm_up", WARM_UP)

    def _get_actual(self, generate_log, generate_json):
        # acquire actual results
        transfer_logs_as_json(generate_log, generate_json)
        self.actual = read_json(generate_json)

    def _test_helper(self, test_obj):
        """
        Core test function

        Args:
            test_obj: the object we want to test compare.
            test_type: deterministic or approximate, default is None.

        Here we temperally test `loss`, 'time' and `allocated memory`
        """
        comparison_selection = {
            "loss": self._compare_loss,
            "time": self._compare_time,
            "memo info": self._compare_memory
        }

        if test_obj in comparison_selection:
            print(f"===================== Begin comparing {test_obj} ===================")
            expected_list = self.expected[test_obj]
            actual_list = self.actual[test_obj]
            print(f"The list of expected values: {expected_list}")
            print(f"The list of actual values: {actual_list}")
            # Check if lists exist and are non-empty
            if not actual_list:
                raise ValueError(f"Actual list for {test_obj} is empty or not found. Maybe program has failed! Check it.")

            # Check if lists have the same length
            if len(expected_list) != len(actual_list):
                raise ValueError(f"Actual lengths of the lists for {test_obj} do not match. Maybe program has failed! Check it.")

            compare_func = comparison_selection[test_obj]
            compare_func(expected_list, actual_list)
        else:
            raise ValueError(f"Unsupported test object: {test_obj}")

    def _compare_loss(self, expected_list, actual_list):
        # Because "deterministic computation" affects the throughput, so we just test
        # lm loss in case of approximation.
        for step, (expected_val, actual_val) in enumerate(zip(expected_list, actual_list)):
            print(f"Checking step {step + 1} for loss")
            if actual_val == pytest.approx(expected=expected_val, rel=self.margin_loss):
                raise AssertionError(f"The loss at step {step} should be approximate to {expected_val} but it is {actual_val}.")

    def _compare_time(self, expected_list, actual_list):
        # First few iterations might take a little longer. So we take the last 70 percent of the timings
        expected_steps = len(expected_list) - self.warm_up
        actual_steps = len(actual_list) - self.warm_up
        if expected_steps <= 0 or actual_steps <= 0:
            raise ValueError(f"Warm up steps must less than expected steps {len(expected_list)} or actual steps {len(actual_list)}")
        expected_avg_time = sum(expected_list[self.warm_up:]) / expected_steps
        actual_avg_time = sum(actual_list[self.warm_up:]) / actual_steps

        check_is_valid(
            actual_val=actual_avg_time,
            expected_val=expected_avg_time,
            margin=self.margin_time_percent,
            greater=True,
            message=f"The actual avg time {actual_avg_time} exceed expected avg time {expected_avg_time}"
        )

    def _compare_memory(self, expected_list, actual_list):
        for expected_val, actual_val in zip(expected_list, actual_list):
            check_is_valid(
                actual_val=actual_val["allocated memory"],
                expected_val=expected_val["allocated memory"],
                margin=self.margin_memory_percent,
                greater=True,
                message=f'The actual memory {actual_val["allocated memory"]} seems to be abnormal compare to expected {expected_val["allocated memory"]}.'
            )
            check_is_valid(
                actual_val=actual_val["max allocated memory"],
                expected_val=expected_val["max allocated memory"],
                margin=self.margin_memory_percent,
                greater=True,
                message=f'The actual max memory {actual_val["max allocated memory"]} seems to be abnormal compare to expected {expected_val["max allocated memory"]}.'
            )

    def test_time(self, baseline_json, generate_log, generate_json):
        self._get_baseline(baseline_json)
        self._get_actual(generate_log, generate_json)
        self._test_helper("time")

    def test_allocated_memory(self, baseline_json, generate_log, generate_json):
        self._get_baseline(baseline_json)
        self._get_actual(generate_log, generate_json)
        self._test_helper("memo info")
