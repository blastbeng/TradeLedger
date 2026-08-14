import time
import threading
from collections import defaultdict

class HealthMetrics:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self._llm_calls = defaultdict(lambda: {"success": 0, "failure": 0})
        self._data_sources = defaultdict(lambda: {"available": True, "last_check": 0.0})
        self._loop_latencies = defaultdict(list)
        self._task_times = defaultdict(list)
        self._metrics_lock = threading.Lock()

    def record_llm_call(self, model: str, success: bool):
        with self._metrics_lock:
            status = "success" if success else "failure"
            self._llm_calls[model][status] += 1

    def record_data_source(self, source: str, available: bool):
        with self._metrics_lock:
            self._data_sources[source]["available"] = available
            self._data_sources[source]["last_check"] = time.time()

    def record_loop_latency(self, loop_name: str, duration: float):
        with self._metrics_lock:
            self._loop_latencies[loop_name].append(duration)
            if len(self._loop_latencies[loop_name]) > 100:
                self._loop_latencies[loop_name].pop(0)

    def record_task_time(self, task_name: str, duration: float):
        with self._metrics_lock:
            self._task_times[task_name].append(duration)
            if len(self._task_times[task_name]) > 100:
                self._task_times[task_name].pop(0)

    def get_metrics(self) -> dict:
        with self._metrics_lock:
            return {
                "llm_calls": dict(self._llm_calls),
                "data_sources": dict(self._data_sources),
                "loop_latencies": {k: sum(v)/len(v) if v else 0 for k, v in self._loop_latencies.items()},
                "task_times": {k: sum(v)/len(v) if v else 0 for k, v in self._task_times.items()},
            }

health_metrics = HealthMetrics()
