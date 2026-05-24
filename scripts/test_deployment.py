import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class TestResult:
    name: str
    passed: bool
    status_code: int | None = None
    latency_ms: float = 0.0
    detail: str = ""


@dataclass
class Suite:
    results: list[TestResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return len(self.results) - self.passed

    def add(self, result: TestResult) -> None:
        self.results.append(result)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.name}  ({result.latency_ms:.0f} ms)  {result.detail}")

    def print_summary(self) -> None:
        total = len(self.results)
        print(f"\n{'─' * 50}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.failed:
            print("Failed tests:")
            for r in self.results:
                if not r.passed:
                    print(f"  • {r.name}: {r.detail}")


class DeploymentTester:
    """Runs a suite of smoke tests against a deployed API instance."""

    _SAMPLE_PRICES: list[float] = [100.0 + i * 0.5 for i in range(60)]

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    def _get(self, path: str) -> tuple[requests.Response, float]:
        t0 = time.perf_counter()
        resp = self._session.get(f"{self._base}{path}", timeout=self._timeout)
        return resp, (time.perf_counter() - t0) * 1_000

    def _post(self, path: str, body: Any) -> tuple[requests.Response, float]:
        t0 = time.perf_counter()
        resp = self._session.post(
            f"{self._base}{path}", json=body, timeout=self._timeout
        )
        return resp, (time.perf_counter() - t0) * 1_000

    def _make_result(
        self,
        name: str,
        resp: requests.Response,
        elapsed_ms: float,
        *,
        expected_status: int = 200,
        extra_check: str = "",
    ) -> TestResult:
        passed = resp.status_code == expected_status
        detail = extra_check if extra_check else f"HTTP {resp.status_code}"
        if not passed:
            detail = f"expected {expected_status}, got {resp.status_code}"
        return TestResult(name=name, passed=passed, status_code=resp.status_code,
                          latency_ms=elapsed_ms, detail=detail)

    def test_health(self) -> TestResult:
        resp, ms = self._get("/health")
        passed = resp.status_code == 200
        detail = ""
        if passed:
            data = resp.json()
            if data.get("model") != "loaded":
                passed = False
                detail = f"model field is '{data.get('model')}', expected 'loaded'"
            else:
                detail = f"model={data.get('model')}"
        else:
            detail = f"HTTP {resp.status_code}"
        return TestResult(name="GET /health", passed=passed,
                          status_code=resp.status_code, latency_ms=ms, detail=detail)

    def test_model_info(self) -> TestResult:
        resp, ms = self._get("/model-info")
        result = self._make_result("GET /model-info", resp, ms)
        if result.passed:
            data = resp.json()
            result.detail = f"version={data.get('version')}  seq_len={data.get('sequence_length')}"
        return result

    def test_predict_valid(self) -> TestResult:
        body = {"stock_symbol": "PETR4.SA", "historical_prices": self._SAMPLE_PRICES}
        resp, ms = self._post("/predict", body)
        result = self._make_result("POST /predict (valid)", resp, ms)
        if result.passed:
            data = resp.json()
            price = data.get("predicted_price")
            result.detail = f"predicted_price={price:.4f}" if price else "no price field"
        return result

    def test_predict_too_few_prices(self) -> TestResult:
        body = {"stock_symbol": "PETR4.SA", "historical_prices": [100.0] * 10}
        resp, ms = self._post("/predict", body)
        passed = resp.status_code == 422
        detail = "correctly rejected" if passed else f"expected 422, got {resp.status_code}"
        return TestResult(name="POST /predict (too few prices)", passed=passed,
                          status_code=resp.status_code, latency_ms=ms, detail=detail)

    def test_predict_negative_price(self) -> TestResult:
        bad_prices = self._SAMPLE_PRICES.copy()
        bad_prices[0] = -5.0
        body = {"stock_symbol": "PETR4.SA", "historical_prices": bad_prices}
        resp, ms = self._post("/predict", body)
        passed = resp.status_code == 422
        detail = "correctly rejected" if passed else f"expected 422, got {resp.status_code}"
        return TestResult(name="POST /predict (negative price)", passed=passed,
                          status_code=resp.status_code, latency_ms=ms, detail=detail)

    def test_predict_batch(self) -> TestResult:
        body = {
            "requests": [
                {"stock_symbol": "PETR4.SA", "historical_prices": self._SAMPLE_PRICES},
                {"stock_symbol": "VALE3.SA", "historical_prices": [i + 50.0 for i in range(60)]},
            ]
        }
        resp, ms = self._post("/predict-batch", body)
        result = self._make_result("POST /predict-batch", resp, ms)
        if result.passed:
            data = resp.json()
            result.detail = f"total={data.get('total')}"
        return result

    def test_latency_under_threshold(self, threshold_ms: float = 2_000) -> TestResult:
        body = {"stock_symbol": "PETR4.SA", "historical_prices": self._SAMPLE_PRICES}
        resp, ms = self._post("/predict", body)
        passed = resp.status_code == 200 and ms < threshold_ms
        detail = f"{ms:.0f} ms (threshold {threshold_ms:.0f} ms)"
        return TestResult(name="POST /predict latency", passed=passed,
                          status_code=resp.status_code, latency_ms=ms, detail=detail)

    def test_docs_reachable(self) -> TestResult:
        resp, ms = self._get("/docs")
        return self._make_result("GET /docs", resp, ms)

    def run(self) -> Suite:
        print(f"\nTesting API at: {self._base}\n{'─' * 50}")
        suite = Suite()
        tests = [
            self.test_health,
            self.test_model_info,
            self.test_predict_valid,
            self.test_predict_too_few_prices,
            self.test_predict_negative_price,
            self.test_predict_batch,
            self.test_latency_under_threshold,
            self.test_docs_reachable,
        ]
        for test_fn in tests:
            try:
                suite.add(test_fn())
            except requests.exceptions.ConnectionError:
                suite.add(TestResult(name=test_fn.__name__, passed=False,
                                     detail="connection refused"))
            except requests.exceptions.Timeout:
                suite.add(TestResult(name=test_fn.__name__, passed=False,
                                     detail=f"timed out after {self._timeout}s"))
        suite.print_summary()
        return suite


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test a deployed LSTM API instance.")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the deployed API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="HTTP request timeout in seconds (default: 30)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tester = DeploymentTester(base_url=args.url, timeout=args.timeout)
    suite = tester.run()
    sys.exit(0 if suite.failed == 0 else 1)


if __name__ == "__main__":
    main()
