"""Calibração de limiares: acurácia top-1 por população, falso positivo e latência."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import yaml

from cyhmo.domain.contracts import GameState, Interpretation, Transcript
from cyhmo.domain.errors import CyhmoError
from cyhmo.domain.ports import Interpreter

GRAMMARS_DIRNAME = "grammars"
InterpreterFactory = Callable[[float, float], Interpreter]


class CalibrationDatasetError(CyhmoError):
    """Dataset de calibração ausente ou fora do formato esperado."""


@dataclass(frozen=True)
class CalibrationCase:
    text: str
    expected: tuple[str, ...]
    lang: str = "pt-BR"
    mode: str = "battle"
    enemies: int | None = None
    core: bool = True

    def state(self, grammar: tuple[str, ...]) -> GameState:
        enemies = None if self.enemies is None else tuple({"index": index + 1} for index in range(self.enemies))
        return GameState(mode=self.mode, can_talk=True, enemies=enemies, grammar=grammar)


@dataclass(frozen=True)
class CalibrationDataset:
    version: int
    grammar: tuple[str, ...]
    cases: tuple[CalibrationCase, ...]
    source: str = ""


@dataclass(frozen=True)
class SpontaneousDataset:
    version: int
    phrases: tuple[str, ...]
    source: str = ""


@dataclass(frozen=True)
class ThresholdResult:
    accept_threshold: float
    reject_threshold: float
    cases: int
    correct: int
    core_cases: int
    core_correct: int
    with_primary_cases: int
    with_primary_correct: int
    without_primary_cases: int
    without_primary_correct: int
    spontaneous: int
    false_positives: int
    latency_p50_ms: float
    latency_p95_ms: float
    failures: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def top1_accuracy(self) -> float:
        return _ratio(self.correct, self.cases)

    @property
    def core_accuracy(self) -> float:
        return _ratio(self.core_correct, self.core_cases)

    @property
    def with_primary_accuracy(self) -> float:
        return _ratio(self.with_primary_correct, self.with_primary_cases)

    @property
    def without_primary_accuracy(self) -> float:
        return _ratio(self.without_primary_correct, self.without_primary_cases)

    @property
    def false_positive_rate(self) -> float:
        return _ratio(self.false_positives, self.spontaneous)


@dataclass(frozen=True)
class CalibrationReport:
    results: tuple[ThresholdResult, ...]
    dataset: str = ""
    spontaneous: str = ""

    def best(self, max_false_positive_rate: float = 0.02) -> ThresholdResult | None:
        eligible = [result for result in self.results if result.false_positive_rate <= max_false_positive_rate]
        return max(eligible, key=lambda result: (result.top1_accuracy, -result.accept_threshold), default=None)


def load_dataset(path: Path) -> CalibrationDataset:
    raw = _read_yaml(Path(path))
    cases = raw.get("cases")
    if not isinstance(cases, list) or not cases:
        raise CalibrationDatasetError(f"{path}: 'cases' deve ser uma lista não vazia")
    grammar = _resolve_grammar(raw.get("grammar"), Path(path).parent)
    parsed = tuple(_parse_case(item, Path(path)) for item in cases)
    return CalibrationDataset(int(raw.get("version", 1)), grammar, parsed, str(path))


def load_spontaneous(path: Path) -> SpontaneousDataset:
    raw = _read_yaml(Path(path))
    phrases = raw.get("phrases")
    if not isinstance(phrases, list) or not phrases:
        raise CalibrationDatasetError(f"{path}: 'phrases' deve ser uma lista não vazia")
    return SpontaneousDataset(int(raw.get("version", 1)), tuple(str(phrase) for phrase in phrases), str(path))


def load_grammar_fixture(path: Path) -> tuple[str, ...]:
    raw = _read_yaml(Path(path))
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise CalibrationDatasetError(f"{path}: 'entries' deve ser uma lista não vazia")
    return tuple(dict.fromkeys(str(entry) for entry in entries))


def run_calibration(
    make_interpreter: InterpreterFactory,
    dataset: CalibrationDataset,
    spontaneous: SpontaneousDataset | None,
    grid: Sequence[tuple[float, float]],
    on_progress: Callable[[int, int], bool] | None = None,
) -> CalibrationReport:
    """``on_progress(feitos, total)`` devolvendo ``False`` interrompe a varredura —
    é como a interface para uma calibração longa sem matar a thread na marra."""
    results: list[ThresholdResult] = []
    for accept, reject in grid:
        results.append(_evaluate(make_interpreter(accept, reject), accept, reject, dataset, spontaneous))
        if on_progress is not None and not on_progress(len(results), len(grid)):
            break
    return CalibrationReport(tuple(results), dataset.source, "" if spontaneous is None else spontaneous.source)


def format_report(report: CalibrationReport) -> str:
    header = f"{'accept':>7} {'reject':>7} {'top1':>6} {'core':>6} {'c/pt':>6} {'s/pt':>6} {'FP':>6} {'p50ms':>7} {'p95ms':>7}"
    lines = [f"dataset: {report.dataset}", f"espontâneo: {report.spontaneous or '-'}", header]
    for result in report.results:
        lines.append(
            f"{result.accept_threshold:>7.2f} {result.reject_threshold:>7.2f} "
            f"{result.top1_accuracy:>6.1%} {result.core_accuracy:>6.1%} "
            f"{result.with_primary_accuracy:>6.1%} {result.without_primary_accuracy:>6.1%} "
            f"{result.false_positive_rate:>6.1%} {result.latency_p50_ms:>7.1f} {result.latency_p95_ms:>7.1f}"
        )
    best = report.best()
    lines.append(
        "melhor par (FP ≤ 2%): "
        + ("nenhum" if best is None else f"accept={best.accept_threshold:.2f} reject={best.reject_threshold:.2f}")
    )
    return "\n".join(lines)


def to_payload(report: CalibrationReport, include_latency: bool = True) -> dict[str, Any]:
    best = report.best()
    return {
        "dataset": report.dataset,
        "spontaneous": report.spontaneous,
        "results": [_result_payload(result, include_latency) for result in report.results],
        "best": None if best is None else _result_payload(best, include_latency),
    }


def to_json(report: CalibrationReport, include_latency: bool = True) -> str:
    return json.dumps(to_payload(report, include_latency), ensure_ascii=False, indent=2, sort_keys=True)


def _evaluate(
    interpreter: Interpreter,
    accept: float,
    reject: float,
    dataset: CalibrationDataset,
    spontaneous: SpontaneousDataset | None,
) -> ThresholdResult:
    latencies: list[float] = []
    failures: list[dict[str, Any]] = []
    tally = {key: 0 for key in ("correct", "core", "core_correct", "with", "with_correct", "without", "without_correct")}
    for case in dataset.cases:
        interpretation = interpreter.interpret(_transcript(case.text, case.lang), case.state(dataset.grammar))
        latencies.append(interpretation.latency_ms)
        correct = interpretation.keys == case.expected
        population = _population(interpreter, interpretation, case)
        tally["correct"] += correct
        tally["core"] += case.core
        tally["core_correct"] += case.core and correct
        tally[population] += 1
        tally[f"{population}_correct"] += correct
        if not correct:
            failures.append({"text": case.text, "expected": list(case.expected), "got": list(interpretation.keys), "reason": interpretation.reason})
    false_positives = 0
    phrases = () if spontaneous is None else spontaneous.phrases
    for phrase in phrases:
        interpretation = interpreter.interpret(_transcript(phrase, "pt-BR"), GameState(mode="unknown", can_talk=True, grammar=dataset.grammar))
        latencies.append(interpretation.latency_ms)
        false_positives += not interpretation.is_empty
    return ThresholdResult(
        accept_threshold=accept,
        reject_threshold=reject,
        cases=len(dataset.cases),
        correct=tally["correct"],
        core_cases=tally["core"],
        core_correct=tally["core_correct"],
        with_primary_cases=tally["with"],
        with_primary_correct=tally["with_correct"],
        without_primary_cases=tally["without"],
        without_primary_correct=tally["without_correct"],
        spontaneous=len(phrases),
        false_positives=false_positives,
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
        failures=tuple(failures),
    )


def _population(interpreter: Interpreter, interpretation: Interpretation, case: CalibrationCase) -> str:
    probe = getattr(interpreter, "has_primary_language_examples", None)
    key = case.expected[0] if case.expected else (interpretation.keys[0] if interpretation.keys else None)
    if key is None or probe is None:
        return "without"
    return "with" if probe(key) else "without"


def _transcript(text: str, lang: str) -> Transcript:
    return Transcript(text=text, lang=lang, confidence=1.0, t_speech_end=0.0, raw_text=text)


def _resolve_grammar(spec: Any, base_dir: Path) -> tuple[str, ...]:
    names = [spec] if isinstance(spec, str) else spec
    if not isinstance(names, list) or not names:
        raise CalibrationDatasetError("'grammar' deve ser o nome de uma fixture em datasets/grammars ou uma lista de nomes")
    entries: dict[str, None] = {}
    for name in names:
        fixture = base_dir / GRAMMARS_DIRNAME / f"{name}.yaml"
        if not fixture.exists():
            raise CalibrationDatasetError(f"fixture de gramática não encontrada: {fixture}")
        for entry in load_grammar_fixture(fixture):
            entries.setdefault(entry, None)
    return tuple(entries)


def _parse_case(item: Any, path: Path) -> CalibrationCase:
    if not isinstance(item, dict) or not str(item.get("text", "")).strip():
        raise CalibrationDatasetError(f"{path}: caso sem 'text': {item!r}")
    expected = item.get("expected", [])
    if not isinstance(expected, list):
        raise CalibrationDatasetError(f"{path}: 'expected' de {item['text']!r} deve ser lista")
    enemies = item.get("enemies")
    return CalibrationCase(
        text=str(item["text"]),
        expected=tuple(str(key) for key in expected),
        lang=str(item.get("lang", "pt-BR")),
        mode=str(item.get("mode", "battle")),
        enemies=None if enemies is None else int(enemies),
        core=bool(item.get("core", True)),
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CalibrationDatasetError(f"{path}: não foi possível ler — {exc}") from exc
    if not isinstance(raw, dict):
        raise CalibrationDatasetError(f"{path}: esperado um mapeamento YAML")
    return raw


def _result_payload(result: ThresholdResult, include_latency: bool) -> dict[str, Any]:
    payload = asdict(result)
    payload["failures"] = list(result.failures)
    payload.update(
        top1_accuracy=round(result.top1_accuracy, 4),
        core_accuracy=round(result.core_accuracy, 4),
        with_primary_accuracy=round(result.with_primary_accuracy, 4),
        without_primary_accuracy=round(result.without_primary_accuracy, 4),
        false_positive_rate=round(result.false_positive_rate, 4),
    )
    if not include_latency:
        payload.pop("latency_p50_ms")
        payload.pop("latency_p95_ms")
    return payload


def _percentile(values: Sequence[float], percent: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    quantiles = statistics.quantiles(values, n=100, method="inclusive")
    return float(quantiles[percent - 1])


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
