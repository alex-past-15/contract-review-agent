"""Оценка экономии в рублях.
Recall берём из замеров на тесте, остальные цифры - допущения, их можно менять."""
from dataclasses import dataclass


@dataclass
class Assumptions:
    contracts_per_month: int = 200   # договоров в месяц
    manual_hours: float = 1.5        # часов на ручной первичный разбор одного договора
    assisted_hours: float = 0.4      # часов на проверку отчёта агента
    lawyer_rate: float = 2500.0      # стоимость часа юриста, руб.
    miss_cost: float = 40000.0       # средний ущерб от пропущенного существенного риска, руб.
    miss_matters: float = 0.10       # доля пропусков, которые оказываются существенными


def monthly_effect(a: Assumptions, recall: float) -> dict:
    manual = a.contracts_per_month * a.manual_hours * a.lawyer_rate
    assisted = a.contracts_per_month * a.assisted_hours * a.lawyer_rate
    # то, что агент пропустил, юрист при быстрой проверке отчёта тоже не увидит
    # грубо считаем, что в договоре около 5 значимых пунктов
    missed = a.contracts_per_month * 5 * (1 - recall) * a.miss_matters
    risk_cost = missed * a.miss_cost
    saving = manual - assisted - risk_cost
    return {
        "manual_cost": round(manual),
        "assisted_cost": round(assisted),
        "risk_cost": round(risk_cost),
        "net_saving": round(saving),
        "hours_saved": round(a.contracts_per_month * (a.manual_hours - a.assisted_hours), 1),
        "saving_pct": round(100 * saving / manual, 1),
    }
