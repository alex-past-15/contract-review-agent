"""Оценка риска по простым правилам, чтобы юрист мог их прочитать и поправить."""

# категория: (уровень риска, если пункт найден; пояснение)
PRESENT = {
    "Uncapped Liability": ("high", "Ответственность не ограничена: потенциальный ущерб не имеет потолка."),
    "Non-Compete": ("high", "Ограничение конкуренции: может закрыть для компании часть рынка или клиентов."),
    "Liquidated Damages": ("medium", "Фиксированная неустойка: сумма штрафа определена заранее."),
    "Exclusivity": ("medium", "Эксклюзивность: обязательства работать только с одной стороной."),
    "Change Of Control": ("medium", "Смена контроля может дать контрагенту право расторгнуть или изменить договор."),
    "Termination For Convenience": ("medium", "Досрочный выход без причины: стабильность договора под вопросом."),
    "Renewal Term": ("medium", "Автопродление: договор может продлиться без явного решения сторон."),
    "Notice Period To Terminate Renewal": ("medium", "Срок уведомления об отказе от продления: пропуск дедлайна означает продление."),
    "Anti-Assignment": ("low", "Ограничение передачи прав: проверить, влияет ли на сделки M&A и реструктуризацию."),
    "Ip Ownership Assignment": ("medium", "Передача прав на интеллектуальную собственность: проверить объём передачи."),
    "Audit Rights": ("low", "Право аудита: подготовить учёт и доступ к документам."),
    "Cap On Liability": ("low", "Ответственность ограничена: проверить, достаточен ли лимит."),
}

# иногда риск в том, что пункта нет
ABSENT = {
    "Cap On Liability": ("medium", "Пункт об ограничении ответственности не найден: возможна неограниченная ответственность."),
}

ORDER = {"high": 0, "medium": 1, "low": 2}


def score(found: set[str], all_categories: list[str]) -> list[dict]:
    items = []
    for cat in all_categories:
        if cat in found:
            level, why = PRESENT[cat]
            items.append({"category": cat, "risk": level, "why": why, "status": "found"})
        elif cat in ABSENT:
            level, why = ABSENT[cat]
            items.append({"category": cat, "risk": level, "why": why, "status": "absent"})
    return sorted(items, key=lambda x: ORDER[x["risk"]])


def overall(items: list[dict]) -> str:
    levels = {i["risk"] for i in items}
    return "high" if "high" in levels else "medium" if "medium" in levels else "low"
