"""Реестр планировщиков: декоратор @register_planner и фабрика by_name."""

from __future__ import annotations

from typing import Callable, Type

from apex_lab.algorithms.base import BasePlanner

_REGISTRY: dict[str, Type[BasePlanner]] = {}


def register_planner(name: str) -> Callable[[Type[BasePlanner]], Type[BasePlanner]]:
    """Декоратор для регистрации класса планировщика по строковому ключу.

    Пример::

        @register_planner("a_star")
        class AStarPlanner(BasePlanner):
            ...

    Args:
        name: Строковый идентификатор (используется в YAML-конфигах и CLI).

    Returns:
        Тот же класс без изменений (декоратор — только регистрация).
    """
    def decorator(cls: Type[BasePlanner]) -> Type[BasePlanner]:
        if name in _REGISTRY:
            raise ValueError(f"Планировщик '{name}' уже зарегистрирован.")
        _REGISTRY[name] = cls
        return cls
    return decorator


def get_planner(name: str) -> Type[BasePlanner]:
    """Вернуть класс планировщика по имени.

    Args:
        name: Строковый ключ, зарегистрированный через @register_planner.

    Raises:
        KeyError: Если планировщик с таким именем не найден.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(
            f"Планировщик '{name}' не найден. Доступные: {available}"
        )
    return _REGISTRY[name]


def list_planners() -> list[str]:
    """Список зарегистрированных имён планировщиков (в алфавитном порядке)."""
    return sorted(_REGISTRY)
