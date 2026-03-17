# Задача

- Пользователь попросил отладить и исправить корректность `vulkan.radixSort` на `Intel UHD 770`.
- Если исправление могло вредить скорости на NVIDIA, нужно было оставить для NVIDIA один вариант, а для остальных другой.
- Затем нужно было проверить корректность и на `lavapipe`.

# Ход работы

- Сначала проверила гипотезу про subgroup size:
  - быстрый путь был жёстко написан под `subgroup = 32`;
  - pipeline требовал `requiredSubgroupSize` по неправильному условию.
- Исправила условие в `engine.cpp`:
  - теперь `requiredSubgroupSize = 32` запрашивается через `VK_EXT_subgroup_size_control`.
- После этого:
  - `Intel UHD 770` начал корректно проходить быстрый путь;
  - `lavapipe` всё равно оставался некорректным.
- Для надёжности добавила второй набор portable radix-sort shaders и вернула старый алгоритм как fallback.
- Итоговая логика выбора:
  - `NVIDIA` и `Intel` -> `fast-subgroup`
  - остальные Vulkan devices -> `portable-fallback`

# Проверка

- `Intel UHD 770`
  - `fast-subgroup`
  - median `0.0102515 s`
  - `PASSED`
- `NVIDIA GeForce GTX 1650`
  - `fast-subgroup`
  - median `0.00364193 s`
  - `PASSED`
- `llvmpipe`
  - `portable-fallback`
  - median `0.161213 s`
  - `PASSED`

# Подробности

- Подробная документация обновлена в:
  - `docs/vulkan_radixsort_gtx1650_optimization.md`
