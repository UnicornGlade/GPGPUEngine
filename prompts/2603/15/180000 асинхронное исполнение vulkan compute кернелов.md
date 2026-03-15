# Асинхронное исполнение Vulkan compute кернелов

## Сообщения пользователя

- `новый промпт - нужно сделать асинхронное исполнение кернелов, сейчас после запуска кернела выполняется блокирующее ожидание`
- `... подумай какие из уже имеющихся юнит тестов от этого значительно замедляются ...`
- `... придумай новые юнит тесты ...`
- `... отпрофилируй с помощью nvidia nsight vulkan profiler ...`
- `... подумай как организовать асинхронное исполнение кернелов ...`
- `да, обнови, и не забудь сохранить итоговый полный план в документацию ...`
- `кроме того зафиксируй этот подход с документированием в AGENTS.md`
- `Implement the plan.`

## Короткий лог действий

- Подготовлен и сохранён подробный план в `docs/vulkan_async_compute_plan.md`.
- Для Vulkan compute path убрано обязательное ожидание после каждого submit и добавлен bounded async launch manager.
- Введены inflight launch records, двухслотовый `rassert`, whole-resource tracking и timestamp-based async stats.
- Добавлены новые Vulkan tests `asyncTinyWriteValueChain` и `asyncAplusbChain`.
- Обновлён `AGENTS.md`: нетривиальные задачи теперь требуют подробного `docs/...` плана и краткого `prompts/...` файла со ссылкой на него.
- Добавлен helper `scripts/profile_vulkan_async_nsys.py` для разборки `nsys` SQLite.

## Короткий лог результатов

- Async compute path компилируется и проходит целевые Vulkan tests.
- На GTX 1650 `asyncTinyWriteValueChain` и `asyncAplusbChain` печатают встроенную async-статистику.
- Descriptor pool и teardown path были доработаны для корректной работы при множестве inflight launches.
- Подробный текущий статус, baseline и profiling procedure зафиксированы в `docs/vulkan_async_compute_plan.md`.
