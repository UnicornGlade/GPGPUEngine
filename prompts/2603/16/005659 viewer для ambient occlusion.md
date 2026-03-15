# Задача

- Пользователь попросил сделать простое кроссплатформенное окно для показа уже существующего ambient occlusion renderer-а.
- Нужны autoplay orbit, FPS overlay, ручное управление мышью, zoom колесом и double-click focus по depth.

# Короткий лог

- Изучены текущие варианты: существующий `CImg` windowing, Vulkan render wrapper, AO compute pipeline.
- Выбран простой путь: `AO compute + CImg window + CPU blit`.
- Подробный план записан в [docs/ao_viewer_plan.md](/home/polarnick/codex/GPGPUEngine/docs/ao_viewer_plan.md).
- Реализован `main_ao_viewer` с autoplay orbit, FPS overlay, mouse drag orbit, wheel zoom и double-click focus по depth.
- Добавлен `t-hit` depth framebuffer в Vulkan AO kernels.
- Добавлен `ao_viewer_test`, локально `4/4` passed.
- Headless smoke run на GTX 1650 прошёл.
- Полноценный показ окна из текущей TTY-сессии не проверен, потому что здесь нет `DISPLAY`/`WAYLAND_DISPLAY`.
