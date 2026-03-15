# План: интерактивный viewer для ambient occlusion

## Постановка
- Использовать уже существующий расчёт ambient occlusion.
- Показать результат в отдельном окне.
- Камера по умолчанию плавно летает вокруг модели.
- При ручном вводе мышью автополёт останавливается и управление перехватывается пользователем.
- Колесо мыши меняет zoom.
- Double-click по геометрии переносит фокус в точку под курсором.
- В углу должен отображаться FPS.
- Решение должно быть простым и работать на Windows/macOS/Linux.

## Варианты

### 1. Существующий compute AO + CImg window + CPU blit
- Плюсы:
  - уже есть в репозитории;
  - без новых зависимостей;
  - кроссплатформенно;
  - достаточно для интерактивного viewer-а и управления камерой.
- Минусы:
  - кадр нужно читать с GPU на CPU для показа в окне;
  - не идеальный путь вывода для high-FPS renderer-а;
  - в headless-терминале окно не откроется без X11/Wayland display.

### 2. GLFW + texture upload/present
- Плюсы:
  - лучшее основание для будущего viewer-а;
  - проще наращивать UI.
- Минусы:
  - новая зависимость;
  - больше platform-specific glue;
  - заметно больший объём работы.

### 3. Полноценный Vulkan swapchain viewer
- Плюсы:
  - самый правильный долгосрочный вариант.
- Минусы:
  - для этой задачи слишком дорогой по времени;
  - требует отдельной подсистемы окна/present/surface.

## Выбранный путь
- Делать вариант 1.
- Viewer будет использовать существующий AO compute pipeline.
- Для показа окна использовать текущую `libs/images` обёртку поверх CImg.
- Для overlay FPS использовать рисование текста прямо в CPU image перед показом.
- Для фокуса по double-click добавить отдельный `depth/t-hit` framebuffer и восстанавливать world-point на CPU через ту же формулу primary ray, что используется в kernels.

## План реализации
1. Добавить в `libs/images` минимальную поддержку mouse wheel.
2. Добавить host-side helper для primary ray и orbit camera math.
3. Расширить AO kernels и host wrappers дополнительным depth framebuffer.
4. Добавить новый executable viewer для сцен из `main_linear_bvh`.
5. Реализовать:
   - autoplay orbit;
   - ручной orbit мышью;
   - wheel zoom;
   - double-click focus по depth;
   - FPS overlay.
6. Добавить unit tests для camera/orbit/picking math.
7. Добавить headless-safe smoke mode:
   - если нет `DISPLAY`/`WAYLAND_DISPLAY`, viewer должен завершаться с понятным сообщением, а тесты не должны зависеть от GUI.

## Ограничения
- В текущей терминальной Ubuntu-сессии `DISPLAY` и `WAYLAND_DISPLAY` не заданы.
- `xvfb-run` в окружении не найден.
- Значит локальный полноценный запуск окна может оказаться недоступен из этой сессии даже если где-то в фоне есть graphical session.
- Это не мешает сделать код, но влияет на локальную проверку окна.

## Что реализовано
- Добавлен новый executable `main_ao_viewer`.
- Viewer сейчас использует Vulkan compute AO kernel и показывает результат через `libs/images`/CImg.
- Камера умеет:
  - autoplay orbit вокруг сцены;
  - ручной orbit левой кнопкой мыши;
  - zoom колесом;
  - double-click focus по `t-hit` depth framebuffer.
- В overlay поверх изображения рисуется FPS и текущий режим камеры.
- В `libs/images` добавлена минимальная поддержка mouse wheel.
- В Vulkan AO shaders добавлен дополнительный depth framebuffer с `tBest` или `-1` для background.
- Добавлены unit tests для host-side camera math и headless smoke path для viewer-а.

## Проверка
- `./build/ao_viewer_test`:
  - `4/4` passed.
- `GPGPU_VISIBLE_DEVICES=3 AVK_ENABLE_VALIDATION_LAYERS=false ./build/main_ao_viewer --headless-smoke --device=0`:
  - passed;
  - на `data/gnome/gnome.ply` получено `1254430 / 4138477` hit pixels.
- `./build/main_ao_viewer --device=0` без display:
  - завершается с понятным сообщением, что нужен graphical session или `--headless-smoke`.

## Важное ограничение текущей версии
- Viewer пока опирается на brute-force Vulkan AO kernel.
- Для больших сцен он честно откажется, потому что LBVH traversal kernels в `src/kernels/vk/ray_tracing_render_using_lbvh.comp` всё ещё не реализованы.
- Поэтому текущий рабочий сценарий — небольшие сцены вроде `data/gnome/gnome.ply`.
