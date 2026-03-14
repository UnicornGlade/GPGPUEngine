# Задача

Пользователь попросил:
- перенести в этот репозиторий кодовую базу, похожую на `GPGPUTasks2025`;
- убедиться, что проект собирается, тесты проходят и CI настроен;
- добавить структурированный `LEADERBOARD.md` по победителям задач;
- затем начать перенос лучших решений в `main`, начиная с быстрого `radix sort` на трех API.

# Действия

- Проверено текущее состояние репозитория: он почти пустой, есть только `AGENTS.md`, коммитов нет.
- Определен remote `origin`: `git@github.com:UnicornGlade/GPGPUEngine.git`.
- Начат импорт upstream-контекста и подготовка базовой структуры истории.

# Результаты

- Создан task log для этой задачи.
- В локальный git импортированы upstream-ветки `task00`...`task09`.
- `main` переведен на кодовую базу `task08` как на общий фундамент библиотеки.
- В `main` завендорен `VulkanMemoryAllocator` header для воспроизводимой сборки без machine-specific SDK include path.
- В `main` добавлен `main_radix_sort` и перенесены radix sort реализации:
  - CUDA из победившего PR `#545`;
  - OpenCL из PR `#568`;
  - Vulkan-порт сделан поверх того же host-side пайплайна.
- Локально подтверждено:
  - сборка `main` без CUDA проходит;
  - сборка `main` с `GPU_CUDA_SUPPORT=ON` проходит;
  - `libgpu_test` проходит полностью;
  - `main_aplusb` работает;
  - `main_radix_sort` работает через OpenCL, Vulkan и CUDA на локальной RTX 4090.
- Добавлен `LEADERBOARD.md` по задачам 03-08.
- Добавлена GitHub Actions matrix для `Linux` / `macOS` / `Windows`.
- По логам CI найдены и исправлены platform-specific проблемы:
  - на macOS поправлена установка зависимостей и линковка `OpenMP`;
  - на Linux добавлен fallback для OpenCL runtime и отключено требование Vulkan validation layers в unit-tests;
  - на Windows исправлена кросс-платформенная декларация `setenv` для Vulkan engine под `_WIN32`.
- Локально перепроверена целевая сборка таргетов `libgpu_test`, `main_aplusb`, `main_linear_bvh`, `main_radix_sort`.
- Backend-тесты теперь корректно делают `skip`, если на раннере нет Vulkan/OpenCL devices, вместо ложного падения.
- На `macOS` и `Windows` в CI включена сборка `libgpu_test` и запуск smoke subset.
- На `Windows` для Vulkan smoke tests добавлена software implementation через `SwiftShader`.
- После первого прогона новой схемы найдены и исправлены два точечных дефекта:
  - `GTEST_SKIP` был вынесен из общего test helper header, чтобы не ломать production targets, которые его include-ят;
  - линковка `OpenMP` переведена на target-based подключение через `OpenMP::OpenMP_CXX`, чтобы `libgpu_test` корректно линковался на macOS.
- Следующий прогон подтвердил:
  - `macOS` runtime smoke tests проходят;
  - на `Windows` осталась отдельная проблема согласованности toolchain для `gtest` (`MinGW` binary vs `vcpkg`/MSVC library), после чего workflow переведен на `mingw-w64-x86_64-gtest`.
- После перевода Windows на `MSVC` сборка `libgpu_test` прошла, и следующая ошибка оказалась уже только в пути запуска smoke tests:
  - для multi-config Visual Studio binary лежит в подкаталоге `RelWithDebInfo`, а не рядом с `build/libs/gpu`.
- Следующий прогон показал, что строка из matrix не интерполирует PowerShell-выражение `${env:BUILD_TYPE}`.
- Для Windows smoke test path использован явный путь `.\\RelWithDebInfo\\libgpu_test.exe`.
- По следующему запросу пользователя начат отдельный проход по OpenCL на hosted runners:
  - на `Windows` добавлена установка официального Intel CPU OpenCL runtime;
  - на `macOS` дополнительных install step пока не добавляется, так как `opencl.*` уже запускаются и могут пройти, если runtime/device присутствует на runner.
- По следующему шагу для `macOS` в CI добавлены:
  - `brew install pocl`
  - `brew install clinfo`
  - явный `clinfo` probe step перед сборкой и тестами
- После анализа результата на `macOS` выяснилось:
  - `pocl` установился, но `clinfo` видел только Apple OpenCL platform и падал при запросе devices;
  - для следующей попытки в probe и macOS smoke tests явно прокинут `OCL_ICD_FILENAMES=$POCL_ROOT/lib/libpocl.dylib`.
