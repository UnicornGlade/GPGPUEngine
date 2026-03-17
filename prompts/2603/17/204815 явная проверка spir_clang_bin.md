# Задача

- Пользователь: добавить явную проверку и понятное сообщение для причины несборки `libgpu_test`.

# Краткий лог

- Воспроизведена сборка `libgpu_test` в `cmake-build-relwithdebinfo`.
- Выявлены две причины:
  - вне `vcvars64` у `cl.exe` нет стандартных include paths;
  - в Windows-конфигурации отсутствует `SPIR_CLANG_BIN`, из-за чего OpenCL header generation падал на `-x is not recognized`.
- В `libs/gpu/CMakeLists.txt` добавлена явная `FATAL_ERROR`-проверка для `SPIR_CLANG_BIN` на Windows.

# Результат

- При пустом или неверном `SPIR_CLANG_BIN` CMake теперь падает сразу с понятным сообщением и примером корректного пути к `clang.exe`.
