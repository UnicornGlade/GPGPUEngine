#include <libbase/runtime_assert.h>
#include <libbase/string_utils.h>
#include <libbase/timer.h>
#include <libgpu/context.h>
#include <libgpu/device.h>
#include <libimages/images.h>
#include <libutils/misc.h>

#include "cpu_helpers/build_bvh_cpu.h"
#include "io/camera_reader.h"
#include "io/scene_reader.h"
#include "kernels/defines.h"
#include "kernels/kernels.h"
#include "viewer/camera_orbit.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>

namespace {

struct Options {
    std::string scene_path = "data/gnome/gnome.ply";
    std::string camera_path;
    size_t device_index = 0;
    bool headless_smoke = false;
};

bool hasDisplay()
{
#if defined(__linux__)
    const char *display = std::getenv("DISPLAY");
    const char *wayland = std::getenv("WAYLAND_DISPLAY");
    return (display != nullptr && display[0] != '\0')
        || (wayland != nullptr && wayland[0] != '\0');
#else
    return true;
#endif
}

Options parseOptions(int argc, char **argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (starts_with(arg, "--scene=")) {
            options.scene_path = arg.substr(std::string("--scene=").size());
        } else if (starts_with(arg, "--camera=")) {
            options.camera_path = arg.substr(std::string("--camera=").size());
        } else if (starts_with(arg, "--device=")) {
            options.device_index = static_cast<size_t>(std::stoull(arg.substr(std::string("--device=").size())));
        } else if (arg == "--headless-smoke") {
            options.headless_smoke = true;
        } else {
            rassert(false, 2026031601400600001, "Unknown argument", arg);
        }
    }
    if (options.camera_path.empty()) {
        std::filesystem::path scene_dir = std::filesystem::path(options.scene_path).parent_path();
        options.camera_path = (scene_dir / "camera.txt").string();
    }
    return options;
}

gpu::Device chooseVulkanDevice(size_t device_index)
{
    std::vector<gpu::Device> devices = gpu::selectAllDevices(ALL_GPUS, true);
    std::vector<gpu::Device> vulkan_devices;
    for (const gpu::Device &device: devices) {
        if (device.supports_vulkan && device.isGPU()) {
            vulkan_devices.push_back(device);
        }
    }
    rassert(!vulkan_devices.empty(), 2026031601400600002, "No Vulkan GPU devices found");
    rassert(device_index < vulkan_devices.size(), 2026031601400600003, device_index, vulkan_devices.size());
    return vulkan_devices[device_index];
}

std::string formatOverlay(double fps, const viewer::OrbitCameraState &camera_state)
{
    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(1);
    out << "FPS: " << fps;
    out << "  dist: " << camera_state.distance;
    if (camera_state.auto_orbit_enabled) {
        out << "  auto";
    } else {
        out << "  manual";
    }
    return out.str();
}

image8u makeDisplayImage(const image32f &ambient_occlusion, double fps, const viewer::OrbitCameraState &camera_state)
{
    image8u out(ambient_occlusion.width(), ambient_occlusion.height(), 3);
    for (size_t j = 0; j < ambient_occlusion.height(); ++j) {
        for (size_t i = 0; i < ambient_occlusion.width(); ++i) {
            float ao = ambient_occlusion.ptr(j)[i];
            float clamped = std::clamp(ao, 0.0f, 1.0f);
            unsigned char v = static_cast<unsigned char>(std::round(clamped * 255.0f));
            out(j, i, 0) = v;
            out(j, i, 1) = v;
            out(j, i, 2) = v;
        }
    }

    drawText(out, 8, 8, formatOverlay(fps, camera_state));
    return out;
}

void renderAmbientOcclusionFrame(
    avk2::KernelSource &kernel,
    unsigned int nfaces,
    const gpu::gpu_mem_32f &vertices_gpu,
    const gpu::gpu_mem_32u &faces_gpu,
    gpu::gpu_mem_32i &framebuffer_face_id_gpu,
    gpu::gpu_mem_32f &framebuffer_ambient_occlusion_gpu,
    gpu::gpu_mem_32f &framebuffer_depth_gpu,
    gpu::shared_device_buffer_typed<CameraViewGPU> &camera_gpu,
    const CameraViewGPU &camera)
{
    camera_gpu.writeN(&camera, 1);
    kernel.exec(
        nfaces,
        gpu::WorkSize(16, 16, camera.K.width, camera.K.height),
        vertices_gpu, faces_gpu,
        framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu, framebuffer_depth_gpu,
        camera_gpu);
}

void runViewer(const Options &options)
{
    rassert(std::filesystem::exists(options.scene_path), 2026031601400600004, options.scene_path);
    rassert(std::filesystem::exists(options.camera_path), 2026031601400600005, options.camera_path);

    gpu::Device device = chooseVulkanDevice(options.device_index);
    std::cout << "Using Vulkan device: " << device.name << std::endl;
    gpu::Context context = activateContext(device, gpu::Context::TypeVulkan);

    SceneGeometry scene = loadScene(options.scene_path);
    rassert(!scene.vertices.empty(), 2026031601400600006);
    rassert(!scene.faces.empty(), 2026031601400600007);
    rassert(scene.faces.size() <= 5000, 2026031601400600008,
        "AO viewer currently uses brute-force Vulkan AO kernel, please use a smaller scene or implement LBVH traversal first",
        scene.faces.size());

    CameraViewGPU base_camera = loadViewState(options.camera_path);
    point3f scene_center = viewer::computeSceneCenter(scene.vertices);
    float scene_radius = viewer::computeSceneRadius(scene.vertices, scene_center);
    viewer::OrbitCameraState orbit = viewer::makeOrbitStateFromCamera(base_camera, scene_center);
    orbit.distance = std::max(orbit.distance, scene_radius * 1.5f);

    const unsigned int width = base_camera.K.width;
    const unsigned int height = base_camera.K.height;
    const unsigned int nfaces = static_cast<unsigned int>(scene.faces.size());
    std::cout << "Scene: " << options.scene_path << "  faces=" << nfaces << "  framebuffer=" << width << "x" << height << std::endl;

    gpu::gpu_mem_32f vertices_gpu(3 * scene.vertices.size());
    gpu::gpu_mem_32u faces_gpu(3 * scene.faces.size());
    gpu::shared_device_buffer_typed<CameraViewGPU> camera_gpu(1);
    gpu::gpu_mem_32i framebuffer_face_id_gpu(width * height);
    gpu::gpu_mem_32f framebuffer_ambient_occlusion_gpu(width * height);
    gpu::gpu_mem_32f framebuffer_depth_gpu(width * height);

    vertices_gpu.writeN(reinterpret_cast<const float*>(scene.vertices.data()), 3 * scene.vertices.size());
    faces_gpu.writeN(reinterpret_cast<const unsigned int*>(scene.faces.data()), 3 * scene.faces.size());

    avk2::KernelSource kernel(avk2::getRTBruteForce());

    CameraViewGPU current_camera = viewer::makeCameraFromOrbit(base_camera, orbit);
    renderAmbientOcclusionFrame(
        kernel, nfaces,
        vertices_gpu, faces_gpu,
        framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu, framebuffer_depth_gpu,
        camera_gpu, current_camera);

    image32f ambient_occlusion = image32f(width, height, 1);
    image32f depth = image32f(width, height, 1);
    framebuffer_ambient_occlusion_gpu.readN(ambient_occlusion.ptr(), width * height);
    framebuffer_depth_gpu.readN(depth.ptr(), width * height);

    if (options.headless_smoke) {
        size_t nhits = 0;
        for (size_t idx = 0; idx < width * height; ++idx) {
            if (depth.ptr()[idx] > 0.0f) {
                ++nhits;
            }
        }
        std::cout << "Headless AO viewer smoke frame rendered: hit_pixels=" << nhits << "/" << (width * height) << std::endl;
        return;
    }

    rassert(hasDisplay(), 2026031601400600009,
        "No graphical display detected. Run from a graphical session or use --headless-smoke for non-window validation.");

    ImageWindow window("AO Viewer");
    window.display(makeDisplayImage(ambient_occlusion, 0.0, orbit));

    using clock = std::chrono::steady_clock;
    auto last_frame_time = clock::now();
    auto last_click_time = clock::time_point::min();
    point2i last_click_pos(-10000, -10000);
    int previous_buttons = 0;
    bool drag_active = false;
    point2i last_mouse_pos(-1, -1);
    double smoothed_fps = 0.0;

    while (!window.isClosed()) {
        keycode_t key = window.wait(1);
        if (key == getEscapeKeyCode()) {
            break;
        }

        auto now = clock::now();
        float dt = std::chrono::duration<float>(now - last_frame_time).count();
        last_frame_time = now;
        if (dt > 0.0f) {
            double fps = 1.0 / dt;
            smoothed_fps = (smoothed_fps == 0.0) ? fps : (0.9 * smoothed_fps + 0.1 * fps);
        }

        int mouse_x = window.getMouseX();
        int mouse_y = window.getMouseY();
        int buttons = window.getMouseClick();
        bool left_pressed = (buttons & MOUSE_LEFT) != 0;
        bool left_pressed_prev = (previous_buttons & MOUSE_LEFT) != 0;

        int wheel = window.getMouseWheel();
        if (wheel != 0) {
            orbit.auto_orbit_enabled = false;
            viewer::zoomOrbit(orbit, std::pow(0.9f, static_cast<float>(wheel)));
            window.resetMouseWheel();
        }

        if (left_pressed && !left_pressed_prev && mouse_x >= 0 && mouse_y >= 0) {
            point2i current_click(mouse_x, mouse_y);
            auto since_last_click_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_click_time).count();
            point2i delta_click = current_click - last_click_pos;
            bool is_double_click = last_click_time != clock::time_point::min()
                && since_last_click_ms <= 350
                && std::abs(delta_click.x) <= 4
                && std::abs(delta_click.y) <= 4;
            last_click_time = now;
            last_click_pos = current_click;

            if (is_double_click
                && mouse_x < static_cast<int>(width)
                && mouse_y < static_cast<int>(height)
                && depth.ptr(mouse_y)[mouse_x] > 0.0f) {
                orbit.auto_orbit_enabled = false;
                point3f new_focus = viewer::reconstructWorldPoint(current_camera, float(mouse_x) + 0.5f, float(mouse_y) + 0.5f, depth.ptr(mouse_y)[mouse_x]);
                orbit = viewer::makeOrbitStateFromCamera(current_camera, new_focus);
                orbit.auto_orbit_enabled = false;
            }
            drag_active = true;
            last_mouse_pos = current_click;
        } else if (!left_pressed) {
            drag_active = false;
        }

        if (drag_active && left_pressed && mouse_x >= 0 && mouse_y >= 0 && last_mouse_pos.x >= 0) {
            int dx = mouse_x - last_mouse_pos.x;
            int dy = mouse_y - last_mouse_pos.y;
            if (dx != 0 || dy != 0) {
                orbit.auto_orbit_enabled = false;
                viewer::orbitAroundFocus(orbit, -0.01f * dx, -0.01f * dy);
            }
            last_mouse_pos = point2i(mouse_x, mouse_y);
        }

        if (orbit.auto_orbit_enabled) {
            viewer::orbitAroundFocus(orbit, dt * 0.4f, 0.0f);
        }

        previous_buttons = buttons;
        current_camera = viewer::makeCameraFromOrbit(base_camera, orbit);

        renderAmbientOcclusionFrame(
            kernel, nfaces,
            vertices_gpu, faces_gpu,
            framebuffer_face_id_gpu, framebuffer_ambient_occlusion_gpu, framebuffer_depth_gpu,
            camera_gpu, current_camera);

        framebuffer_ambient_occlusion_gpu.readN(ambient_occlusion.ptr(), width * height);
        framebuffer_depth_gpu.readN(depth.ptr(), width * height);

        image8u display = makeDisplayImage(ambient_occlusion, smoothed_fps, orbit);
        window.display(display);
    }
}

} // namespace

int main(int argc, char **argv)
{
    try {
        runViewer(parseOptions(argc, argv));
    } catch (std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
