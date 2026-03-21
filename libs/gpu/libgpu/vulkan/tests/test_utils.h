#pragma once

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <libgpu/context.h>
#include <libgpu/vulkan/engine.h>
#include <libbase/gtest_utils.h>
#include <libbase/runtime_assert.h>

// https://stackoverflow.com/a/61968208
#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
	EXPECT_GE((VAL), (MIN));           \
	EXPECT_LE((VAL), (MAX))

namespace {

std::vector<gpu::Device> enumVKDevices(bool silent=false)
{
	std::vector<gpu::Device> devices = gpu::enumDevices(true, true, silent);

	std::vector<gpu::Device> vk_devices;
	for (auto &device : devices) {
		if (device.supports_vulkan) {
			vk_devices.push_back(device);
		}
	}
	if (!silent) {
		std::cout << "Vulkan supported devices: " << vk_devices.size() << " out of " << devices.size() << std::endl;
	}

	rassert(vk_devices.size() > 0, 364773896);
	return vk_devices;
}

bool isEnabled(const std::string &env_variable_name, bool enabled_by_default)
{
	char *env_variable_value = getenv(env_variable_name.c_str()); // can be disabled with env variable <env_variable_name>=false
	if (env_variable_value) {
		if (env_variable_value == std::string("false")) {
			return false;
		} else if (env_variable_value == std::string("true")) {
			return true;
		} else {
			rassert(false, 556882154, "unrecognized env value (expected: true/false)", env_variable_name + "=" + env_variable_value);
		}
	} else {
		return enabled_by_default;
	}
}

bool isValidationLayersEnabled()
{
	// enable validation layers in Vulkan unit-tests, so we have them evaluated regularly on CI - preventing regressions
	return isEnabled("AVK_ENABLE_VALIDATION_LAYERS", true); // can be disabled with env variable AVK_ENABLE_VALIDATION_LAYERS=false
}

bool isMemoryGuardsEnabled()
{
	return isEnabled("AVK_ENABLE_MEMORY_GUARDS", false);
}

bool isMemoryGuardsChecksAfterKernelsEnabled()
{
	return isEnabled("AVK_ENABLE_MEMORY_GUARDS_CHECKS_AFTER_KERNELS", false);
}

gpu::Context activateVKContext(gpu::Device &device, bool silent=false)
{
	device.supports_opencl = false;
	device.supports_cuda = false;
	rassert(device.supports_vulkan, 771075327231479);

	if (!silent) {
		rassert(device.printInfo(), 7710753277608479);
	}

	gpu::Context gpu_context;
	gpu_context.initVulkan(device.device_id_vulkan);

	gpu_context.setVKValidationLayers(isValidationLayersEnabled());
	gpu_context.setMemoryGuards(isMemoryGuardsEnabled());
	gpu_context.setMemoryGuardsChecksAfterKernels(isMemoryGuardsChecksAfterKernelsEnabled());

	gpu_context.activate();
	return gpu_context;
}

void checkValidationLayerCallback()
{
	std::shared_ptr<avk2::InstanceContext> instance_context = avk2::InstanceContext::getGlobalInstanceContext(isValidationLayersEnabled());
	bool validation_errors_happend = instance_context->isDebugCallbackTriggered();
	if (validation_errors_happend) {
		instance_context->setDebugCallbackTriggered(false); // so that further tests will not fail because of previous test's validation errors
		rassert(!validation_errors_happend, "Validation layer detected a problem! 45124124321312");
	}
}

void checkNumberOfConstructedContexts()
{
	// let's ensure that one/two avk2::InstanceContext were constructed
	// the first one (without any validation layers) should be requested by VulkanEnum::enumDevices
	// and if validation layers are used - the second context also will be constructed
	std::shared_ptr<avk2::InstanceContext> instance_context = avk2::InstanceContext::getGlobalInstanceContext(isValidationLayersEnabled());
	rassert(instance_context->getConstructionIndex() == isValidationLayersEnabled(), 345124124123);
}

void checkPostInvariants()
{
	checkValidationLayerCallback();
	checkNumberOfConstructedContexts();
}

typedef std::mt19937 Random;

template <typename T>
T generate_random_color(Random &r, T min_value, T max_value)
{
	std::uniform_int_distribution<T> random_color(min_value, max_value);
	return random_color(r);
}

struct RemoteJpegSpec {
	const char *filename;
	const char *url;
};

std::string shellQuote(const std::string &value)
{
	std::string quoted = "'";
	for (char c : value) {
		if (c == '\'') {
			quoted += "'\\''";
		} else {
			quoted += c;
		}
	}
	quoted += "'";
	return quoted;
}

bool fileLooksLikeJpeg(const std::filesystem::path &path)
{
	std::ifstream input(path, std::ios::binary);
	if (!input.good()) {
		return false;
	}
	unsigned char soi[2] = {0, 0};
	input.read(reinterpret_cast<char *>(soi), 2);
	return input.gcount() == 2 && soi[0] == 0xFFu && soi[1] == 0xD8u;
}

bool shellCommandExists(const std::string &command)
{
	const std::string probe = "command -v " + shellQuote(command) + " >/dev/null 2>&1";
	return std::system(probe.c_str()) == 0;
}

void downloadFileViaShell(const std::string &url, const std::filesystem::path &output_path)
{
	const std::filesystem::path temp_path = output_path.string() + ".download";
	std::filesystem::remove(temp_path);
	std::filesystem::create_directories(output_path.parent_path());

	std::string command;
	if (shellCommandExists("curl")) {
		command = "curl -L --fail --silent --show-error"
				+ std::string(" --retry 5 --retry-delay 1 --retry-all-errors")
				+ std::string(" -A ")
				+ shellQuote("Mozilla/5.0 CodexJPEGBenchmark/1.0")
				+ " -o "
				+ shellQuote(temp_path.string())
				+ " "
				+ shellQuote(url);
	} else if (shellCommandExists("wget")) {
		command = "wget -q"
				+ std::string(" --tries=6 --waitretry=1")
				+ std::string(" --user-agent=")
				+ shellQuote("Mozilla/5.0 CodexJPEGBenchmark/1.0")
				+ " -O "
				+ shellQuote(temp_path.string())
				+ " "
				+ shellQuote(url);
	} else {
		rassert(false, 2026032119183300001, "Neither curl nor wget is available for JPEG benchmark download");
	}

	std::cout << "Downloading JPEG benchmark image: " << output_path.filename().string()
			  << " from " << url << std::endl;
	const int exit_code = std::system(command.c_str());
	rassert(exit_code == 0, 2026032119183300002, url, output_path.string(), exit_code);
	rassert(fileLooksLikeJpeg(temp_path), 2026032119183300003, url, temp_path.string());
	std::filesystem::rename(temp_path, output_path);
}

std::filesystem::path ensureDefaultRealJpegBenchmarkDir()
{
	namespace fs = std::filesystem;

	const std::array<RemoteJpegSpec, 5> kRemoteJpegs = {{
		{"mp4_first_night_mary_river.jpg", "https://commons.wikimedia.org/wiki/Special:Redirect/file/First%20night%20at%20Mary%20River%20%289103309112%29.jpg"},
		{"k4_ts_2014_12_23_2268.jpg", "https://commons.wikimedia.org/wiki/Special:Redirect/file/TS%202014-12-23-2268%20%2815464308094%29.jpg"},
		{"mp20_namwon_nongak.jpg", "https://commons.wikimedia.org/wiki/Special:Redirect/file/%EB%82%A8%EC%9B%90%EB%86%8D%EC%95%85.jpg"},
		{"mp40_vaz_2109_8000x6000.jpg", "https://commons.wikimedia.org/wiki/Special:Redirect/file/VAZ%202109%208000x6000.jpg"},
		{"mp60_earth_from_orbit_8000x8000.jpg", "https://svs.gsfc.nasa.gov/vis/a010000/a011200/a011268/cover-original.jpg"},
	}};

	fs::path current = fs::current_path();
	fs::path repo_root;
	for (;;) {
		if (fs::exists(current / ".gitignore") && fs::exists(current / "libs")) {
			repo_root = current;
			break;
		}
		if (current == current.root_path()) {
			break;
		}
		current = current.parent_path();
	}
	rassert(!repo_root.empty(), 2026032119183300004, fs::current_path().string());

	const fs::path target_dir = repo_root / ".local_data" / "gpgpu_jpeg_benchmark_real";
	fs::create_directories(target_dir);
	for (const RemoteJpegSpec &spec : kRemoteJpegs) {
		const fs::path local_path = target_dir / spec.filename;
		if (fileLooksLikeJpeg(local_path)) {
			continue;
		}
		downloadFileViaShell(spec.url, local_path);
	}
	return target_dir;
}

std::filesystem::path findOrPrepareJpegBenchmarkDir(const char *env_dir)
{
	namespace fs = std::filesystem;

	if (env_dir != nullptr && std::string(env_dir) != "") {
		return fs::path(env_dir);
	}

	fs::path current = fs::current_path();
	for (;;) {
		const fs::path candidate = current / "data/jpeg_benchmark";
		if (fs::exists(candidate)) {
			return candidate;
		}
		if (current == current.root_path()) {
			break;
		}
		current = current.parent_path();
	}

	return ensureDefaultRealJpegBenchmarkDir();
}

template <>
char generate_random_color<char>(Random &r, char min_value, char max_value)
{
	// this is to workaround MSVC compilation error (i.e. it can't compile std::uniform_int_distribution<char):
	// random(1863): error C2338: invalid template argument for uniform_int_distribution: N4659 29.6.1.1 [rand.req.genl]/1e requires one of short, int, long, long long, unsigned short, unsigned int, unsigned long, or unsigned long long
	// error C2338: note: char, signed char, unsigned char, char8_t, int8_t, and uint8_t are not allowed
	std::uniform_int_distribution<int> random_color(min_value, max_value);
	int res_int = random_color(r);
	rassert(res_int >= min_value && res_int <= max_value, 117565785);
	return (char) res_int;
}

template <>
unsigned char generate_random_color<unsigned char>(Random &r, unsigned char min_value, unsigned char max_value)
{
	// this is to workaround MSVC compilation error (i.e. it can't compile std::uniform_int_distribution<unsigned char):
	// random(1863): error C2338: invalid template argument for uniform_int_distribution: N4659 29.6.1.1 [rand.req.genl]/1e requires one of short, int, long, long long, unsigned short, unsigned int, unsigned long, or unsigned long long
	// error C2338: note: char, signed char, unsigned char, char8_t, int8_t, and uint8_t are not allowed
	std::uniform_int_distribution<int> random_color(min_value, max_value);
	int res_int = random_color(r);
	rassert(res_int >= min_value && res_int <= max_value, 117565785);
	return (unsigned char) res_int;
}

template <>
float generate_random_color<float>(Random &r, float min_value, float max_value)
{
	std::uniform_real_distribution<float> random_color(min_value, max_value);
	return random_color(r);
}

template <>
double generate_random_color<double>(Random &r, double min_value, double max_value)
{
	std::uniform_real_distribution<double> random_color(min_value, max_value);
	return random_color(r);
}

}
