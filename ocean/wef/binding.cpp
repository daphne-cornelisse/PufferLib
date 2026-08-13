/*
 * Minimal pybind11 wrapper around the WEF C env for Python policy eval.
 *
 * Build:
 *   ./ocean/wef/build_binding.sh
 *
 * Python:
 *   import wef_env
 *   env = wef_env.WefEnv(num_agents=4, episode_length=512, seed=0)
 *   obs = env.reset()                 # (num_agents, OBS_SIZE) float32
 *   obs, rew, term, info = env.step(actions)  # actions (num_agents, 4)
 *   stats = env.consume_log()         # mean metrics since last consume
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Headless software raylib (no display required for step/reset).
#ifndef PLATFORM_MEMORY
#define PLATFORM_MEMORY
#endif
#define PLATFORM PLATFORM_MEMORY

#include "wef.h"

namespace py = pybind11;

namespace {

void dict_set_f(Dict* d, const char* key, double value) {
    dict_set(d, key, value);
}

Dict make_env_dict(const py::kwargs& kwargs) {
    // Defaults match config/wef.ini [env]
    Dict d = {};
    d.name = dict_strdup("env");
    dict_set_f(&d, "num_agents", 4);
    dict_set_f(&d, "min_arena_width", 70);
    dict_set_f(&d, "max_arena_width", 70);
    dict_set_f(&d, "min_arena_height", 70);
    dict_set_f(&d, "max_arena_height", 70);
    dict_set_f(&d, "food_distribution", 1);  // patchy
    dict_set_f(&d, "num_food", 64);
    dict_set_f(&d, "patch_radius", 6);
    dict_set_f(&d, "patch_radius_std", 1.5);
    dict_set_f(&d, "patch_density", 0.001);
    dict_set_f(&d, "electric_field_radius", 15);
    dict_set_f(&d, "reflection_wall_range", 100);
    dict_set_f(&d, "episode_length", 512);

    for (auto& item : kwargs) {
        std::string key = py::str(item.first);
        // Accept both Python names and ini names
        if (key == "seed") {
            continue;  // handled separately
        }
        double val = 0.0;
        if (py::isinstance<py::bool_>(item.second)) {
            val = item.second.cast<bool>() ? 1.0 : 0.0;
        } else if (py::isinstance<py::int_>(item.second)) {
            val = (double)item.second.cast<long>();
        } else if (py::isinstance<py::float_>(item.second)) {
            val = item.second.cast<double>();
        } else {
            throw std::runtime_error("unsupported kwarg type for " + key);
        }
        dict_set_f(&d, key.c_str(), val);
    }
    return d;
}

void free_dict(Dict* d) {
    if (!d) {
        return;
    }
    for (int i = 0; i < d->size; i++) {
        free(d->items[i].str);
        free(d->items[i].values);
    }
    free(d->items);
    free(d->name);
    d->items = nullptr;
    d->name = nullptr;
    d->size = d->cap = 0;
}

}  // namespace

class WefEnv {
public:
    explicit WefEnv(py::kwargs kwargs) {
        unsigned int seed = 0;
        if (kwargs.contains("seed")) {
            seed = (unsigned int)kwargs["seed"].cast<long>();
        }

        Dict d = make_env_dict(kwargs);
        env_ = (Wef*)calloc(1, sizeof(Wef));
        if (!env_) {
            free_dict(&d);
            throw std::bad_alloc();
        }
        env_->rng = seed;
        puf_init(env_, &d);
        free_dict(&d);

        num_agents_ = env_->num_agents;
        obs_.assign((size_t)num_agents_ * OBS_SIZE, 0.0f);
        act_.assign((size_t)num_agents_ * NUM_ATNS, 0.0f);
        rew_.assign((size_t)num_agents_, 0.0f);
        term_.assign((size_t)num_agents_, 0.0f);

        for (int i = 0; i < num_agents_; i++) {
            env_->agents[i].observations = obs_.data() + (size_t)i * OBS_SIZE;
            env_->agents[i].actions = act_.data() + (size_t)i * NUM_ATNS;
            env_->agents[i].rewards = rew_.data() + i;
            env_->agents[i].terminals = term_.data() + i;
            env_->agents[i].action_mask = nullptr;
            env_->agents[i].policy = 0;
        }
        // Zero log accumulators
        memset(&env_->log, 0, sizeof(env_->log));
        puf_reset(env_);
    }

    ~WefEnv() {
        if (env_) {
            puf_close(env_);
            free(env_);
            env_ = nullptr;
        }
    }

    py::array_t<float> reset(py::object seed = py::none()) {
        if (!seed.is_none()) {
            env_->rng = (unsigned int)seed.cast<long>();
        }
        memset(&env_->log, 0, sizeof(env_->log));
        puf_reset(env_);
        return obs_array();
    }

    py::tuple step(py::array_t<float, py::array::c_style | py::array::forcecast> actions) {
        if (actions.ndim() != 2 || actions.shape(0) != num_agents_ ||
            actions.shape(1) != NUM_ATNS) {
            throw std::runtime_error(
                "actions must have shape (num_agents, " + std::to_string(NUM_ATNS) + ")");
        }
        auto buf = actions.unchecked<2>();
        for (int i = 0; i < num_agents_; i++) {
            for (int a = 0; a < NUM_ATNS; a++) {
                act_[(size_t)i * NUM_ATNS + a] = buf(i, a);
            }
        }
        puf_step(env_);

        py::dict info;
        info["tick"] = env_->tick;
        info["food_eaten"] = env_->food_eaten;
        info["episode_return_running"] = env_->episode_return;
        // Snapshot completed-episode accumulators (means if n>0)
        info["log_n"] = env_->log.n;
        if (env_->log.n > 0) {
            info["episode_return"] = env_->log.episode_return / env_->log.n;
            info["episode_length"] = env_->log.episode_length / env_->log.n;
            info["score"] = env_->log.score / env_->log.n;
            info["perf"] = env_->log.perf / env_->log.n;
        }

        return py::make_tuple(obs_array(), rew_array(), term_array(), info);
    }

    py::dict consume_log() {
        py::dict out;
        float n = env_->log.n;
        out["n"] = n;
        if (n > 0) {
            out["episode_return"] = env_->log.episode_return / n;
            out["episode_length"] = env_->log.episode_length / n;
            out["score"] = env_->log.score / n;
            out["perf"] = env_->log.perf / n;
            out["food_eaten_mean"] = env_->log.food_eaten_mean / n;
            out["eod_rate"] = env_->log.eod_rate / n;
            out["collisions_fish"] = env_->log.collisions_fish / n;
            out["bites"] = env_->log.bites / n;
            out["food_per_fish_area"] = env_->log.food_per_fish_area / n;
        } else {
            out["episode_return"] = 0.0f;
            out["episode_length"] = 0.0f;
            out["score"] = 0.0f;
            out["perf"] = 0.0f;
        }
        memset(&env_->log, 0, sizeof(env_->log));
        return out;
    }

    int num_agents() const { return num_agents_; }
    int obs_size() const { return OBS_SIZE; }
    int num_actions() const { return NUM_ATNS; }
    int episode_length() const { return env_->episode_length; }

private:
    py::array_t<float> obs_array() {
        return py::array_t<float>(
            {num_agents_, OBS_SIZE},
            {sizeof(float) * OBS_SIZE, sizeof(float)},
            obs_.data());
    }
    py::array_t<float> rew_array() {
        return py::array_t<float>({num_agents_}, {sizeof(float)}, rew_.data());
    }
    py::array_t<float> term_array() {
        return py::array_t<float>({num_agents_}, {sizeof(float)}, term_.data());
    }

    Wef* env_ = nullptr;
    int num_agents_ = 0;
    std::vector<float> obs_, act_, rew_, term_;
};

PYBIND11_MODULE(wef_env, m) {
    m.doc() = "WEF multi-agent env (PufferLib C port)";
    m.attr("OBS_SIZE") = OBS_SIZE;
    m.attr("NUM_ACTIONS") = NUM_ATNS;
    m.attr("MAX_AGENTS") = MAX_AGENTS;

    py::class_<WefEnv>(m, "WefEnv")
        .def(py::init([](py::kwargs kwargs) {
            return new WefEnv(std::move(kwargs));
        }))
        .def("reset", &WefEnv::reset, py::arg("seed") = py::none())
        .def("step", &WefEnv::step, py::arg("actions"))
        .def("consume_log", &WefEnv::consume_log)
        .def_property_readonly("num_agents", &WefEnv::num_agents)
        .def_property_readonly("obs_size", &WefEnv::obs_size)
        .def_property_readonly("num_actions", &WefEnv::num_actions)
        .def_property_readonly("episode_length", &WefEnv::episode_length);
}
