#include "pushworld.h"
#include <Python.h>

static PyObject* vec_full_map(PyObject* self, PyObject* args);
static PyObject* vec_num_puzzles(PyObject* self, PyObject* args);
static PyObject* vec_set_puzzle_indices(PyObject* self, PyObject* args);
static PyObject* vec_last_solved(PyObject* self, PyObject* args);
static PyObject* vec_get_coverage_goal(PyObject* self, PyObject* args);
static PyObject* vec_get_coverage_agent(PyObject* self, PyObject* args);
static PyObject* vec_set_coverage_enabled(PyObject* self, PyObject* args);
static PyObject* vec_clear_coverage(PyObject* self, PyObject* args);

#define MY_METHODS \
    {"vec_full_map", vec_full_map, METH_VARARGS, "Get full map tiles"}, \
    {"vec_num_puzzles", vec_num_puzzles, METH_VARARGS, "Get puzzle count"}, \
    {"vec_set_puzzle_indices", vec_set_puzzle_indices, METH_VARARGS, "Set puzzle indices"}, \
    {"vec_last_solved", vec_last_solved, METH_VARARGS, "Get last solved flags"}, \
    {"vec_get_coverage_goal", vec_get_coverage_goal, METH_VARARGS, "Get goal-object coverage counts"}, \
    {"vec_get_coverage_agent", vec_get_coverage_agent, METH_VARARGS, "Get agent coverage counts"}, \
    {"vec_set_coverage_enabled", vec_set_coverage_enabled, METH_VARARGS, "Enable coverage tracking for env"}, \
    {"vec_clear_coverage", vec_clear_coverage, METH_VARARGS, "Clear coverage counts for env"}
#define Env PushWorld
#define MY_SHARED
#include "../env_binding.h"

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* path_obj = PyDict_GetItemString(kwargs, "puzzle_dir");
    if (path_obj == NULL || !PyUnicode_Check(path_obj)) {
        PyErr_SetString(PyExc_TypeError, "puzzle_dir must be a string");
        return NULL;
    }

    const char* puzzle_dir = PyUnicode_AsUTF8(path_obj);
    if (puzzle_dir == NULL) {
        PyErr_SetString(PyExc_ValueError, "Invalid puzzle_dir");
        return NULL;
    }

    int* levels = NULL;
    int num_levels = 0;
    PyObject* levels_obj = PyDict_GetItemString(kwargs, "levels");
    if (levels_obj != NULL && levels_obj != Py_None) {
        if (!PySequence_Check(levels_obj)) {
            PyErr_SetString(PyExc_TypeError, "levels must be a list of ints");
            return NULL;
        }
        Py_ssize_t n = PySequence_Size(levels_obj);
        if (n < 0) {
            PyErr_SetString(PyExc_ValueError, "Invalid levels sequence");
            return NULL;
        }
        levels = (int*)calloc((size_t)n, sizeof(int));
        if (!levels) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate levels");
            return NULL;
        }
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject* item = PySequence_GetItem(levels_obj, i);
            if (!item || !PyLong_Check(item)) {
                Py_XDECREF(item);
                free(levels);
                PyErr_SetString(PyExc_TypeError, "levels must be a list of ints");
                return NULL;
            }
            long val = PyLong_AsLong(item);
            Py_DECREF(item);
            levels[i] = (int)val;
        }
        num_levels = (int)n;
    }

    int max_puzzles = 0;
    PyObject* max_obj = PyDict_GetItemString(kwargs, "max_puzzles");
    if (max_obj != NULL && max_obj != Py_None) {
        if (PyLong_Check(max_obj)) {
            max_puzzles = (int)PyLong_AsLong(max_obj);
        } else if (PyFloat_Check(max_obj)) {
            max_puzzles = (int)PyFloat_AsDouble(max_obj);
        } else {
            if (levels) free(levels);
            PyErr_SetString(PyExc_TypeError, "max_puzzles must be an int");
            return NULL;
        }
    }

    PuzzleSet* puzzles = pw_load_puzzles(puzzle_dir, levels, num_levels, max_puzzles);
    if (levels) {
        free(levels);
    }
    return PyLong_FromVoidPtr(puzzles);
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    int max_episode_length = 0;
    if (PyDict_GetItemString(kwargs, "max_episode_length") != NULL) {
        max_episode_length = (int)unpack(kwargs, "max_episode_length");
    } else {
        max_episode_length = (int)unpack(kwargs, "horizon");
    }
    int vision = unpack(kwargs, "vision");
    double count_based_reward_coef = 0.0;
    if (PyDict_GetItemString(kwargs, "count_based_reward_coef") != NULL) {
        count_based_reward_coef = unpack(kwargs, "count_based_reward_coef");
    }
    int count_based_global = 1;
    if (PyDict_GetItemString(kwargs, "count_based_global") != NULL) {
        count_based_global = (int)unpack(kwargs, "count_based_global");
    }

    PyObject* handle_obj = PyDict_GetItemString(kwargs, "state");
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "state handle must be an integer");
        return 1;
    }

    PuzzleSet* puzzles = (PuzzleSet*)PyLong_AsVoidPtr(handle_obj);
    if (!puzzles) {
        PyErr_SetString(PyExc_ValueError, "Invalid puzzle handle");
        return 1;
    }

    init_pushworld(env, puzzles, max_episode_length, vision);
    env->count_based_reward_coef = (float)count_based_reward_coef;
    env->count_based_global = count_based_global != 0;
    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "no_op_rate", log->no_op_rate);
    assign_to_dict(dict, "solved", log->solved);
    return 0;
}

static PyObject* vec_full_map(PyObject* self, PyObject* args) {
    int argc = PyTuple_Size(args);
    if (argc < 1 || argc > 2) {
        PyErr_SetString(PyExc_TypeError, "vec_full_map requires vec handle and optional env_id");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    int env_id = 0;
    if (argc == 2) {
        PyObject* env_id_arg = PyTuple_GetItem(args, 1);
        if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type)) {
            PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
            return NULL;
        }
        env_id = (int)PyLong_AsLong(env_id_arg);
    }
    if (env_id < 0 || env_id >= vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "env_id out of range");
        return NULL;
    }

    Env* env = vec->envs[env_id];
    npy_intp dims[2] = {env->height, env->width};
    PyObject* arr = PyArray_SimpleNew(2, dims, NPY_INT32);
    if (!arr) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate full map array");
        return NULL;
    }

    int* data = (int*)PyArray_DATA((PyArrayObject*)arr);
    int width = env->width;
    int height = env->height;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            data[y * width + x] = pw_tile_at(env, x, y);
        }
    }

    return arr;
}

static PyObject* vec_num_puzzles(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_num_puzzles requires vec handle");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    Env* env = vec->envs[0];
    return PyLong_FromLong(env->puzzles->num_puzzles);
}

static PyObject* vec_set_puzzle_indices(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 2) {
        PyErr_SetString(PyExc_TypeError, "vec_set_puzzle_indices requires vec handle and indices");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject* indices_obj = PyTuple_GetItem(args, 1);
    if (!PySequence_Check(indices_obj)) {
        PyErr_SetString(PyExc_TypeError, "indices must be a sequence of ints");
        return NULL;
    }

    Py_ssize_t n = PySequence_Size(indices_obj);
    if (n != vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "indices length must match num_envs");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject* item = PySequence_GetItem(indices_obj, i);
        if (!item || !PyLong_Check(item)) {
            Py_XDECREF(item);
            PyErr_SetString(PyExc_TypeError, "indices must be ints");
            return NULL;
        }
        long idx = PyLong_AsLong(item);
        Py_DECREF(item);

        Env* env = vec->envs[i];
        if (idx >= env->puzzles->num_puzzles) {
            PyErr_SetString(PyExc_ValueError, "puzzle index out of range");
            return NULL;
        }
        env->forced_puzzle_idx = (idx < 0) ? -1 : (int)idx;
        c_reset(env);
    }

    Py_RETURN_NONE;
}

static PyObject* vec_last_solved(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 1) {
        PyErr_SetString(PyExc_TypeError, "vec_last_solved requires vec handle");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    npy_intp dims[1] = {vec->num_envs};
    PyObject* arr = PyArray_SimpleNew(1, dims, NPY_UINT8);
    if (!arr) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate last_solved array");
        return NULL;
    }

    unsigned char* data = (unsigned char*)PyArray_DATA((PyArrayObject*)arr);
    for (int i = 0; i < vec->num_envs; i++) {
        data[i] = vec->envs[i]->last_solved;
    }

    return arr;
}

static PyObject* vec_get_coverage_goal(PyObject* self, PyObject* args) {
    int argc = PyTuple_Size(args);
    if (argc < 1 || argc > 2) {
        PyErr_SetString(PyExc_TypeError, "vec_get_coverage_goal requires vec handle and optional env_id");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    int env_id = 0;
    if (argc == 2) {
        PyObject* env_id_arg = PyTuple_GetItem(args, 1);
        if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type)) {
            PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
            return NULL;
        }
        env_id = (int)PyLong_AsLong(env_id_arg);
    }
    if (env_id < 0 || env_id >= vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "env_id out of range");
        return NULL;
    }

    Env* env = vec->envs[env_id];
    npy_intp dims[2] = {env->height, env->width};
    PyObject* arr = PyArray_SimpleNew(2, dims, NPY_UINT32);
    if (!arr) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate coverage array");
        return NULL;
    }

    uint32_t* data = (uint32_t*)PyArray_DATA((PyArrayObject*)arr);
    int stride = env->coverage_stride;
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            data[y * env->width + x] = env->coverage_goal[y * stride + x];
        }
    }

    return arr;
}

static PyObject* vec_get_coverage_agent(PyObject* self, PyObject* args) {
    int argc = PyTuple_Size(args);
    if (argc < 1 || argc > 2) {
        PyErr_SetString(PyExc_TypeError, "vec_get_coverage_agent requires vec handle and optional env_id");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    int env_id = 0;
    if (argc == 2) {
        PyObject* env_id_arg = PyTuple_GetItem(args, 1);
        if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type)) {
            PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
            return NULL;
        }
        env_id = (int)PyLong_AsLong(env_id_arg);
    }
    if (env_id < 0 || env_id >= vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "env_id out of range");
        return NULL;
    }

    Env* env = vec->envs[env_id];
    npy_intp dims[2] = {env->height, env->width};
    PyObject* arr = PyArray_SimpleNew(2, dims, NPY_UINT32);
    if (!arr) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate coverage array");
        return NULL;
    }

    uint32_t* data = (uint32_t*)PyArray_DATA((PyArrayObject*)arr);
    int stride = env->coverage_stride;
    for (int y = 0; y < env->height; y++) {
        for (int x = 0; x < env->width; x++) {
            data[y * env->width + x] = env->coverage_agent[y * stride + x];
        }
    }

    return arr;
}

static PyObject* vec_set_coverage_enabled(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 3) {
        PyErr_SetString(PyExc_TypeError, "vec_set_coverage_enabled requires vec handle, env_id, enabled");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject* env_id_arg = PyTuple_GetItem(args, 1);
    PyObject* enabled_arg = PyTuple_GetItem(args, 2);
    if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type) || !PyObject_TypeCheck(enabled_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_id and enabled must be integers");
        return NULL;
    }

    int env_id = (int)PyLong_AsLong(env_id_arg);
    int enabled = (int)PyLong_AsLong(enabled_arg);
    if (env_id < 0 || env_id >= vec->num_envs) {
        PyErr_SetString(PyExc_ValueError, "env_id out of range");
        return NULL;
    }

    Env* env = vec->envs[env_id];
    env->coverage_enabled = enabled != 0;
    Py_RETURN_NONE;
}

static PyObject* vec_clear_coverage(PyObject* self, PyObject* args) {
    int argc = PyTuple_Size(args);
    if (argc < 1 || argc > 2) {
        PyErr_SetString(PyExc_TypeError, "vec_clear_coverage requires vec handle and optional env_id");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    int env_id = -1;
    if (argc == 2) {
        PyObject* env_id_arg = PyTuple_GetItem(args, 1);
        if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type)) {
            PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
            return NULL;
        }
        env_id = (int)PyLong_AsLong(env_id_arg);
    }

    int start = 0;
    int end = vec->num_envs;
    if (env_id >= 0) {
        if (env_id >= vec->num_envs) {
            PyErr_SetString(PyExc_ValueError, "env_id out of range");
            return NULL;
        }
        start = env_id;
        end = env_id + 1;
    }

    for (int i = start; i < end; i++) {
        Env* env = vec->envs[i];
        size_t cover_size = (size_t)env->max_width * (size_t)env->max_height;
        memset(env->coverage_goal, 0, cover_size * sizeof(uint32_t));
        memset(env->coverage_agent, 0, cover_size * sizeof(uint32_t));
    }

    Py_RETURN_NONE;
}
