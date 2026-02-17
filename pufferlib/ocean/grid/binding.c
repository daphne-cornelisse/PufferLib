#include "grid.h"

#define Env Grid 
#define MY_SHARED
#define MY_GET_COVERAGE_COUNTS
#define MY_VEC_GET_COVERAGE_COUNTS
#include "../env_binding.h"

static PyObject* my_shared(PyObject* self, PyObject* args, PyObject* kwargs) {
    int num_maps = unpack(kwargs, "num_maps");
    int max_size = unpack(kwargs, "max_size");
    float difficulty = unpack(kwargs, "difficulty");
    int size = unpack(kwargs, "size");
    int seed = unpack(kwargs, "seed");
    int vision = unpack(kwargs, "vision");
    int speed = unpack(kwargs, "speed");
    float count_based_ri_c = unpack(kwargs, "count_based_ri_c");
    State* levels = calloc(num_maps, sizeof(State));

    if (max_size <= 5) {
        PyErr_SetString(PyExc_ValueError, "max_size must be  > 5");
        return NULL;
    }

    // Temporary env used to gen maps
    Grid env;
    env.max_size = max_size;
    env.num_maps = num_maps;
    init_grid(&env);

    srand(seed);
    int start_seed = 0; // Temp: make it deterministic
    for (int i = 0; i < num_maps; i++) {
        int sz = size;
        if (size == -1) {
            //sz = 5 + (rand() % (max_size-5));
            sz = max_size;
        }
        
        // Ensure odd size
        if (sz % 2 == 0) {
            sz -= 1;
        }
        
        create_maze_level(&env, sz, sz, difficulty, start_seed + i);

        // Compute total_traversable cells for each level
        //printf("GRID\n");
        env.total_traversable = 0;
        for (int r = 0; r < env.height; r++) {
            for (int c = 0; c < env.width; c++) {
                int adr = grid_offset(&env, r, c);
                if (env.grid[adr] == EMPTY || env.grid[adr] == GOAL || env.grid[adr] == AGENT) {
                    env.total_traversable++;
                    //printf("%3d ", 1);
                }
                // else {
                //     //printf("%3d ", 0);  
                // }
            }
            //printf("\n");
        }

        init_state(&levels[i], max_size, 1);
        get_state(&env, &levels[i]);
        //printf("Seed: %d\n", start_seed + i);
        printf("Generated maze %d/%d of size %dx%d, traversable tiles: %d\n", i+1, num_maps, sz, sz, env.total_traversable);
    }

    return PyLong_FromVoidPtr(levels);
}

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->max_size = unpack(kwargs, "max_size");
    env->num_maps = unpack(kwargs, "num_maps");
    env->count_based_ri_c = unpack(kwargs, "count_based_ri_c");
    env->vision = unpack(kwargs, "vision");
    env->speed = unpack(kwargs, "speed");
    int horizon = unpack(kwargs, "horizon");
    if (horizon > 1) {
        env->horizon = horizon;
    } else {
        env->horizon = 2*env->max_size*env->max_size;
    }
    init_grid(env);

    PyObject* handle_obj = PyDict_GetItemString(kwargs, "state");
    if (!PyObject_TypeCheck(handle_obj, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "state handle must be an integer");
        return 1;
    }

    State* levels = (State*)PyLong_AsVoidPtr(handle_obj);
    if (!levels) {
        PyErr_SetString(PyExc_ValueError, "Invalid state handle");
        return 1;
    }

    env->levels = levels;
    return 0;
}

static PyObject* get_coverage_counts(PyObject* self, PyObject* args) {
    // Unpack the environment handle from args
    Env* env = unpack_env(args);
    if (!env) {
        return NULL;
    }
    
    // Create 2D numpy array
    npy_intp dims[2] = {env->height, env->width};
    PyObject* array = PyArray_SimpleNew(2, dims, NPY_INT32);
    
    if (!array) {
        return NULL;
    }
    
    // Copy data from C array to numpy array
    int* data = (int*)PyArray_DATA((PyArrayObject*)array);
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            int adr = grid_offset(env, r, c);
            data[r * env->width + c] = env->coverage_counts[adr];
        }
    }
    
    return array;
}

static PyObject* vec_get_coverage_counts(PyObject* self, PyObject* args) {
    // Unpack the vecenv handle
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    
    // Get the first environment
    Env* env = vec->envs[0];
    
    // Create 2D numpy array
    npy_intp dims[2] = {env->height, env->width};
    PyObject* array = PyArray_SimpleNew(2, dims, NPY_INT32);
    
    if (!array) {
        return NULL;
    }
    
    // Copy data from C array to numpy array
    int* data = (int*)PyArray_DATA((PyArrayObject*)array);
    for (int r = 0; r < env->height; r++) {
        for (int c = 0; c < env->width; c++) {
            int adr = grid_offset(env, r, c);
            data[r * env->width + c] = env->coverage_counts[adr];
        }
    }
    
    return array;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "unique_visited", log->unique_visited);
    assign_to_dict(dict, "cum_coverage", log->cum_coverage);
    return 0;
}