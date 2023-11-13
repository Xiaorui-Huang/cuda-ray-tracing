#ifndef ARGP_PAESER_H
#define ARGP_PAESER_H

#include <argp.h>
#ifndef BLOCK_DIM
#define BLOCK_DIM 64
#endif

#define KEY_PPM 1001
#define KEY_BVH 1002

const char *argp_program_version = "CUDA (Real-Timr) Ray Tracing v0.1";
const char *argp_program_bug_address = "hxr.richard@gmail.com";

/* Program documentation. */
static char doc[] = "Real-Time Ray Tracing with CUDA.\n\n"
                    "Involve parallel Ray Tracing and parallel BVH construction. (WIP)\n"
                    "Intend to provide interactive component for ray tracing.\n";

/* A description of the arguments we accept. */
static char args_doc[] =
    "[-f FILE='../data/bunny.json'] [-b SIZE] [-r RESOLUTION] [--ppm] [--no-bvh]";

/*
name: This is the long name of the option, used with two dashes in command-line
    arguments (e.g., --filename). It's a string.

key: This is the single character used for the short form of the option,
    prefixed by a single dash (e.g., -f). It's an int, and you can use ASCII values
    for characters or any unique integer for options that don't have a short form.
    For boolean flags without short forms, like --ppm, we use a unique integer (like
    1001) to identify them in the parse_opt function.

arg: This specifies the name of the argument that the option takes. For options
    that don't require an argument (like boolean flags), this should be set to 0 or
    NULL.

flags: This field is used for additional flags to control the behavior of the
    option. For most cases, this is set to 0. There are special flags like
    OPTION_ARG_OPTIONAL or OPTION_HIDDEN to modify the option's behavior.

doc: This is a description of what the option does. It's used in the
    automatically generated help and usage messages.

group: This is used to group related options in the help output. Options with
    the same group number will be shown together. Usually, 0 is used unless you have
    a specific need to group options.
*/
// clang-format off
static struct argp_option options[] = {
    {"filename", 'f', "FILE", 0, "Path to the Scene JSON config file. Default: ../data/bunny.json"},
    {"blocksize", 'b', "SIZE", 0, "Block size for CUDA processing. Default: 64 x 64"},
    {"resolution", 'r', "RESOLUTION", 0, "Resolution of the output image (e.g. 360, 720, 1080). Default: 720"},
    {"no-bvh", KEY_BVH, 0, 0, "Turn off Ray Tracing with BVH. Default: BVH is ON"}, // Boolean flag for PPM output
    {"ppm", KEY_PPM, 0, 0, "Create .ppm (Portable Pixmap) output. Default: OFF"}, // Boolean flag for PPM output
    {0}};
// clang-format on

struct arguments {
    char *filename = const_cast<char *>("../data/bunny.json"); // Default filename
    int blocksize = BLOCK_DIM;                                 // Default block size
    int resolution = 720;                                      // Default height
    bool no_bvh = false;                                           
    bool ppm = false;                                          // Default PPM output
};

/* Parse a single option. */
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
    struct arguments *arguments = static_cast<struct arguments *>(state->input);

    switch (key) {
    case 'f':
        arguments->filename = arg; // arg is directly assignable to std::string
        break;
    case 'b':
        arguments->blocksize = std::stoi(arg);
        break;
    case 'r':
        arguments->resolution = std::stoi(arg);
        break;
    case KEY_PPM:
        arguments->ppm = true;
        break;
    case KEY_BVH:
        arguments->no_bvh = true;
        break;

    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

/* Our argp parser. */
static struct argp argp = {options, parse_opt, args_doc, doc};

#endif // ARGUMENTS_H