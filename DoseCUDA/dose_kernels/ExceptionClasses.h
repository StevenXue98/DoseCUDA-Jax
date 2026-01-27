#pragma once

#ifndef EXCEPTION_CLASSES_H
#define EXCEPTION_CLASSES_H

#include <stdexcept>


struct Exception: public std::runtime_error {
    enum Type {
        STDLIB,
        CUDA
    } type;

    static void raise(Type tp, const char *fmt, ...);
};


#endif /* EXCEPTION_CLASSES_H */
